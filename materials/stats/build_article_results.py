#!/usr/bin/env python3
"""Build one article analysis from every saved action-order condition."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checker import build_metrics
from experiment_config import PROBLEM_IDS
from shuffler import _extract_action_blocks
from materials.stats.run_reference_order_statistics import (
    bootstrap_mean_ci,
    cmh,
    holm,
    stable_seed,
    wilson_interval,
)


DATA_ROOT = PROJECT_ROOT / "materials" / "logistics" / "alpha"
DEFAULT_OUTPUT = PROJECT_ROOT / "materials" / "stats" / "article_final"
RUNS = 15

MODEL_LABELS = {
    "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
    "gpt-oss-120b": "GPT-OSS-120B",
    "nemotron-3-super": "Nemotron-3-Super",
}
# Public compatibility alias: prefixes of the reference-aligned family.
MODEL_SOURCES = {
    "deepseek/deepseek-v4-flash": "reference_deepinfra_20260921",
    "gpt-oss-120b": "reference_final_20260917",
    "nemotron-3-super": "reference_final_20260917",
}

SERIES_CONFIGS: tuple[dict[str, Any], ...] = ({
    "key": "combined",
    "label": "Все варианты порядка схем действий",
    "baseline": "canonical",
    "orders": (
        "canonical", "canonical_disp_1", "canonical_disp_2", "canonical_disp_3",
        "plan_front", "plan_back", "plan_scatter",
        "reference_disp_1", "reference_disp_2", "reference_reverse",
    ),
    "order_labels": {
        "canonical": "Исходный", "canonical_disp_1": "Исх. смещение 1/3",
        "canonical_disp_2": "Исх. смещение 2/3", "canonical_disp_3": "Исх. макс. смещение",
        "plan_front": "План в начале", "plan_back": "План в конце", "plan_scatter": "План рассеян",
        "reference_disp_1": "План в начале: смещение 1/3",
        "reference_disp_2": "План в начале: смещение 2/3", "reference_reverse": "План в начале: обратный",
    },
    "order_sources": {
        "canonical": {"raw_order": "canonical", "prefixes": {model: None for model in MODEL_LABELS}},
        "canonical_disp_1": {"raw_order": "disp_1", "prefixes": {model: None for model in MODEL_LABELS}},
        "canonical_disp_2": {"raw_order": "disp_2", "prefixes": {model: None for model in MODEL_LABELS}},
        "canonical_disp_3": {"raw_order": "disp_3", "prefixes": {model: None for model in MODEL_LABELS}},
        "plan_front": {"raw_order": "plan_front", "prefixes": {model: None for model in MODEL_LABELS}},
        "plan_back": {"raw_order": "plan_back", "prefixes": {model: None for model in MODEL_LABELS}},
        "plan_scatter": {"raw_order": "plan_scatter", "prefixes": {model: None for model in MODEL_LABELS}},
        "reference_disp_1": {"raw_order": "disp_1", "prefixes": MODEL_SOURCES},
        "reference_disp_2": {"raw_order": "disp_2", "prefixes": MODEL_SOURCES},
        "reference_reverse": {"raw_order": "reverse", "prefixes": MODEL_SOURCES},
    },
    "model_sources": {model: None for model in MODEL_LABELS},
},)
ORDER_KEYS = SERIES_CONFIGS[0]["orders"]
ORDER_LABELS_RU = SERIES_CONFIGS[0]["order_labels"]
BINARY_METRICS = ("reachability", "executability")


def output_model_name(model: str) -> str:
    return model.split("/")[-1].replace(":", "-").replace(".", "-")


def order_directory(prefix: str | None, order_key: str) -> str:
    if prefix is None:
        return order_key
    return prefix if order_key == "reference" else f"{prefix}_{order_key}"


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError(f"cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def cached_recovered_rows() -> dict[tuple[str, str, str, int], dict[str, Any]]:
    """Reuse prior strict VAL values only when the validator binary is absent.

    Two DeepSeek plans were preserved without their result JSON.  Their strict
    VAL outcomes were already recorded by the earlier article build.  Keeping
    this narrow fallback makes the combined table reproducible on machines
    that inspect artefacts but do not have VAL installed; a local VAL binary
    remains the preferred source whenever available.
    """
    path = DEFAULT_OUTPUT / "analysis_rows.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        cached: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        for row in csv.DictReader(handle):
            previous_series = row.get("series", "reference")
            if previous_series == "reference":
                condition = "reference" if row["order"] == "reference" else f"reference_{row['order']}"
            elif previous_series == "combined" and row["order"].startswith("reference"):
                condition = row["order"]
            else:
                continue
            cached[(row["model"], row["problem"], condition, int(row["run"]))] = row
        return cached


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    cached_rows = cached_recovered_rows()
    for series in SERIES_CONFIGS:
        for model in series["model_sources"]:
            for problem in PROBLEM_IDS:
                problem_dir = DATA_ROOT / problem
                for order_key in series["orders"]:
                    source = series["order_sources"][order_key]
                    order_dir = problem_dir / order_directory(source["prefixes"][model], source["raw_order"])
                    for run in range(1, RUNS + 1):
                        run_dir = order_dir / str(run) / output_model_name(model)
                        result_path = run_dir / "llm_result.json"
                        plan_path = run_dir / "llm.plan"
                        artifact_source = "llm_result"
                        if result_path.is_file():
                            payload = json.loads(result_path.read_text(encoding="utf-8"))
                            strict = ((payload.get("metrics") or {}).get("strict") or {})
                        elif plan_path.is_file():
                            try:
                                strict = build_metrics(
                                    order_dir / "domain.pddl", problem_dir / f"{problem}.pddl",
                                    plan_path, problem_dir / f"{problem}.plan",
                                )["strict"]
                                artifact_source = "recovered_plan_only"
                            except FileNotFoundError:
                                cached = cached_rows.get((model, problem, order_key, run))
                                if cached is None:
                                    raise
                                strict = {
                                    "reachability": int(cached["reachability"]),
                                    "executability": int(cached["executability"]),
                                    "plan_length": cached["plan_length"] or None,
                                }
                                artifact_source = "cached_recovered_plan_only"
                        else:
                            missing.append(str(result_path))
                            continue
                        if not strict:
                            missing.append(f"{result_path} (missing strict VAL metrics)")
                            continue
                        rows.append({
                            "series": series["key"], "series_label": series["label"],
                            "model": model, "model_label": MODEL_LABELS[model],
                            "problem": problem, "order": order_key,
                            "order_label": series["order_labels"][order_key], "run": run,
                            "reachability": int(bool(strict.get("reachability"))),
                            "executability": int(bool(strict.get("executability"))),
                            "plan_length": strict.get("plan_length") if strict.get("reachability") else None,
                        })
                        artifacts.append({
                            "series": series["key"], "model": model, "problem": problem,
                            "order": order_key, "run": run, "artifact_source": artifact_source,
                            "result_path": str(result_path.relative_to(PROJECT_ROOT)),
                            "plan_path": str(plan_path.relative_to(PROJECT_ROOT)),
                        })
    if missing:
        raise RuntimeError("article series is incomplete ({} cells):\n{}".format(len(missing), "\n".join(missing[:12])))
    expected = sum(len(s["model_sources"]) * len(PROBLEM_IDS) * len(s["orders"]) * RUNS for s in SERIES_CONFIGS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, loaded {len(rows)}")
    return rows, artifacts


def grouped_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["series"], row["model"], row["problem"], row["order"])].append(row)
    return grouped


def problems_with_changed_order(
    series: dict[str, Any], model: str, baseline: str, compared: str
) -> list[str]:
    """Return tasks where the compared domain is an actual reordering.

    A transformation may degenerate to the baseline order for a particular
    task. Such a task contains no action-order intervention and is therefore
    omitted from that pairwise effect test.
    """
    changed: list[str] = []
    for problem in PROBLEM_IDS:
        problem_dir = DATA_ROOT / problem
        signatures = []
        for order_key in (baseline, compared):
            source = series["order_sources"][order_key]
            domain_path = problem_dir / order_directory(
                source["prefixes"][model], source["raw_order"]
            ) / "domain.pddl"
            signatures.append(tuple(_extract_action_blocks(domain_path.read_text(encoding="utf-8"))[3]))
        if signatures[0] != signatures[1]:
            changed.append(problem)
    return changed


def descriptive_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = grouped_rows(rows)
    summary: list[dict[str, Any]] = []
    task_rates: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for series in SERIES_CONFIGS:
        for model in series["model_sources"]:
            for order_key in series["orders"]:
                selected = [r for r in rows if r["series"] == series["key"] and r["model"] == model and r["order"] == order_key]
                base = {"series": series["key"], "series_label": series["label"], "model": model, "model_label": MODEL_LABELS[model], "order": order_key, "order_label": series["order_labels"][order_key]}
                reached = sum(r["reachability"] for r in selected)
                executable = sum(r["executability"] for r in selected)
                outcomes.append({**base, "reached": reached, "executable_without_goal": executable - reached, "non_executable": len(selected) - executable, "total": len(selected)})
                for metric in BINARY_METRICS:
                    successes = sum(r[metric] for r in selected)
                    low, high = wilson_interval(successes, len(selected))
                    summary.append({**base, "metric": metric, "successes": successes, "total": len(selected), "rate": successes / len(selected), "ci95_low_wilson": low, "ci95_high_wilson": high})
                    for problem in PROBLEM_IDS:
                        task = grouped[(series["key"], model, problem, order_key)]
                        task_successes = sum(r[metric] for r in task)
                        task_rates.append({**base, "problem": problem, "metric": metric, "successes": task_successes, "total": len(task), "rate": task_successes / len(task)})
    return summary, task_rates, outcomes


def binary_tests(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    *,
    baseline_override: str | None = None,
) -> list[dict[str, Any]]:
    grouped = grouped_rows(rows)
    output: list[dict[str, Any]] = []
    for series in SERIES_CONFIGS:
        baseline = baseline_override or series["baseline"]
        if baseline not in series["orders"]:
            raise ValueError(f"unknown baseline {baseline!r} for series {series['key']!r}")
        for model in series["model_sources"]:
            for metric in BINARY_METRICS:
                family: list[dict[str, Any]] = []
                for compared in (order for order in series["orders"] if order != baseline):
                    comparison_problems = problems_with_changed_order(series, model, baseline, compared)
                    strata: list[tuple[int, int, int, int]] = []
                    differences: list[float] = []
                    directions = {"positive": 0, "negative": 0, "zero": 0}
                    for problem in comparison_problems:
                        baseline_rows = grouped[(series["key"], model, problem, baseline)]
                        compared_rows = grouped[(series["key"], model, problem, compared)]
                        baseline_successes = sum(r[metric] for r in baseline_rows)
                        compared_successes = sum(r[metric] for r in compared_rows)
                        strata.append((compared_successes, len(compared_rows) - compared_successes, baseline_successes, len(baseline_rows) - baseline_successes))
                        difference = compared_successes / len(compared_rows) - baseline_successes / len(baseline_rows)
                        differences.append(difference)
                        directions["positive" if difference > 0 else "negative" if difference < 0 else "zero"] += 1
                    low, high = bootstrap_mean_ci(differences, seed=stable_seed(series["key"], model, metric, compared), samples=bootstrap_samples)
                    family.append({
                        "series": series["key"], "series_label": series["label"], "model": model, "model_label": MODEL_LABELS[model], "metric": metric,
                        "baseline_order": baseline, "baseline_order_label": series["order_labels"][baseline], "compared_order": compared, "compared_order_label": series["order_labels"][compared],
                        "tasks_in_test": len(comparison_problems), **cmh(strata), "task_mean_risk_difference": sum(differences) / len(differences), "task_difference_ci95_low": low, "task_difference_ci95_high": high,
                        "tasks_compared_better": directions["positive"], "tasks_baseline_better": directions["negative"], "tasks_tied": directions["zero"],
                    })
                for record, adjusted_p in zip(family, holm([float(record["p_value"]) for record in family])):
                    record["p_value_holm_model_family"] = adjusted_p
                    record["significant_holm_0_05"] = adjusted_p < 0.05
                output.extend(family)
    return output


def conditional_plan_lengths(rows: list[dict[str, Any]], bootstrap_samples: int) -> list[dict[str, Any]]:
    grouped = grouped_rows(rows)
    output: list[dict[str, Any]] = []
    for series in SERIES_CONFIGS:
        baseline_order = series["baseline"]
        for model in series["model_sources"]:
            for compared in (order for order in series["orders"] if order != baseline_order):
                differences: list[float] = []
                for problem in problems_with_changed_order(series, model, baseline_order, compared):
                    baseline = [float(r["plan_length"]) for r in grouped[(series["key"], model, problem, baseline_order)] if r["plan_length"] is not None]
                    alternative = [float(r["plan_length"]) for r in grouped[(series["key"], model, problem, compared)] if r["plan_length"] is not None]
                    if baseline and alternative:
                        differences.append(sum(alternative) / len(alternative) - sum(baseline) / len(baseline))
                low, high = bootstrap_mean_ci(differences, seed=stable_seed(series["key"], model, compared, "article_length"), samples=bootstrap_samples)
                output.append({
                    "series": series["key"], "series_label": series["label"], "model": model, "model_label": MODEL_LABELS[model],
                    "baseline_order": baseline_order, "compared_order": compared, "compared_order_label": series["order_labels"][compared],
                    "tasks_with_reachable_plans_in_both_orders": len(differences), "task_mean_plan_length_difference": sum(differences) / len(differences) if differences else None,
                    "task_difference_ci95_low": low, "task_difference_ci95_high": high,
                    "interpretation": "Conditional on reachability in both orders; exploratory, not an unconditional plan-quality effect.",
                })
    return output


def format_p(value: float) -> str:
    return f"{value:.2e}" if value < 0.0001 else f"{value:.5f}".rstrip("0").rstrip(".")


def write_series_tables(output: Path, summary: list[dict[str, Any]], tests: list[dict[str, Any]]) -> None:
    tables_dir = output / "series_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for series in SERIES_CONFIGS:
        series_summary = [r for r in summary if r["series"] == series["key"]]
        index = {(r["model"], r["order"], r["metric"]): r for r in series_summary}
        orders = series["orders"]
        lines = [f"# {series['label']}", "", "В ячейках: `достижение цели / исполнимость`; по 300 запусков на комбинацию.", "", "| Модель | " + " | ".join(series["order_labels"][o] for o in orders) + " |", "| --- | " + " | ".join("---:" for _ in orders) + " |"]
        for model in series["model_sources"]:
            cells = []
            for order in orders:
                reach, execute = index[(model, order, "reachability")], index[(model, order, "executability")]
                cells.append(f"{100 * reach['rate']:.1f}% / {100 * execute['rate']:.1f}%")
            lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
        (tables_dir / f"table_summary_{series['key']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for metric, title in (("reachability", "Сравнения вероятности достижения цели"), ("executability", "Сравнения исполнимости")):
            selected = [r for r in tests if r["series"] == series["key"] and r["metric"] == metric]
            family_size = len(series["orders"]) - 1
            table = [f"# {title}: {series['label']}", "", f"CMH-тест со стратификацией по задаче; поправка Холма отдельно по {family_size} сравнениям для каждой метрики внутри модели. Задачи, в которых преобразование не изменило порядок, исключены из соответствующего сравнения.", "", "| Модель | Сравнение | Задач | Базовый | Сравниваемый | Δ, п.п. | Общий OR | p Холма |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for r in selected:
                table.append("| {model} | {order} | {tasks} | {baseline:.1f}% | {compared:.1f}% | {difference:+.1f} | {odds:.3g} | {p} |".format(model=r["model_label"], order=r["compared_order_label"], tasks=r["tasks_in_test"], baseline=100 * r["baseline_rate"], compared=100 * r["compared_rate"], difference=100 * r["risk_difference"], odds=r["common_odds_ratio"], p=format_p(r["p_value_holm_model_family"])))
            (tables_dir / f"table_{metric}_{series['key']}.md").write_text("\n".join(table) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build article tables from all action-order series.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    rows, artifacts = load_rows()
    summary, task_rates, outcomes = descriptive_tables(rows)
    tests = binary_tests(rows, args.bootstrap_samples)
    lengths = conditional_plan_lengths(rows, args.bootstrap_samples)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "analysis_rows.csv", rows)
    write_csv(output / "artifact_provenance.csv", artifacts)
    write_csv(output / "summary_rates.csv", summary)
    write_csv(output / "task_rates.csv", task_rates)
    write_csv(output / "outcome_composition.csv", outcomes)
    write_csv(output / "binary_cmh_tests.csv", tests)
    write_csv(output / "conditional_plan_length.csv", lengths)
    write_series_tables(output, summary, tests)
    recovered = [a for a in artifacts if a["artifact_source"] != "llm_result"]
    manifest = {
        "series": [{"key": s["key"], "label": s["label"], "baseline": s["baseline"], "orders": list(s["orders"]), "order_sources": s["order_sources"]} for s in SERIES_CONFIGS],
        "problems": list(PROBLEM_IDS), "runs_per_problem_order": RUNS, "input_rows": len(rows), "recovered_plan_only_rows": recovered,
        "methods": {"primary_outcome": "strict VAL reachability", "binary_test": "Cochran-Mantel-Haenszel, problem as stratum, independent API calls", "multiple_testing": "Holm separately within every model and metric across the nine order comparisons", "descriptive_intervals": "95% Wilson intervals", "plan_length": "conditional exploratory task-level summary among reachable plans"},
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Analyzed {len(rows)} rows across {len(SERIES_CONFIGS)} series ({len(recovered)} recovered from saved plans); wrote article tables to {output}")


if __name__ == "__main__":
    main()
