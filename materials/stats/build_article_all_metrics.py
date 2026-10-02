#!/usr/bin/env python3
"""Build descriptive article tables for every planning-quality metric.

The input series is identical to ``build_article_results.py``.  Estimates are
computed in two stages: runs are averaged within each planning problem and the
reported point estimate is the mean across problems.  The 95% intervals are a
non-parametric bootstrap over problems, so repeated calls for one problem are
not treated as 300 unrelated benchmark instances.

Conditional metrics retain their natural analysis sets: conditional
reachability uses executable plans, while plan length and optimality ratio use
only plans that reached the goal.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_article_results import (
    DATA_ROOT,
    DEFAULT_OUTPUT,
    MODEL_LABELS,
    MODEL_SOURCES,
    ORDER_KEYS,
    ORDER_LABELS_RU,
    SERIES_CONFIGS,
    RUNS,
    binary_tests,
    cached_recovered_rows,
    load_rows as load_primary_rows,
    problems_with_changed_order,
    order_directory,
    output_model_name,
    write_csv,
)
from checker import build_metrics
from experiment_config import PROBLEM_IDS, REFERENCE_PLAN_ACTION_COUNTS_BY_ID
from materials.stats.run_reference_order_statistics import bootstrap_mean_ci, holm, stable_seed
from materials.stats.run_statistical_tests import sign_flip_permutation_p


DEFAULT_ALL_METRICS_OUTPUT = DEFAULT_OUTPUT / "all_metrics"

METRICS: tuple[dict[str, Any], ...] = (
    {
        "key": "executability",
        "label_ru": "Исполнимость",
        "subset_ru": "Все запуски",
        "format": "percent",
    },
    {
        "key": "reachability",
        "label_ru": "Достижение цели",
        "subset_ru": "Все запуски",
        "format": "percent",
    },
    {
        "key": "conditional_reachability",
        "label_ru": "Условное достижение цели",
        "subset_ru": "Только исполнимые планы",
        "format": "percent",
    },
    {
        "key": "optimality_ratio",
        "label_ru": "Отношение длины к эталону",
        "subset_ru": "Только планы, достигшие цели",
        "format": "ratio",
    },
    {
        "key": "plan_length",
        "label_ru": "Длина плана",
        "subset_ru": "Только планы, достигшие цели",
        "format": "number",
    },
    {
        "key": "execution_progress",
        "label_ru": "Прогресс исполнения",
        "subset_ru": "Запуски с вычислимым прогрессом",
        "format": "percent",
    },
    {
        "key": "first_failure_step",
        "label_ru": "Первый ошибочный шаг",
        "subset_ru": "Ошибки исполнения с известным шагом",
        "format": "number",
    },
    {
        "key": "non_executable_failure",
        "label_ru": "Доля ошибок разбора",
        "subset_ru": "Неисполнимые планы без тайм-аутов валидатора",
        "format": "percent",
    },
)
METRIC_BY_KEY = {metric["key"]: metric for metric in METRICS}
SECONDARY_METRIC_KEYS = tuple(
    metric["key"] for metric in METRICS
    if metric["key"] not in {"executability", "reachability"}
)


def _load_run_metrics(
    *, model: str, prefix: str | None, problem: str, order_key: str,
    condition_key: str, run: int
) -> tuple[dict[str, Any], dict[str, Any], str]:
    problem_dir = DATA_ROOT / problem
    order_name = order_directory(prefix, order_key)
    order_dir = problem_dir / order_name
    run_dir = order_dir / str(run) / output_model_name(model)
    result_path = run_dir / "llm_result.json"
    plan_path = run_dir / "llm.plan"

    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics") or {}
        strict = metrics.get("strict") or {}
        legacy = metrics.get("legacy") or {}
        source = "llm_result"
    elif plan_path.is_file():
        try:
            metrics = build_metrics(
                order_dir / "domain.pddl",
                problem_dir / f"{problem}.pddl",
                plan_path,
                problem_dir / f"{problem}.plan",
            )
            strict = metrics.get("strict") or {}
            legacy = metrics.get("legacy") or {}
            source = "recovered_plan_only"
        except FileNotFoundError:
            cached = cached_recovered_rows().get((model, problem, condition_key, run))
            if cached is None or int(cached["reachability"]) != 1:
                raise
            plan_length = int(cached["plan_length"])
            strict = {
                "executability": True,
                "reachability": True,
                "plan_length": plan_length,
                "execution_progress": 1.0,
                "first_failure_step": None,
                "non_executable_failure": None,
            }
            legacy = {
                "optimality_ratio": plan_length / REFERENCE_PLAN_ACTION_COUNTS_BY_ID[problem]
            }
            source = "cached_recovered_plan_only"
    else:
        raise FileNotFoundError(f"missing result and plan: {run_dir}")

    if not strict:
        raise RuntimeError(f"missing strict metrics: {result_path}")
    return strict, legacy, source


def load_metric_rows() -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    recovered = 0
    for series in SERIES_CONFIGS:
        for model in series["model_sources"]:
            for problem in PROBLEM_IDS:
                for order_key in series["orders"]:
                    source_spec = series["order_sources"][order_key]
                    for run in range(1, RUNS + 1):
                        strict, legacy, source = _load_run_metrics(
                            model=model, prefix=source_spec["prefixes"][model], problem=problem,
                            order_key=source_spec["raw_order"], condition_key=order_key, run=run,
                        )
                        recovered += int(source != "llm_result")
                        reachable = bool(strict.get("reachability"))
                        executable = bool(strict.get("executability"))
                        failure = strict.get("non_executable_failure")
                        rows.append(
                            {
                                "series": series["key"],
                                "series_label": series["label"],
                                "model": model,
                                "model_label": MODEL_LABELS[model],
                                "problem": problem,
                                "order": order_key,
                                "order_label": series["order_labels"][order_key],
                                "run": run,
                                "executability": int(executable),
                                "reachability": int(reachable),
                                "conditional_reachability": int(reachable) if executable else None,
                                "optimality_ratio": legacy.get("optimality_ratio") if reachable else None,
                                "plan_length": strict.get("plan_length") if reachable else None,
                                "execution_progress": strict.get("execution_progress"),
                                "first_failure_step": strict.get("first_failure_step"),
                                "non_executable_failure": (
                                    1 if failure == "parse_error"
                                    else 0 if failure == "state_execution_error"
                                    else None
                                ),
                            }
                        )

    expected = sum(
        len(series["model_sources"]) * len(PROBLEM_IDS) * len(series["orders"]) * RUNS
        for series in SERIES_CONFIGS
    )
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, loaded {len(rows)}")
    return rows, recovered


def summarize_metrics(
    rows: list[dict[str, Any]], bootstrap_samples: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["series"], row["model"], row["order"], row["problem"])].append(row)

    summaries: list[dict[str, Any]] = []
    task_means_output: list[dict[str, Any]] = []
    for series in SERIES_CONFIGS:
        for model in series["model_sources"]:
            for order_key in series["orders"]:
                for metric in METRICS:
                    metric_key = metric["key"]
                    task_means: list[float] = []
                    observations = 0
                    for problem in PROBLEM_IDS:
                        values = [
                            float(row[metric_key])
                            for row in grouped[(series["key"], model, order_key, problem)]
                            if row[metric_key] is not None
                        ]
                        if not values:
                            continue
                        task_mean = sum(values) / len(values)
                        task_means.append(task_mean)
                        observations += len(values)
                        task_means_output.append(
                            {
                                "series": series["key"], "series_label": series["label"],
                                "model": model, "model_label": MODEL_LABELS[model],
                                "problem": problem, "order": order_key,
                                "order_label": series["order_labels"][order_key],
                                "metric": metric_key, "metric_label": metric["label_ru"],
                                "n_observations": len(values), "task_mean": task_mean,
                            }
                        )
                    if not task_means:
                        mean = ci_low = ci_high = None
                    else:
                        mean = sum(task_means) / len(task_means)
                        ci_low, ci_high = bootstrap_mean_ci(
                            task_means,
                            seed=stable_seed(series["key"], model, order_key, metric_key, "all_metrics"),
                            samples=bootstrap_samples,
                        )
                    summaries.append(
                        {
                            "series": series["key"], "series_label": series["label"],
                            "model": model, "model_label": MODEL_LABELS[model],
                            "order": order_key, "order_label": series["order_labels"][order_key],
                            "metric": metric_key, "metric_label": metric["label_ru"],
                            "analysis_subset": metric["subset_ru"],
                            "n_observations": observations, "n_tasks": len(task_means),
                            "task_mean": mean, "ci95_low_task_bootstrap": ci_low,
                            "ci95_high_task_bootstrap": ci_high,
                        }
                    )
    return summaries, task_means_output


def _format_value(metric_key: str, value: float | None) -> str:
    if value is None:
        return "—"
    style = METRIC_BY_KEY[metric_key]["format"]
    if style == "percent":
        return f"{100 * value:.1f}%"
    if style == "ratio":
        return f"{value:.3f}"
    return f"{value:.1f}"


def write_metric_tables(output: Path, summaries: list[dict[str, Any]]) -> None:
    for series in SERIES_CONFIGS:
        tables_dir = output / "tables" / series["key"]
        tables_dir.mkdir(parents=True, exist_ok=True)
        index = {(r["model"], r["order"], r["metric"]): r for r in summaries if r["series"] == series["key"]}
        for metric in METRICS:
            metric_key = metric["key"]
            orders = series["orders"]
            lines = [f"# {metric['label_ru']}: {series['label']}", "", f"Подвыборка: {metric['subset_ru']}. В ячейках: среднее по задачам [95% bootstrap CI]; `n` — число задач/наблюдений.", "", "| Модель | " + " | ".join(series["order_labels"][order] for order in orders) + " |", "| --- | " + " | ".join("---:" for _ in orders) + " |"]
            for model in series["model_sources"]:
                cells = []
                for order_key in orders:
                    row = index[(model, order_key, metric_key)]
                    point, low, high = (_format_value(metric_key, row[key]) for key in ("task_mean", "ci95_low_task_bootstrap", "ci95_high_task_bootstrap"))
                    cells.append(f"{point} [{low}; {high}], n={row['n_tasks']}/{row['n_observations']}")
                lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
            (tables_dir / f"table_{metric_key}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_definitions(output: Path) -> None:
    lines = [
        "# Определения метрик",
        "",
        "Все точки — средние сначала внутри задачи, затем между задачами. "
        "Интервалы — 95% bootstrap CI по задачам.",
        "",
        "| Метрика | Анализируемая подвыборка |",
        "| --- | --- |",
    ]
    for metric in METRICS:
        lines.append(f"| {metric['label_ru']} | {metric['subset_ru']} |")
    (output / "metric_definitions.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def secondary_significance_tests(
    task_means: list[dict[str, Any]],
    bootstrap_samples: int,
    *,
    baseline_override: str | None = None,
) -> list[dict[str, Any]]:
    index = {
        (row["series"], row["model"], row["metric"], row["order"], row["problem"]):
        float(row["task_mean"])
        for row in task_means
    }
    output: list[dict[str, Any]] = []
    for series in SERIES_CONFIGS:
        baseline = baseline_override or series["baseline"]
        if baseline not in series["orders"]:
            raise ValueError(f"unknown baseline {baseline!r} for series {series['key']!r}")
        for model in series["model_sources"]:
            for metric_key in SECONDARY_METRIC_KEYS:
                family: list[dict[str, Any]] = []
                for compared in (order for order in series["orders"] if order != baseline):
                    differences: list[float] = []
                    baseline_values: list[float] = []
                    compared_values: list[float] = []
                    for problem in problems_with_changed_order(series, model, baseline, compared):
                        baseline_value = index.get((series["key"], model, metric_key, baseline, problem))
                        compared_value = index.get((series["key"], model, metric_key, compared, problem))
                        if baseline_value is None or compared_value is None:
                            continue
                        baseline_values.append(baseline_value)
                        compared_values.append(compared_value)
                        differences.append(compared_value - baseline_value)
                    p_value = sign_flip_permutation_p(
                        differences,
                        seed=stable_seed(series["key"], model, metric_key, compared, "sign_flip"),
                    )
                    low, high = bootstrap_mean_ci(
                        differences,
                        seed=stable_seed(series["key"], model, metric_key, compared, "difference_ci"),
                        samples=bootstrap_samples,
                    )
                    family.append({
                        "series": series["key"],
                        "series_label": series["label"],
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "metric": metric_key,
                        "metric_label": METRIC_BY_KEY[metric_key]["label_ru"],
                        "analysis_subset": METRIC_BY_KEY[metric_key]["subset_ru"],
                        "baseline_order": baseline,
                        "baseline_order_label": series["order_labels"][baseline],
                        "compared_order": compared,
                        "compared_order_label": series["order_labels"][compared],
                        "tasks_in_test": len(differences),
                        "baseline_mean": sum(baseline_values) / len(baseline_values) if baseline_values else None,
                        "compared_mean": sum(compared_values) / len(compared_values) if compared_values else None,
                        "mean_difference": sum(differences) / len(differences) if differences else None,
                        "difference_ci95_low": low,
                        "difference_ci95_high": high,
                        "p_value": p_value,
                        "test": "paired problem-level sign-flip permutation",
                    })
                valid = [record for record in family if record["p_value"] is not None]
                adjusted = holm([float(record["p_value"]) for record in valid])
                adjusted_by_order = {
                    record["compared_order"]: value
                    for record, value in zip(valid, adjusted)
                }
                for record in family:
                    adjusted_p = adjusted_by_order.get(record["compared_order"])
                    record["p_value_holm_metric_family"] = adjusted_p
                    record["significant_holm_0_05"] = adjusted_p is not None and adjusted_p < 0.05
                output.extend(family)
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_significant_results(
    article_output: Path,
    binary_test_rows: list[dict[str, Any]],
    secondary_tests: list[dict[str, Any]],
    *,
    file_suffix: str = "",
    baseline_label: str = "Исходный",
) -> None:
    records: list[dict[str, Any]] = []
    binary_labels = {
        "reachability": "Достижение цели",
        "executability": "Исполнимость",
    }
    for row in binary_test_rows:
        significant = row["significant_holm_0_05"]
        if isinstance(significant, str):
            significant = significant.lower() == "true"
        if not significant:
            continue
        records.append({
            "model": row["model_label"],
            "metric": binary_labels[row["metric"]],
            "comparison": row["compared_order_label"],
            "tasks": int(row["tasks_in_test"]),
            "baseline": float(row["baseline_rate"]),
            "compared": float(row["compared_rate"]),
            "difference": float(row["risk_difference"]),
            "format": "percent",
            "test": "CMH, задача как страта",
            "p_holm": float(row["p_value_holm_model_family"]),
            "status": "основная" if row["metric"] == "reachability" else "вторичная",
        })
    for row in secondary_tests:
        if not row["significant_holm_0_05"]:
            continue
        records.append({
            "model": row["model_label"],
            "metric": row["metric_label"],
            "comparison": row["compared_order_label"],
            "tasks": row["tasks_in_test"],
            "baseline": row["baseline_mean"],
            "compared": row["compared_mean"],
            "difference": row["mean_difference"],
            "format": METRIC_BY_KEY[row["metric"]]["format"],
            "test": "парный sign-flip по задачам",
            "p_holm": row["p_value_holm_metric_family"],
            "status": "дополнительная",
        })

    write_csv(article_output / f"significant_results{file_suffix}.csv", records)
    lines = [
        "# Статистически значимые различия",
        "",
        f"Базовый порядок: **{baseline_label}**. В таблицу включены только сравнения "
        "с `p < 0,05` после поправки Холма, "
        "применённой отдельно к девяти порядкам внутри каждой модели и метрики.",
        "",
        f"| Модель | Метрика | Статус | Порядок | Задач | {baseline_label} | Сравниваемый | Δ | Тест | p Холма |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for record in records:
        metric_key = next(
            (key for key, metric in METRIC_BY_KEY.items() if metric["label_ru"] == record["metric"]),
            None,
        )
        if record["format"] == "percent":
            baseline = f"{100 * record['baseline']:.1f}%"
            compared = f"{100 * record['compared']:.1f}%"
            difference = f"{100 * record['difference']:+.1f} п.п."
        elif record["format"] == "ratio":
            baseline = f"{record['baseline']:.3f}"
            compared = f"{record['compared']:.3f}"
            difference = f"{record['difference']:+.3f}"
        else:
            baseline = f"{record['baseline']:.1f}"
            compared = f"{record['compared']:.1f}"
            difference = f"{record['difference']:+.1f}"
        p_value = record["p_holm"]
        formatted_p = f"{p_value:.2e}" if p_value < 0.0001 else f"{p_value:.5f}".rstrip("0").rstrip(".")
        lines.append(
            f"| {record['model']} | {record['metric']} | {record['status']} | "
            f"{record['comparison']} | {record['tasks']} | {baseline} | {compared} | "
            f"{difference} | {record['test']} | {formatted_p} |"
        )
    (article_output / f"table_significant_results{file_suffix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all-metric article tables.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ALL_METRICS_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")

    rows, recovered = load_metric_rows()
    summaries, task_means = summarize_metrics(rows, args.bootstrap_samples)
    secondary_tests = secondary_significance_tests(task_means, args.bootstrap_samples)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_metrics_summary.csv", summaries)
    write_csv(output / "all_metrics_task_means.csv", task_means)
    write_csv(output / "secondary_significance_tests.csv", secondary_tests)
    write_metric_tables(output, summaries)
    write_definitions(output)
    canonical_binary_tests = _read_csv(output.parent / "binary_cmh_tests.csv")
    write_significant_results(
        output.parent,
        canonical_binary_tests,
        secondary_tests,
        baseline_label="Исходный",
    )

    primary_rows, _ = load_primary_rows()
    plan_front_binary_tests = binary_tests(
        primary_rows,
        args.bootstrap_samples,
        baseline_override="plan_front",
    )
    plan_front_secondary_tests = secondary_significance_tests(
        task_means,
        args.bootstrap_samples,
        baseline_override="plan_front",
    )
    write_csv(output.parent / "binary_cmh_tests_plan_front.csv", plan_front_binary_tests)
    write_csv(output / "secondary_significance_tests_plan_front.csv", plan_front_secondary_tests)
    write_significant_results(
        output.parent,
        plan_front_binary_tests,
        plan_front_secondary_tests,
        file_suffix="_plan_front",
        baseline_label="План в начале",
    )

    manifest = {
        "series": [
            {"key": series["key"], "orders": list(series["orders"]), "baseline": series["baseline"]}
            for series in SERIES_CONFIGS
        ],
        "problems": list(PROBLEM_IDS),
        "runs_per_problem_order": RUNS,
        "input_rows": len(rows),
        "recovered_plan_only_rows": recovered,
        "metrics": [metric["key"] for metric in METRICS],
        "method": (
            "Mean within each task, then mean across tasks; 95% non-parametric "
            "bootstrap interval over tasks. Conditional metrics exclude undefined runs. "
            "Secondary comparisons use paired problem-level sign-flip permutation tests "
            "with Holm correction across nine orders separately within each model and metric."
        ),
    }
    (output / "all_metrics_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Analyzed {len(rows)} rows ({recovered} recovered from saved plans); "
        f"wrote all-metric tables to {output}"
    )


if __name__ == "__main__":
    main()
