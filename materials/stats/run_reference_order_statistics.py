#!/usr/bin/env python3
"""Statistical analysis for a reference-plan-order experiment series.

This script intentionally does not reuse the legacy ``canonical`` analysis.
Each PDDL problem is a CMH stratum; stochastic API calls are independent and
their numeric run labels are *not* paired between action orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import MODEL_NAMES, PROBLEM_IDS


DATA_ROOT = PROJECT_ROOT / "materials" / "logistics" / "alpha"
BINARY_METRICS = ("reachability", "executability")


def output_model_name(model: str) -> str:
    return model.split("/")[-1].replace(":", "-").replace(".", "-")


def orders_for(prefix: str) -> tuple[str, str, str, str]:
    return (prefix, f"{prefix}_disp_1", f"{prefix}_disp_2", f"{prefix}_reverse")


def load_rows(orders: tuple[str, ...], runs: int) -> list[dict[str, Any]]:
    """Load a complete balanced series or fail instead of silently imputing 0."""

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for problem in PROBLEM_IDS:
        for order in orders:
            for run in range(1, runs + 1):
                for model in MODEL_NAMES:
                    result_path = (
                        DATA_ROOT / problem / order / str(run) / output_model_name(model) / "llm_result.json"
                    )
                    if not result_path.exists():
                        missing.append(str(result_path))
                        continue
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    strict = ((payload.get("metrics") or {}).get("strict") or {})
                    if not strict:
                        missing.append(f"{result_path} (missing VAL metrics)")
                        continue
                    rows.append(
                        {
                            "model": model,
                            "problem": problem,
                            "order": order,
                            "run": run,
                            "reachability": int(bool(strict.get("reachability"))),
                            "executability": int(bool(strict.get("executability"))),
                            # Length is deliberately conditional on reachability.
                            "plan_length": strict.get("plan_length") if strict.get("reachability") else None,
                        }
                    )
    if missing:
        preview = "\n".join(missing[:10])
        suffix = "\n..." if len(missing) > 10 else ""
        raise RuntimeError(f"series is incomplete ({len(missing)} missing cells):\n{preview}{suffix}")
    return rows


def wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_mean_ci(values: list[float], *, seed: int, samples: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def stable_seed(*tokens: str) -> int:
    """Stable across Python processes (unlike the built-in hash())."""

    value = 0
    for token in tokens:
        for character in token:
            value = (value * 131 + ord(character)) & 0xFFFFFFFF
    return value


def cmh(strata: list[tuple[int, int, int, int]]) -> dict[str, float | int | None]:
    """CMH for independent 2x2 tables: compared success/fail, baseline success/fail."""

    oe = variance = numerator = denominator = 0.0
    base_s = base_n = comp_s = comp_n = used = 0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n <= 1 or a + b == 0 or c + d == 0:
            continue
        compared_n, baseline_n, success_n, failure_n = a + b, c + d, a + c, b + d
        expected = compared_n * success_n / n
        oe += a - expected
        variance += compared_n * baseline_n * success_n * failure_n / (n * n * (n - 1))
        numerator += a * d / n
        denominator += b * c / n
        comp_s += a
        comp_n += compared_n
        base_s += c
        base_n += baseline_n
        used += 1
    statistic = oe * oe / variance if variance else 0.0
    base_rate = base_s / base_n if base_n else None
    comp_rate = comp_s / comp_n if comp_n else None
    return {
        "n_strata": used,
        "baseline_successes": base_s,
        "baseline_total": base_n,
        "baseline_rate": base_rate,
        "compared_successes": comp_s,
        "compared_total": comp_n,
        "compared_rate": comp_rate,
        "risk_difference": comp_rate - base_rate if base_rate is not None and comp_rate is not None else None,
        "common_odds_ratio": numerator / denominator if denominator else (math.inf if numerator else None),
        "cmh_chi_square": statistic,
        "p_value": math.erfc(math.sqrt(statistic / 2)),
    }


def holm(p_values: list[float]) -> list[float]:
    ordering = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(ordering):
        candidate = min(1.0, p_values[index] * (len(p_values) - rank))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["problem"], row["order"])].append(row)
    return grouped


def binary_results(
    rows: list[dict[str, Any]], baseline: str, orders: tuple[str, ...], bootstrap_samples: int
) -> list[dict[str, Any]]:
    grouped = group_rows(rows)
    compared_orders = [order for order in orders if order != baseline]
    output: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        family: list[dict[str, Any]] = []
        for metric in BINARY_METRICS:
            for compared in compared_orders:
                strata: list[tuple[int, int, int, int]] = []
                task_differences: list[float] = []
                direction = {"positive": 0, "negative": 0, "zero": 0}
                for problem in PROBLEM_IDS:
                    base_rows = grouped[(model, problem, baseline)]
                    compared_rows = grouped[(model, problem, compared)]
                    base_s = sum(row[metric] for row in base_rows)
                    comp_s = sum(row[metric] for row in compared_rows)
                    strata.append((comp_s, len(compared_rows) - comp_s, base_s, len(base_rows) - base_s))
                    diff = comp_s / len(compared_rows) - base_s / len(base_rows)
                    task_differences.append(diff)
                    direction["positive" if diff > 0 else "negative" if diff < 0 else "zero"] += 1
                ci_low, ci_high = bootstrap_mean_ci(
                    task_differences,
                    seed=stable_seed(model, metric, compared),
                    samples=bootstrap_samples,
                )
                result = {
                    "model": model,
                    "metric": metric,
                    "baseline_order": baseline,
                    "compared_order": compared,
                    **cmh(strata),
                    "task_mean_risk_difference": sum(task_differences) / len(task_differences),
                    "task_difference_ci95_low": ci_low,
                    "task_difference_ci95_high": ci_high,
                    "tasks_compared_better": direction["positive"],
                    "tasks_baseline_better": direction["negative"],
                    "tasks_tied": direction["zero"],
                }
                family.append(result)
        adjusted = holm([float(result["p_value"]) for result in family])
        for result, adjusted_p in zip(family, adjusted):
            result["p_value_holm_model_six_tests"] = adjusted_p
            result["significant_holm_0_05"] = adjusted_p < 0.05
        output.extend(family)
    return output


def descriptive_results(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = group_rows(rows)
    summary: list[dict[str, Any]] = []
    task_rates: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for order in sorted({row["order"] for row in rows}):
            for metric in BINARY_METRICS:
                all_rows = [row for row in rows if row["model"] == model and row["order"] == order]
                successes = sum(row[metric] for row in all_rows)
                low, high = wilson_interval(successes, len(all_rows))
                summary.append({
                    "model": model, "order": order, "metric": metric,
                    "successes": successes, "total": len(all_rows), "rate": successes / len(all_rows),
                    "ci95_low_wilson": low, "ci95_high_wilson": high,
                })
                for problem in PROBLEM_IDS:
                    values = grouped[(model, problem, order)]
                    success = sum(row[metric] for row in values)
                    task_rates.append({
                        "model": model, "order": order, "metric": metric, "problem": problem,
                        "successes": success, "total": len(values), "rate": success / len(values),
                    })
    return summary, task_rates


def conditional_plan_length_results(
    rows: list[dict[str, Any]], baseline: str, orders: tuple[str, ...], bootstrap_samples: int
) -> list[dict[str, Any]]:
    grouped = group_rows(rows)
    compared_orders = [order for order in orders if order != baseline]
    results: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for compared in compared_orders:
            differences: list[float] = []
            for problem in PROBLEM_IDS:
                def lengths(order: str) -> list[float]:
                    return [float(row["plan_length"]) for row in grouped[(model, problem, order)] if row["plan_length"] is not None]
                base, comp = lengths(baseline), lengths(compared)
                if base and comp:
                    differences.append(sum(comp) / len(comp) - sum(base) / len(base))
            low, high = bootstrap_mean_ci(
                differences,
                seed=stable_seed(model, compared, "length"),
                samples=bootstrap_samples,
            )
            results.append({
                "model": model, "baseline_order": baseline, "compared_order": compared,
                "tasks_with_reachable_plans_in_both_orders": len(differences),
                "task_mean_plan_length_difference": sum(differences) / len(differences) if differences else None,
                "task_difference_ci95_low": low, "task_difference_ci95_high": high,
                "interpretation": "Conditional on at least one reachable plan in both orders; not a population-wide plan-quality effect.",
            })
    return results


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else [])
        writer.writeheader()
        writer.writerows(records)


def write_report(
    path: Path, series_label: str, baseline: str, binary: list[dict[str, Any]],
    summary: list[dict[str, Any]], lengths: list[dict[str, Any]]
) -> None:
    primary_test_count = len({(r["metric"], r["compared_order"]) for r in binary if r["model"] == MODEL_NAMES[0]})
    lines = [
        "# Reference-order experiment statistics", "",
        f"Series: `{series_label}`; baseline order: `{baseline}`.", "",
        "Primary outcomes are strict VAL reachability and executability. Each PDDL problem is a CMH stratum; API calls are independent and run ids are not paired between orders.",
        f"Holm correction is applied over {primary_test_count} primary tests per model.", "",
        "## Descriptive rates", "",
        "| model | order | metric | successes / total | rate | 95% Wilson CI |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in summary:
        lines.append(f"| {r['model']} | {r['order']} | {r['metric']} | {r['successes']} / {r['total']} | {r['rate']:.3f} | [{r['ci95_low_wilson']:.3f}, {r['ci95_high_wilson']:.3f}] |")
    lines += ["", "## CMH tests", "", "| model | metric | comparison | pooled risk diff | task-mean diff [95% CI] | direction across tasks (+/−/=) | p Holm |", "| --- | --- | --- | ---: | --- | ---: | ---: |"]
    for r in binary:
        ci = f"{r['task_mean_risk_difference']:+.3f} [{r['task_difference_ci95_low']:+.3f}, {r['task_difference_ci95_high']:+.3f}]"
        direction = f"{r['tasks_compared_better']}/{r['tasks_baseline_better']}/{r['tasks_tied']}"
        lines.append(f"| {r['model']} | {r['metric']} | {r['baseline_order']} → {r['compared_order']} | {r['risk_difference']:+.3f} | {ci} | {direction} | {r['p_value_holm_model_six_tests']:.4g} |")
    lines += ["", "## Conditional plan length", "", "Plan length is summarized only for reachable plans and must not be interpreted as an unconditional quality comparison.", ""]
    for r in lengths:
        lines.append(f"- {r['model']}, `{r['baseline_order']}` → `{r['compared_order']}`: {r['tasks_with_reachable_plans_in_both_orders']} tasks; mean difference {r['task_mean_plan_length_difference']}; CI [{r['task_difference_ci95_low']}, {r['task_difference_ci95_high']}].")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an isolated reference-order experiment series.")
    parser.add_argument("--prefix", default="reference_final_20260917")
    parser.add_argument(
        "--orders", nargs="+", metavar="ORDER",
        help="Explicit order directories. Defaults to the four reference-order variants for --prefix.",
    )
    parser.add_argument("--baseline", help="Baseline order. Defaults to the first selected order.")
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.bootstrap_samples < 100:
        parser.error("--runs must be positive and --bootstrap-samples must be at least 100")
    selected_orders = tuple(args.orders) if args.orders else orders_for(args.prefix)
    baseline = args.baseline or selected_orders[0]
    if baseline not in selected_orders:
        parser.error("--baseline must be one of --orders")
    if len(selected_orders) < 2:
        parser.error("at least two orders are required")
    rows = load_rows(selected_orders, args.runs)
    binary = binary_results(rows, baseline, selected_orders, args.bootstrap_samples)
    summary, task_rates = descriptive_results(rows)
    lengths = conditional_plan_length_results(rows, baseline, selected_orders, args.bootstrap_samples)
    out = args.output_dir or Path(__file__).resolve().parent / args.prefix
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary_rates.csv", summary)
    write_csv(out / "task_rates.csv", task_rates)
    write_csv(out / "binary_cmh_tests.csv", binary)
    write_csv(out / "conditional_plan_length.csv", lengths)
    write_report(out / "README.md", args.prefix, baseline, binary, summary, lengths)
    (out / "analysis_manifest.json").write_text(json.dumps({"series": args.prefix, "orders": selected_orders, "baseline": baseline, "runs": args.runs, "input_rows": len(rows), "methods": {"binary": "CMH stratified by problem; independent runs", "multiple_testing": f"Holm across {(len(selected_orders) - 1) * len(BINARY_METRICS)} tests per model", "plan_length": "conditional descriptive task-level analysis"}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Analyzed {len(rows)} complete rows; wrote results to {out}")


if __name__ == "__main__":
    main()
