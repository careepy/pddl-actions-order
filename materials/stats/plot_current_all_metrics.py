#!/usr/bin/env python3
"""Build the no-wild manuscript's eight-metric figure from the current audit."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from plot_article_all_metrics import plot_all_metrics
from build_article_results import SERIES_CONFIGS
from run_order_audit import DEFAULT_OUT, MODELS, ORDERS, PROBLEM_IDS, bootstrap_ci, load_data
from experiment_config import REFERENCE_PLAN_ACTION_COUNTS_BY_ID


SOURCE_METRIC = {
    "executability": "executability",
    "reachability": "reachability",
    "conditional_reachability": "conditional_reachability",
    "optimality_ratio": "plan_length_ratio",
    "execution_progress": "execution_progress",
    "first_failure_step": "first_failure_step",
    "non_executable_failure": "parse_error_given_nonexecutable",
}


def audited_rows(source: Path, seed: int, bootstrap_samples: int) -> list[dict[str, str]]:
    with (source / "metric_summaries.csv").open(encoding="utf-8", newline="") as handle:
        audit = {(r["model"], r["order"], r["metric"]): r for r in csv.DictReader(handle)}
    data, _, _ = load_data()
    rng = np.random.default_rng(seed)
    rows = []
    for model in MODELS:
        for order in ORDERS:
            for metric, source_metric in SOURCE_METRIC.items():
                item = audit[(model, order, source_metric)]
                rows.append({
                    "series": "combined", "model": model, "order": order, "metric": metric,
                    "task_mean": item["task_mean"],
                    "ci95_low_task_bootstrap": item["ci95_low"],
                    "ci95_high_task_bootstrap": item["ci95_high"],
                })
            task_means = []
            for task in PROBLEM_IDS:
                lengths = [
                    value * REFERENCE_PLAN_ACTION_COUNTS_BY_ID[task]
                    for run in range(1, 16)
                    if (value := data[(model, task, order, run)]["plan_length_ratio"]) is not None
                ]
                if lengths:
                    task_means.append(float(np.mean(lengths)))
            if not task_means:
                raise ValueError(f"No successful plans for {model}, {order}")
            low, high = bootstrap_ci(np.asarray(task_means), rng, bootstrap_samples)
            rows.append({
                "series": "combined", "model": model, "order": order,
                "metric": "plan_length", "task_mean": str(float(np.mean(task_means))),
                "ci95_low_task_bootstrap": str(low), "ci95_high_task_bootstrap": str(high),
            })
    if len(rows) != 3 * 10 * 8:
        raise ValueError(f"Expected 240 plotted values, got {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[3] / "latex")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series = next(s for s in SERIES_CONFIGS if s["key"] == "combined")
    plot_all_metrics(
        audited_rows(args.source, args.seed, args.bootstrap_samples), args.output_dir, series,
        stem="figure_2_math_no_wild",
    )
    print(f"Wrote current eight-metric figure to {args.output_dir}")


if __name__ == "__main__":
    main()
