#!/usr/bin/env python3
"""Recheck secondary manuscript claims with centered resampling of whole tasks.

This analysis replaces the sign-multiplier p-values for the no-wild manuscript.
The source audit and its original sensitivity calculations remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from run_order_audit import (
    DEFAULT_OUT, MODELS, ORDERS, OUTCOMES, PROBLEM_IDS,
    holm, load_data, write_csv,
)
from run_task_mean_tests import centered_task_bootstrap_p, load_outcome_fractions
from experiment_config import REFERENCE_PLAN_ACTION_COUNTS_BY_ID


def distance_lookup(path: Path) -> dict[tuple[str, str, str], float]:
    with path.open(encoding="utf-8", newline="") as source:
        return {
            (row["task"], row["model"], row["order"]):
            int(row["from_original"]) / int(row["maximum"])
            for row in csv.DictReader(source)
        }


def kendall_and_interaction(source: Path, rng: np.random.Generator, samples: int) -> None:
    fractions = load_outcome_fractions(source / "outcome_counts.csv")
    goal = fractions[..., OUTCOMES.index("goal_reached")]
    distances = distance_lookup(source / "kendall_distances.csv")
    slope_rows = []
    for mi, model in enumerate(MODELS):
        slopes = []
        for ti, task in enumerate(PROBLEM_IDS):
            x = np.array([distances[(task, model, order)] for order in ORDERS])
            y = goal[mi, ti]
            xc, yc = x - x.mean(), y - y.mean()
            slopes.append(float(np.dot(xc, yc) / np.dot(xc, xc)))
        slopes = np.asarray(slopes)
        slope_rows.append({
            "model": model, "n_tasks": len(slopes), "mean_slope": float(slopes.mean()),
            "p_task_bootstrap": centered_task_bootstrap_p(slopes[:, None], rng, samples),
            "resamples": samples,
        })
    for row, p in zip(slope_rows, holm([r["p_task_bootstrap"] for r in slope_rows])):
        row["p_holm_three_models"] = p
    write_csv(source / "task_mean_secondary_kendall.csv", slope_rows)

    changes = np.transpose((goal - goal[:, :, :1])[:, :, 1:], (1, 0, 2))
    interaction_vectors = changes - changes.mean(axis=1, keepdims=True)
    p = centered_task_bootstrap_p(interaction_vectors.reshape(len(PROBLEM_IDS), -1), rng, samples)
    write_csv(source / "task_mean_secondary_interaction.csv", [{
        "n_tasks": len(PROBLEM_IDS), "n_orders": len(ORDERS),
        "statistic": float(np.square(interaction_vectors.mean(axis=0)).sum()),
        "p_task_bootstrap": p, "resamples": samples,
    }])


def successful_plan_contrasts(source: Path, rng: np.random.Generator, samples: int) -> None:
    data, sequences, _ = load_data()
    records = []
    for metric in ("plan_length", "plan_length_ratio"):
        for model in MODELS:
            family = []
            for order in ORDERS[1:]:
                differences = []
                for task in PROBLEM_IDS:
                    if sequences[(task, model, "canonical")] == sequences[(task, model, order)]:
                        continue
                    def cell_mean(current_order: str) -> float | None:
                        vals = [
                            data[(model, task, current_order, run)]["plan_length_ratio"]
                            for run in range(1, 16)
                        ]
                        vals = [v for v in vals if v is not None]
                        if not vals:
                            return None
                        mean = float(np.mean(vals))
                        return mean * REFERENCE_PLAN_ACTION_COUNTS_BY_ID[task] if metric == "plan_length" else mean
                    baseline = cell_mean("canonical")
                    compared = cell_mean(order)
                    if baseline is not None and compared is not None:
                        differences.append(compared - baseline)
                values = np.asarray(differences)
                family.append({
                    "model": model, "metric": metric, "baseline": "canonical",
                    "compared": order, "n_tasks_with_both_defined": len(values),
                    "task_mean_difference": float(values.mean()),
                    "p_task_bootstrap": centered_task_bootstrap_p(values[:, None], rng, samples),
                    "resamples": samples,
                })
            for row, p in zip(family, holm([r["p_task_bootstrap"] for r in family])):
                row["p_holm_metric_model"] = p
            records.extend(family)
    write_csv(source / "task_mean_successful_plan_contrasts.csv", records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=62301)
    args = parser.parse_args()
    if args.samples < 100:
        parser.error("--samples must be at least 100")
    rng = np.random.default_rng(args.seed)
    kendall_and_interaction(args.source, rng, args.samples)
    successful_plan_contrasts(args.source, rng, args.samples)
    print(f"Wrote centered task-bootstrap secondary results to {args.source}")


if __name__ == "__main__":
    main()
