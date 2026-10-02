#!/usr/bin/env python3
"""Test average action-order effects with planning problems as the sampling unit.

The centered task bootstrap targets zero mean contrasts across comparable
problems.  The Rademacher multiplier calculation for goal reachability is a
sensitivity analysis, not an exact finite-sample randomization test.

Run from code/ with ``../.venv/bin/python materials/stats/run_task_mean_tests.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from run_order_audit import (
    DATA, MODELS, ORDERS, OUTCOMES, PROBLEM_IDS, DEFAULT_OUT,
    holm, source_directory, write_csv,
)
from shuffler import _extract_action_blocks


def load_outcome_fractions(path: Path) -> np.ndarray:
    fractions = np.zeros((len(MODELS), len(PROBLEM_IDS), len(ORDERS), len(OUTCOMES)))
    model_index = {value: i for i, value in enumerate(MODELS)}
    task_index = {value: i for i, value in enumerate(PROBLEM_IDS)}
    order_index = {value: i for i, value in enumerate(ORDERS)}
    outcome_index = {value: i for i, value in enumerate(OUTCOMES)}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            fractions[model_index[row["model"]], task_index[row["task"]],
                      order_index[row["order"]], outcome_index[row["outcome"]]] = (
                          int(row["count"]) / int(row["denominator"]))
    if not np.allclose(fractions.sum(axis=-1), 1):
        raise ValueError("Every model/task/order cell must contain 15 outcomes")
    return fractions


def centered_task_bootstrap_p(task_vectors: np.ndarray, rng: np.random.Generator,
                              samples: int) -> float:
    """Null-center the whole task vectors, then resample tasks with replacement."""
    observed = float(np.square(task_vectors.mean(axis=0)).sum())
    residuals = task_vectors - task_vectors.mean(axis=0, keepdims=True)
    n_tasks = len(task_vectors)
    exceed = 0
    for start in range(0, samples, 5000):
        count = min(5000, samples - start)
        picks = rng.integers(n_tasks, size=(count, n_tasks))
        bootstrap_means = residuals[picks].mean(axis=1)
        statistics = np.square(bootstrap_means).reshape(count, -1).sum(axis=1)
        exceed += int(np.count_nonzero(statistics >= observed - 1e-14))
    return (exceed + 1) / (samples + 1)


def wild_sign_p(task_vectors: np.ndarray) -> float:
    """Enumerate all task-level Rademacher signs for the centered residuals."""
    n_tasks = len(task_vectors)
    residuals = task_vectors - task_vectors.mean(axis=0, keepdims=True)
    observed = float(np.square(task_vectors.mean(axis=0)).sum())
    exceed = 0
    for start in range(0, 1 << n_tasks, 8192):
        integers = np.arange(start, min(start + 8192, 1 << n_tasks),
                             dtype=np.uint32)[:, None]
        bits = (integers >> np.arange(n_tasks, dtype=np.uint32)) & 1
        signs = 1.0 - 2.0 * bits.astype(float)
        bootstrap_means = signs @ residuals / n_tasks
        statistics = np.square(bootstrap_means).reshape(len(signs), -1).sum(axis=1)
        exceed += int(np.count_nonzero(statistics >= observed - 1e-14))
    return exceed / (1 << n_tasks)


def action_sequences() -> dict[tuple[str, str, str], tuple[str, ...]]:
    sequences = {}
    for task in PROBLEM_IDS:
        for model in MODELS:
            for order in ORDERS:
                domain = DATA / task / source_directory(model, order) / "domain.pddl"
                sequences[(task, model, order)] = tuple(
                    _extract_action_blocks(domain.read_text(encoding="utf-8"))[3])
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=240929)
    parser.add_argument("--omnibus-samples", type=int, default=200000)
    parser.add_argument("--pairwise-samples", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    fractions = load_outcome_fractions(DEFAULT_OUT / "outcome_counts.csv")
    metrics = {
        "reachability": fractions[..., OUTCOMES.index("goal_reached")],
        "executability": fractions[..., OUTCOMES.index("executable_no_goal")]
        + fractions[..., OUTCOMES.index("goal_reached")],
        "outcome": fractions,
    }
    omnibus = []
    for metric_index, (metric, values) in enumerate(metrics.items()):
        rows = []
        for model_index, model in enumerate(MODELS):
            task_vectors = values[model_index] - values[model_index].mean(
                axis=1, keepdims=True)
            rng = np.random.default_rng(np.random.SeedSequence(
                [args.seed, 0, metric_index, model_index]))
            rows.append({
                "metric": metric, "model": model, "n_tasks": len(PROBLEM_IDS),
                "p_task_bootstrap": centered_task_bootstrap_p(
                    task_vectors, rng, args.omnibus_samples),
                "bootstrap_samples": args.omnibus_samples,
            })
        for row, adjusted in zip(rows, holm([r["p_task_bootstrap"] for r in rows])):
            row["p_holm_three_models"] = adjusted
        omnibus.extend(rows)

    sequences = action_sequences()
    pairwise = []
    families = (
        ("original", "canonical", ORDERS[1:]),
        ("plan_front", "plan_front",
         tuple(order for order in ORDERS if order not in ("canonical", "plan_front"))),
    )
    for metric_index, metric in enumerate(("reachability", "executability")):
        values = metrics[metric]
        for model_index, model in enumerate(MODELS):
            for family_index, (family, baseline, alternatives) in enumerate(families):
                rows = []
                for other_index, other in enumerate(alternatives):
                    tasks = [i for i, task in enumerate(PROBLEM_IDS)
                             if sequences[(task, model, baseline)]
                             != sequences[(task, model, other)]]
                    differences = (values[model_index, tasks, ORDERS.index(other)]
                                   - values[model_index, tasks, ORDERS.index(baseline)])
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [args.seed, 1, metric_index, model_index,
                         family_index, other_index]))
                    rows.append({
                        "metric": metric, "model": model, "family": family,
                        "baseline": baseline, "compared": other,
                        "n_tasks": len(tasks),
                        "risk_difference": float(differences.mean()),
                        "p_task_bootstrap": centered_task_bootstrap_p(
                            differences[:, None], rng, args.pairwise_samples),
                        "bootstrap_samples": args.pairwise_samples,
                    })
                for row, adjusted in zip(rows, holm([r["p_task_bootstrap"] for r in rows])):
                    row["p_holm_family"] = adjusted
                pairwise.extend(rows)

    wild = []
    goal = metrics["reachability"]
    for model_index, model in enumerate(MODELS):
        task_vectors = goal[model_index] - goal[model_index].mean(axis=1, keepdims=True)
        wild.append({"model": model, "p_wild_sign": wild_sign_p(task_vectors)})
    for row, adjusted in zip(wild, holm([r["p_wild_sign"] for r in wild])):
        row["p_holm_three_models"] = adjusted

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "task_mean_omnibus.csv", omnibus)
    write_csv(args.output / "task_mean_baseline_contrasts.csv", pairwise)
    write_csv(args.output / "task_mean_wild_sensitivity.csv", wild)
    for row in omnibus:
        print(row["metric"], row["model"], row["p_holm_three_models"])
    for row in wild:
        print("wild reachability", row["model"], row["p_holm_three_models"])
    for row in pairwise:
        if row["p_holm_family"] < .05:
            print("pairwise", row["metric"], row["model"], row["family"],
                  row["compared"], row["p_holm_family"])


if __name__ == "__main__":
    main()
