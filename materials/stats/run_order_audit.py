#!/usr/bin/env python3
"""Independent, task-blocked audit of the ten action-order conditions.

Reads the saved VAL results; never calls a model or changes raw artifacts.
Requires numpy (available in the project's .venv). Run from code/ with
``../.venv/bin/python materials/stats/run_order_audit.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_config import PROBLEM_IDS, REFERENCE_PLAN_ACTION_COUNTS_BY_ID
from shuffler import _extract_action_blocks, kendall_tau_distance


DATA = ROOT / "materials/logistics/alpha"
DEFAULT_OUT = ROOT / "materials/stats/order_audit"
VALIDATOR_RECHECK = ROOT / "materials/stats/validator_recheck.json"
MODELS = ("deepseek/deepseek-v4-flash", "gpt-oss-120b", "nemotron-3-super")
ORDERS = (
    "canonical", "canonical_disp_1", "canonical_disp_2", "canonical_disp_3",
    "plan_front", "plan_back", "plan_scatter", "reference_disp_1",
    "reference_disp_2", "reference_reverse",
)
OUTCOMES = ("parse_error", "execution_error", "executable_no_goal", "goal_reached")
METRICS = ("reachability", "executability", "conditional_reachability",
           "optimality_ratio", "plan_length_ratio", "execution_progress",
           "first_failure_step", "parse_error_given_nonexecutable")
SOURCE_PREFIX = {
    MODELS[0]: "reference_deepinfra_20260921",
    MODELS[1]: "reference_final_20260917",
    MODELS[2]: "reference_final_20260917",
}


def source_directory(model: str, order: str) -> str:
    if order.startswith("canonical_disp_"):
        return order.removeprefix("canonical_")
    if order.startswith("reference_"):
        return SOURCE_PREFIX[model] + "_" + order.removeprefix("reference_")
    return order


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_sign_p(integer_differences: list[int]) -> float:
    """Exact two-sided randomization p for whole-task label swaps.

    Dynamic programming avoids materializing 2**20 permutations. Valid for a
    pair under equality of the two *distributions* of independent run batches
    within each task. Also used descriptively for interaction/conditional
    contrasts, where that exchangeability is not guaranteed.
    """
    values = [abs(int(x)) for x in integer_differences if int(x)]
    if not values:
        return 1.0
    ways = Counter({0: 1})
    for value in values:
        following: Counter[int] = Counter()
        for total, count in ways.items():
            following[total + value] += count
            following[total - value] += count
        ways = following
    observed = abs(sum(integer_differences))
    return sum(n for total, n in ways.items() if abs(total) >= observed) / (2 ** len(values))


def holm(pvalues: list[float]) -> list[float]:
    result = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        running = max(running, min(1.0, pvalues[index] * (len(pvalues) - rank)))
        result[index] = running
    return result


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                 samples: int) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    picks = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[picks].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def omnibus_order_p(counts: np.ndarray, rng: np.random.Generator,
                    samples: int) -> tuple[float, float]:
    """Permute entire order cells within each task under the joint null."""
    centered = counts - counts.mean(axis=1, keepdims=True)
    observed = float(np.square(centered.mean(axis=0)).sum())
    exceed = 0
    for _ in range(samples):
        permuted = np.take_along_axis(centered, rng.permuted(
            np.tile(np.arange(counts.shape[1]), (counts.shape[0], 1)), axis=1), axis=1)
        statistic = float(np.square(permuted.mean(axis=0)).sum())
        exceed += statistic >= observed - 1e-12
    return observed, (exceed + 1) / (samples + 1)


def wild_mean_p(task_vectors: np.ndarray, rng: np.random.Generator,
                samples: int) -> tuple[float, float]:
    """Approximate task-cluster wild test of a zero mean vector.

    Centering makes the bootstrap distribution satisfy the null; one sign is
    shared by all components of a task. This is not an exact randomization test.
    """
    n = len(task_vectors)
    observed = float(np.square(task_vectors.mean(axis=0)).sum())
    residual = task_vectors - task_vectors.mean(axis=0, keepdims=True)
    exceed = 0
    for _ in range(samples):
        signs = rng.choice((-1.0, 1.0), size=n)
        statistic = float(np.square((residual * signs[:, None]).mean(axis=0)).sum())
        exceed += statistic >= observed - 1e-12
    return observed, (exceed + 1) / (samples + 1)


def load_data(validator_recheck: Path = VALIDATOR_RECHECK) -> tuple[dict, dict, list[dict]]:
    """Return run metrics, actual permutations, and provenance.

    The two preserved plans lacking llm_result.json use only their previously
    recorded binary VAL outcome/length, with explicit provenance. Missing
    conditional fields are never silently imputed.
    """
    rechecked = json.loads(validator_recheck.read_text(encoding="utf-8"))["cases"]
    if len(rechecked) != 73 or any(r["outcome"] != "parse_error" for r in rechecked):
        raise ValueError("Expected all 73 original validator timeouts resolved as plan-format errors")
    overrides = {r["path"]: r for r in rechecked}
    if len(overrides) != len(rechecked):
        raise ValueError("Duplicate validator recheck paths")
    applied = set()
    cached_path = ROOT / "materials/stats/article_final/analysis_rows.csv"
    cached = {}
    with cached_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cached[(row["model"], row["problem"], row["order"], int(row["run"]))] = row
    data, sequences, provenance = {}, {}, []
    for task in PROBLEM_IDS:
        for order in ORDERS:
            for model in MODELS:
                directory = DATA / task / source_directory(model, order)
                domain = directory / "domain.pddl"
                sequence = tuple(_extract_action_blocks(domain.read_text(encoding="utf-8"))[3])
                if len(sequence) != 48 or len(set(sequence)) != 48:
                    raise ValueError(f"Expected 48 distinct actions: {domain}")
                sequences[(task, model, order)] = sequence
                for run in range(1, 16):
                    path = directory / str(run) / model.split("/")[-1] / "llm_result.json"
                    source = "llm_result"
                    if path.is_file():
                        raw = json.loads(path.read_text(encoding="utf-8"))
                        strict = (raw.get("metrics") or {}).get("strict") or {}
                        legacy = (raw.get("metrics") or {}).get("legacy") or {}
                        if not isinstance(strict.get("reachability"), bool) or not isinstance(strict.get("executability"), bool):
                            raise ValueError(f"Missing binary VAL outcomes: {path}")
                        key = str(path.relative_to(ROOT))
                        if key in overrides:
                            if not strict.get("validator_timed_out"):
                                raise ValueError(f"Recheck override is not an original timeout: {path}")
                            strict = dict(strict)
                            strict.update(reachability=False, executability=False,
                                          non_executable_failure="parse_error", validator_timed_out=False,
                                          execution_progress=0.0, first_failure_step=None)
                            applied.add(key)
                    else:
                        plan = path.with_name("llm.plan")
                        previous = cached.get((model, task, order, run))
                        if not plan.is_file() or previous is None:
                            raise FileNotFoundError(path)
                        strict = {
                            "reachability": bool(int(previous["reachability"])),
                            "executability": bool(int(previous["executability"])),
                            "plan_length": int(previous["plan_length"]) if previous["plan_length"] else None,
                        }
                        legacy = {}
                        source = "cached_binary_from_preserved_plan"
                        provenance.append({"model": model, "task": task, "order": order,
                                           "run": run, "source": source, "plan": str(plan.relative_to(ROOT))})
                    reached = int(strict["reachability"])
                    executable = int(strict["executability"])
                    if reached and not executable:
                        raise ValueError(f"Goal reached without executability: {path}")
                    error = strict.get("non_executable_failure")
                    if reached:
                        outcome = "goal_reached"
                    elif executable:
                        outcome = "executable_no_goal"
                    elif error == "validator_timeout":
                        outcome = "validator_timeout"
                    elif error == "parse_error":
                        outcome = "parse_error"
                    elif error == "state_execution_error":
                        outcome = "execution_error"
                    elif source.startswith("cached"):
                        raise ValueError(f"Unclassifiable cached outcome: {path}")
                    else:
                        raise ValueError(f"Unclassifiable outcome: {path}")
                    length = strict.get("plan_length")
                    ref_length = REFERENCE_PLAN_ACTION_COUNTS_BY_ID[task]
                    data[(model, task, order, run)] = {
                        "reachability": reached, "executability": executable,
                        "conditional_reachability": reached if executable else None,
                        "plan_length_ratio": float(length) / ref_length if reached and length is not None else None,
                        "optimality_ratio": legacy.get("optimality_ratio") if reached else None,
                        "execution_progress": strict.get("execution_progress") if source == "llm_result" else (1.0 if reached else None),
                        "first_failure_step": strict.get("first_failure_step") if outcome == "execution_error" else None,
                        "parse_error_given_nonexecutable": int(outcome == "parse_error") if outcome in ("parse_error", "execution_error") else None,
                        "outcome": outcome,
                    }
    if len(data) != 9000:
        raise ValueError(f"Expected 9000 runs, got {len(data)}")
    if applied != set(overrides):
        raise ValueError(f"Validator recheck mismatch: {len(applied)} of {len(overrides)} applied")
    return data, sequences, provenance


def counts_array(data: dict, metric: str) -> np.ndarray:
    array = np.empty((len(MODELS), len(PROBLEM_IDS), len(ORDERS)), dtype=int)
    for mi, model in enumerate(MODELS):
        for ti, task in enumerate(PROBLEM_IDS):
            for oi, order in enumerate(ORDERS):
                array[mi, ti, oi] = sum(data[(model, task, order, run)][metric] for run in range(1, 16))
    return array


def pairwise_binary(counts: np.ndarray, sequences: dict, rng: np.random.Generator,
                    samples: int, metric: str) -> list[dict]:
    records = []
    families = (("original", "canonical", ORDERS[1:]),
                ("plan_front", "plan_front", tuple(o for o in ORDERS if o not in ("plan_front", "canonical"))))
    for mi, model in enumerate(MODELS):
        for family, baseline, alternatives in families:
            family_rows = []
            bi = ORDERS.index(baseline)
            for other in alternatives:
                oi = ORDERS.index(other)
                tasks = [i for i, task in enumerate(PROBLEM_IDS)
                         if sequences[(task, model, baseline)] != sequences[(task, model, other)]]
                differences = counts[mi, tasks, oi] - counts[mi, tasks, bi]
                changes = differences.astype(float) / 15
                low, high = bootstrap_ci(changes, rng, samples)
                family_rows.append({
                    "metric": metric, "model": model, "family": family,
                    "baseline": baseline, "compared": other, "n_tasks": len(tasks),
                    "baseline_rate": float(counts[mi, tasks, bi].mean() / 15),
                    "compared_rate": float(counts[mi, tasks, oi].mean() / 15),
                    "risk_difference": float(changes.mean()), "ci95_low": low, "ci95_high": high,
                    "p_exact_task_swap": exact_sign_p(differences.tolist()),
                    "tasks_better": int((differences > 0).sum()),
                    "tasks_worse": int((differences < 0).sum()),
                    "tasks_equal": int((differences == 0).sum()),
                    "identical_tasks_excluded": len(PROBLEM_IDS) - len(tasks),
                })
            adjusted = holm([r["p_exact_task_swap"] for r in family_rows])
            for row, p in zip(family_rows, adjusted):
                row["p_holm_family"] = p
            records.extend(family_rows)
    return records


def binary_omnibus(counts: np.ndarray, rng: np.random.Generator,
                   samples: int, metric: str) -> list[dict]:
    records = []
    for mi, model in enumerate(MODELS):
        statistic, p = omnibus_order_p(counts[mi], rng, samples)
        records.append({"metric": metric, "model": model, "n_orders": len(ORDERS),
                        "n_tasks": len(PROBLEM_IDS), "statistic": statistic,
                        "p_permutation": p, "permutations": samples})
    for row, p in zip(records, holm([r["p_permutation"] for r in records])):
        row["p_holm_three_models"] = p
    return records


def outcome_omnibus(outcome_rows: list[dict], rng: np.random.Generator,
                    samples: int) -> list[dict]:
    """Joint task-blocked test of the four mutually exclusive run outcomes."""
    counts = np.zeros((len(MODELS), len(PROBLEM_IDS), len(ORDERS), len(OUTCOMES)), dtype=int)
    model_index = {value: i for i, value in enumerate(MODELS)}
    task_index = {value: i for i, value in enumerate(PROBLEM_IDS)}
    order_index = {value: i for i, value in enumerate(ORDERS)}
    outcome_index = {value: i for i, value in enumerate(OUTCOMES)}
    for row in outcome_rows:
        counts[model_index[row["model"]], task_index[row["task"]],
               order_index[row["order"]], outcome_index[row["outcome"]]] = row["count"]
    if not np.all(counts.sum(axis=3) == 15):
        raise ValueError("Outcome categories do not sum to 15 per cell")
    output = []
    for mi, model in enumerate(MODELS):
        values = counts[mi]
        centered = values - values.mean(axis=1, keepdims=True)
        observed = float(np.square(centered.mean(axis=0)).sum())
        exceed = 0
        for _ in range(samples):
            permutation = rng.permuted(
                np.tile(np.arange(len(ORDERS)), (len(values), 1)), axis=1)
            shuffled = np.take_along_axis(centered, permutation[:, :, None], axis=1)
            statistic = float(np.square(shuffled.mean(axis=0)).sum())
            exceed += statistic >= observed - 1e-12
        output.append({"model": model, "n_orders": len(ORDERS),
                       "n_tasks": len(PROBLEM_IDS), "statistic": observed,
                       "p_permutation": (exceed + 1) / (samples + 1),
                       "permutations": samples})
    for row, p in zip(output, holm([r["p_permutation"] for r in output])):
        row["p_holm_three_models"] = p
    return output

def interaction(counts: np.ndarray, sequences: dict, rng: np.random.Generator,
                samples: int, metric: str) -> tuple[list[dict], list[dict]]:
    omnibus, pairs = [], []
    changes = (counts - counts[:, :, ORDERS.index("canonical"), None]) / 15
    # shape: tasks x models x compared orders; remove the zero baseline column
    changes = np.transpose(changes[:, :, 1:], (1, 0, 2))
    interaction_vectors = changes - changes.mean(axis=1, keepdims=True)
    statistic, p = wild_mean_p(interaction_vectors.reshape(len(PROBLEM_IDS), -1), rng, samples)
    omnibus.append({"metric": metric, "n_orders": len(ORDERS),
                    "n_tasks": len(PROBLEM_IDS), "statistic": statistic,
                    "p_wild_task": p, "resamples": samples})
    for order in ORDERS[1:]:
        oi = ORDERS.index(order)
        task_ids = [ti for ti, task in enumerate(PROBLEM_IDS) if all(
            sequences[(task, model, "canonical")] != sequences[(task, model, order)]
            for model in MODELS)]
        for first in range(len(MODELS)):
            for second in range(first + 1, len(MODELS)):
                dd = (counts[first, task_ids, oi] - counts[first, task_ids, 0]
                      - counts[second, task_ids, oi] + counts[second, task_ids, 0]) / 15
                _, p2 = wild_mean_p(dd[:, None], rng, samples)
                low, high = bootstrap_ci(dd, rng, samples)
                pairs.append({"metric": metric, "order": order,
                              "model_a": MODELS[first], "model_b": MODELS[second],
                              "n_tasks": len(task_ids), "difference_in_differences": float(dd.mean()),
                              "ci95_low": low, "ci95_high": high, "p_wild_task": p2})
    for row, p in zip(pairs, holm([r["p_wild_task"] for r in pairs])):
        row["p_holm_family"] = p
    return omnibus, pairs


def kendall_distances(sequences: dict) -> list[dict]:
    distances = []
    for task in PROBLEM_IDS:
        for model in MODELS:
            original = sequences[(task, model, "canonical")]
            front = sequences[(task, model, "plan_front")]
            for order in ORDERS:
                seq = sequences[(task, model, order)]
                distances.append({"task": task, "model": model, "order": order,
                                  "from_original": kendall_tau_distance(original, seq),
                                  "from_plan_front": kendall_tau_distance(front, seq),
                                  "maximum": 1128})
    return distances



def adjusted_kendall_association(counts: np.ndarray, distances: list[dict],
                                 rng: np.random.Generator, samples: int,
                                 metric: str) -> list[dict]:
    """Task-centered association across all ten orders.

    This is observational: order distance may be confounded with other
    properties of the permutation or with collection time.
    """
    lookup = {(r["task"], r["model"], r["order"]): r for r in distances}
    slopes = np.empty((len(PROBLEM_IDS), len(MODELS)), dtype=float)
    curvatures = np.empty_like(slopes)
    for mi, model in enumerate(MODELS):
        for ti, task in enumerate(PROBLEM_IDS):
            x = np.array([lookup[(task, model, order)]["from_original"] / 1128
                          for order in ORDERS], dtype=float)
            y = counts[mi, ti].astype(float) / 15
            linear = x - x.mean()
            quadratic = x * x - np.square(x).mean()
            response = y - y.mean()
            slopes[ti, mi] = float(np.dot(linear, response) / np.dot(linear, linear))
            coefficients = np.linalg.lstsq(
                np.column_stack((linear, quadratic)), response, rcond=None)[0]
            curvatures[ti, mi] = float(coefficients[1])
    output = []
    for mi, model in enumerate(MODELS):
        for label, matrix in (("pooled_linear", slopes),
                              ("pooled_quadratic_given_linear", curvatures)):
            values = matrix[:, mi]
            _, p = wild_mean_p(values[:, None], rng, samples)
            low, high = bootstrap_ci(values, rng, samples)
            output.append({"metric": metric, "comparison": label,
                           "model_a": model, "model_b": "", "n_tasks": 20,
                           "coefficient_per_full_distance": float(values.mean()),
                           "ci95_low": low, "ci95_high": high, "p_wild_task": p})
    for first in range(len(MODELS)):
        for second in range(first + 1, len(MODELS)):
            values = slopes[:, first] - slopes[:, second]
            _, p = wild_mean_p(values[:, None], rng, samples)
            low, high = bootstrap_ci(values, rng, samples)
            output.append({"metric": metric, "comparison": "linear_slope_difference",
                           "model_a": MODELS[first], "model_b": MODELS[second],
                           "n_tasks": 20, "coefficient_per_full_distance": float(values.mean()),
                           "ci95_low": low, "ci95_high": high, "p_wild_task": p})

    for comparison in ("pooled_linear", "pooled_quadratic_given_linear",
                       "linear_slope_difference"):
        family_rows = [row for row in output if row["comparison"] == comparison]
        for row, p in zip(family_rows, holm([r["p_wild_task"] for r in family_rows])):
            row["p_holm_family"] = p
    return output


def negative_controls(data: dict, sequences: dict) -> list[dict]:
    output = []
    for model in MODELS:
        for left_i, left in enumerate(ORDERS):
            for right in ORDERS[left_i + 1:]:
                if sequences[("p1", model, left)] != sequences[("p1", model, right)]:
                    continue
                a = sum(data[(model, "p1", left, run)]["reachability"] for run in range(1, 16))
                b = sum(data[(model, "p1", right, run)]["reachability"] for run in range(1, 16))
                output.append({"model": model, "task": "p1", "left": left, "right": right,
                               "left_successes": a, "right_successes": b,
                               "risk_difference": (b - a) / 15})
    return output


def secondary_summaries(data: dict, rng: np.random.Generator,
                        samples: int) -> tuple[list[dict], list[dict], list[dict]]:
    summaries, contrasts, outcome_rows = [], [], []
    cell_values: dict[tuple, list[float]] = {}
    for model in MODELS:
        for task in PROBLEM_IDS:
            for order in ORDERS:
                cell = [data[(model, task, order, run)] for run in range(1, 16)]
                for metric in METRICS:
                    cell_values[(model, task, order, metric)] = [float(r[metric]) for r in cell if r[metric] is not None]
                for outcome in OUTCOMES:
                    outcome_rows.append({"model": model, "task": task, "order": order,
                                         "outcome": outcome,
                                         "count": sum(r["outcome"] == outcome for r in cell), "denominator": 15})
    for model in MODELS:
        for order in ORDERS:
            for metric in METRICS:
                groups = [cell_values[(model, task, order, metric)] for task in PROBLEM_IDS]
                means = np.array([np.mean(g) for g in groups if g], dtype=float)
                low, high = bootstrap_ci(means, rng, samples)
                summaries.append({"model": model, "order": order, "metric": metric,
                                  "n_tasks": len(means), "n_runs": sum(map(len, groups)),
                                  "task_mean": float(means.mean()) if len(means) else math.nan,
                                  "ci95_low": low, "ci95_high": high})
        for metric in METRICS[2:]:
            family = []
            for other in ORDERS[1:]:
                pairs = []
                for task in PROBLEM_IDS:
                    a = cell_values[(model, task, "canonical", metric)]
                    b = cell_values[(model, task, other, metric)]
                    if a and b:
                        pairs.append(float(np.mean(b) - np.mean(a)))
                values = np.array(pairs)
                low, high = bootstrap_ci(values, rng, samples)
                # A sign-flip here is a descriptive sensitivity analysis, not
                # an exact randomization test: selection into the subset varies.
                if len(values):
                    _, p = wild_mean_p(values[:, None], rng, samples)
                else:
                    p = math.nan
                family.append({"model": model, "metric": metric,
                               "baseline": "canonical", "compared": other,
                               "n_tasks_with_both_defined": len(values),
                               "task_mean_difference": float(values.mean()) if len(values) else math.nan,
                               "ci95_low": low, "ci95_high": high,
                               "p_exploratory_wild_task": p})
            valid = [row for row in family if not math.isnan(row["p_exploratory_wild_task"])]
            for row, p in zip(valid, holm([r["p_exploratory_wild_task"] for r in valid])):
                row["p_holm_metric_model"] = p
            contrasts.extend(family)
    return summaries, contrasts, outcome_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--validator-recheck", type=Path, default=VALIDATOR_RECHECK)
    args = parser.parse_args()
    if args.permutations < 100 or args.bootstrap < 100:
        parser.error("at least 100 permutations and bootstrap samples required")
    rng = np.random.default_rng(args.seed)
    data, sequences, provenance = load_data(args.validator_recheck)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "recovered_runs.csv", provenance)
    summaries, contrasts, outcomes = secondary_summaries(data, rng, args.bootstrap)
    write_csv(output / "metric_summaries.csv", summaries)
    write_csv(output / "conditional_exploratory_contrasts.csv", contrasts)
    write_csv(output / "outcome_counts.csv", outcomes)
    write_csv(output / "outcome_omnibus.csv", outcome_omnibus(outcomes, rng, args.permutations))
    write_csv(output / "identical_order_controls.csv", negative_controls(data, sequences))
    distances = kendall_distances(sequences)
    write_csv(output / "kendall_distances.csv", distances)
    for metric in ("reachability", "executability"):
        counts = counts_array(data, metric)
        write_csv(output / f"{metric}_omnibus.csv", binary_omnibus(counts, rng, args.permutations, metric))
        write_csv(output / f"{metric}_baseline_contrasts.csv",
                  pairwise_binary(counts, sequences, rng, args.bootstrap, metric))
        interaction_rows, interaction_pairs = interaction(counts, sequences, rng, args.permutations, metric)
        write_csv(output / f"{metric}_interaction.csv", interaction_rows)
        write_csv(output / f"{metric}_interaction_contrasts.csv", interaction_pairs)
        write_csv(output / f"{metric}_kendall_adjusted.csv",
                  adjusted_kendall_association(counts, distances, rng, args.permutations, metric))
    (output / "manifest.json").write_text(json.dumps({
        "runs": len(data), "tasks": len(PROBLEM_IDS), "models": list(MODELS),
        "orders": list(ORDERS), "recovered_binary_runs": len(provenance),
        "primary_omnibus": "all ten named orders, p1-p20, reachability",
        "n_orders": len(ORDERS),
        "validator_rechecked_plan_errors": 73, "outcomes": list(OUTCOMES),
        "conditional_metrics_are_selected_subsets": True,
        "permutations": args.permutations, "bootstrap": args.bootstrap, "seed": args.seed,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Analyzed {len(data)} runs; results in {output}")


if __name__ == "__main__":
    main()
