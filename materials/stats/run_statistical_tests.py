import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import MODEL_NAMES, PROBLEM_IDS
from shuffler import VARIANT_NAMES
from token_usage import build_token_usage_from_payload


BASELINE_ORDER = "canonical"
COMPARED_ORDERS = [order for order in VARIANT_NAMES if order != BASELINE_ORDER]
DISP3_ORDER = "disp_3"
DISP3_BASELINE_ORDERS = [order for order in VARIANT_NAMES if order != DISP3_ORDER]
DISP3_METRIC = "reachability"
DATA_ROOT = PROJECT_ROOT / "materials" / "logistics" / "alpha"
OUTPUT_DIR = Path(__file__).resolve().parent

BINARY_METRICS = {
    "reachability": "Goal reached under strict VAL validation.",
    "executability": "Plan executed under strict VAL validation.",
    "non_executable_failure": "Strict validation failed before executable completion.",
}
CONDITIONAL_BINARY_METRICS = {
    "conditional_reachability": (
        "Goal reached among executable plans; non-executable plans are excluded per order."
    ),
}
NUMERIC_METRICS = {
    "plan_length": "Analysis plan length: strict VAL length for parsable reachable plans; missing for unparsable plans and all plans that did not reach the goal.",
    "optimality_ratio": "Validated cost divided by reference cost; only available when goal is reached.",
    "first_failure_step": "First failed execution step; only available for state execution errors.",
    "prompt_tokens": "Prompt tokens reported by provider.",
    "completion_tokens": "Completion tokens reported or normalized from provider payload.",
    "reasoning_completion_tokens": "Reasoning-token component of completion tokens.",
    "raw_completion_tokens": "Raw answer-token component of completion tokens.",
    "total_tokens": "Total tokens reported or normalized from provider payload.",
    "duration_sec": "Model call duration in seconds.",
}


def model_output_dir_name(model_name: str) -> str:
    return model_name.split("/")[-1].replace(":", "-").replace(".", "-")


def exact_mcnemar_p(order_only: int, baseline_only: int) -> float:
    discordant = order_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(0, min(order_only, baseline_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_two_sided_p(
    row1_success: int,
    row1_failure: int,
    row2_success: int,
    row2_failure: int,
) -> float:
    row1_total = row1_success + row1_failure
    row2_total = row2_success + row2_failure
    success_total = row1_success + row2_success
    grand_total = row1_total + row2_total
    if row1_total == 0 or row2_total == 0 or grand_total == 0:
        return 1.0

    def table_probability(row1_success_count: int) -> float:
        return math.exp(
            log_choose(row1_total, row1_success_count)
            + log_choose(row2_total, success_total - row1_success_count)
            - log_choose(grand_total, success_total)
        )

    lower = max(0, success_total - row2_total)
    upper = min(row1_total, success_total)
    observed = table_probability(row1_success)
    p_value = sum(
        probability
        for row1_success_count in range(lower, upper + 1)
        for probability in [table_probability(row1_success_count)]
        if probability <= observed + 1e-12
    )
    return min(1.0, p_value)


def t_pdf(x: float, df: int) -> float:
    return math.exp(
        math.lgamma((df + 1) / 2)
        - math.lgamma(df / 2)
        - 0.5 * math.log(df * math.pi)
        - ((df + 1) / 2) * math.log1p((x * x) / df)
    )


def simpson_integral(func, lower: float, upper: float, intervals: int = 12000) -> float:
    if lower == upper:
        return 0.0
    if intervals % 2:
        intervals += 1
    step = (upper - lower) / intervals
    total = func(lower) + func(upper)
    for index in range(1, intervals):
        total += (4 if index % 2 else 2) * func(lower + index * step)
    return total * step / 3


def t_cdf(value: float, df: int) -> float:
    if value == 0:
        return 0.5
    area = simpson_integral(lambda x: t_pdf(x, df), 0.0, abs(value))
    return 0.5 + area if value > 0 else 0.5 - area


def paired_t_test(differences: list[float]) -> dict[str, float | int | None]:
    sample_size = len(differences)
    if sample_size < 2:
        return {
            "n": sample_size,
            "mean_diff": statistics.mean(differences) if differences else None,
            "sd_diff": None,
            "t": None,
            "df": sample_size - 1,
            "p_value": None,
            "cohens_dz": None,
        }

    mean_diff = statistics.mean(differences)
    sd_diff = statistics.stdev(differences)
    if sd_diff == 0:
        t_value = 0.0 if mean_diff == 0 else math.copysign(math.inf, mean_diff)
        p_value = 1.0 if mean_diff == 0 else 0.0
        cohens_dz = 0.0 if mean_diff == 0 else math.copysign(math.inf, mean_diff)
    else:
        t_value = mean_diff / (sd_diff / math.sqrt(sample_size))
        p_value = 2 * (1 - t_cdf(abs(t_value), sample_size - 1))
        p_value = max(0.0, min(1.0, p_value))
        cohens_dz = mean_diff / sd_diff

    return {
        "n": sample_size,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "t": t_value,
        "df": sample_size - 1,
        "p_value": p_value,
        "cohens_dz": cohens_dz,
    }


def sign_flip_permutation_p(
    differences: list[float],
    *,
    samples: int = 100000,
    seed: int = 12345,
) -> float | None:
    if not differences:
        return None

    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 1.0

    observed = abs(sum(nonzero) / len(differences))
    nonzero_count = len(nonzero)
    if nonzero_count <= 20:
        sum_counts = Counter({0.0: 1})
        for value in nonzero:
            next_counts: Counter[float] = Counter()
            for signed_sum, count in sum_counts.items():
                next_counts[signed_sum + value] += count
                next_counts[signed_sum - value] += count
            sum_counts = next_counts

        total = 1 << nonzero_count
        extreme = sum(
            count
            for signed_sum, count in sum_counts.items()
            if abs(signed_sum / len(differences)) >= observed - 1e-12
        )
        return extreme / total

    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        signed_sum = sum(value if rng.random() < 0.5 else -value for value in nonzero)
        if abs(signed_sum / len(differences)) >= observed - 1e-12:
            extreme += 1
    return extreme / samples


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    lower_weight = upper - position
    upper_weight = position - lower
    return ordered[lower] * lower_weight + ordered[upper] * upper_weight


def bootstrap_mean_ci(
    differences: list[float],
    *,
    samples: int = 10000,
    seed: int = 54321,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    if not differences:
        return None, None

    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(samples):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        bootstrap_means.append(sum(sample) / len(sample))

    return (
        percentile(bootstrap_means, alpha / 2),
        percentile(bootstrap_means, 1 - alpha / 2),
    )


def holm_adjust(results: list[dict[str, Any]], p_key: str) -> list[float | None]:
    indexed = [
        (index, result[p_key])
        for index, result in enumerate(results)
        if result.get(p_key) is not None
    ]
    ordered = sorted(indexed, key=lambda item: item[1])
    adjusted: list[float | None] = [None] * len(results)
    running = 0.0
    total = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        value = min(1.0, p_value * (total - rank))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def apply_holm_adjustment(results: list[dict[str, Any]], p_key: str, output_key: str) -> None:
    for result, adjusted_p in zip(results, holm_adjust(results, p_key)):
        result[output_key] = adjusted_p


def analysis_plan_length(strict: dict[str, Any]) -> int | float | None:
    if not strict or strict.get("parsable") is False or not bool(strict.get("reachability")):
        return None
    return strict.get("plan_length")


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for problem_id in PROBLEM_IDS:
        for order_name in VARIANT_NAMES:
            variant_dir = DATA_ROOT / problem_id / order_name
            run_dirs = sorted(
                [
                    child
                    for child in variant_dir.glob("[0-9]*")
                    if child.is_dir() and child.name.isdigit()
                ],
                key=lambda child: int(child.name),
            )
            for run_dir in run_dirs:
                for model_name in MODEL_NAMES:
                    result_path = (
                        run_dir
                        / model_output_dir_name(model_name)
                        / "llm_result.json"
                    )
                    if not result_path.exists():
                        continue
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    metrics = payload.get("metrics") or {}
                    strict = metrics.get("strict") or {}
                    legacy = metrics.get("legacy") or {}
                    token_usage = build_token_usage_from_payload(payload)
                    executable = bool(strict.get("executability"))
                    reachable = bool(strict.get("reachability"))
                    rows.append(
                        {
                            "problem": problem_id,
                            "run": int(run_dir.name),
                            "model": model_name,
                            "order": order_name,
                            "reachability": 1.0 if reachable else 0.0,
                            "executability": 1.0 if executable else 0.0,
                            "conditional_reachability": (1.0 if reachable else 0.0)
                            if executable
                            else None,
                            "non_executable_failure": 1.0
                            if strict.get("non_executable_failure") is not None
                            else 0.0,
                            "plan_length": analysis_plan_length(strict),
                            "optimality_ratio": legacy.get("optimality_ratio")
                            if reachable
                            else None,
                            "first_failure_step": strict.get("first_failure_step"),
                            "prompt_tokens": token_usage["prompt_tokens"],
                            "completion_tokens": token_usage["completion_tokens"],
                            "reasoning_completion_tokens": token_usage[
                                "reasoning_completion_tokens"
                            ],
                            "raw_completion_tokens": token_usage["raw_completion_tokens"],
                            "total_tokens": token_usage["total_tokens"],
                            "duration_sec": payload.get("duration_sec"),
                        }
                    )
    return rows


def build_pairs(
    rows: list[dict[str, Any]],
    model_name: str,
    metric: str,
    baseline_order: str,
    compared_order: str,
    *,
    require_both_executable: bool = False,
) -> list[tuple[float, float]]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["model"] != model_name:
            continue
        by_key[(row["problem"], row["run"])][row["order"]] = row

    pairs: list[tuple[float, float]] = []
    for values in by_key.values():
        if baseline_order not in values or compared_order not in values:
            continue
        baseline = values[baseline_order]
        compared = values[compared_order]
        if require_both_executable and not (
            baseline["executability"] == 1.0 and compared["executability"] == 1.0
        ):
            continue
        baseline_value = baseline.get(metric)
        compared_value = compared.get(metric)
        if baseline_value is None or compared_value is None:
            continue
        pairs.append((float(baseline_value), float(compared_value)))
    return pairs


def mcnemar_result(
    rows: list[dict[str, Any]],
    model_name: str,
    metric: str,
    baseline_order: str,
    compared_order: str,
    *,
    require_both_executable: bool = False,
) -> dict[str, Any]:
    pairs = build_pairs(
        rows,
        model_name,
        metric,
        baseline_order,
        compared_order,
        require_both_executable=require_both_executable,
    )
    n00 = n01 = n10 = n11 = 0
    for baseline_value, compared_value in pairs:
        baseline = int(baseline_value)
        compared = int(compared_value)
        if baseline == 0 and compared == 0:
            n00 += 1
        elif baseline == 0 and compared == 1:
            n01 += 1
        elif baseline == 1 and compared == 0:
            n10 += 1
        elif baseline == 1 and compared == 1:
            n11 += 1

    sample_size = n00 + n01 + n10 + n11
    p_value = exact_mcnemar_p(n01, n10)
    return {
        "model": model_name,
        "metric": metric,
        "test": "exact_mcnemar",
        "baseline_order": baseline_order,
        "compared_order": compared_order,
        "n_pairs": sample_size,
        "n00_both_fail": n00,
        "n01_compared_only_success": n01,
        "n10_baseline_only_success": n10,
        "n11_both_success": n11,
        "baseline_mean": (n10 + n11) / sample_size if sample_size else None,
        "compared_mean": (n01 + n11) / sample_size if sample_size else None,
        "risk_difference": (n01 - n10) / sample_size if sample_size else None,
        "matched_odds_ratio": None if n10 == 0 else n01 / n10,
        "matched_odds_ratio_haldane": (n01 + 0.5) / (n10 + 0.5),
        "p_value": p_value,
        "requires_both_executable": require_both_executable,
    }


def conditional_binary_result(
    rows: list[dict[str, Any]],
    model_name: str,
    metric: str,
    baseline_order: str,
    compared_order: str,
) -> dict[str, Any]:
    baseline_values = [
        float(row[metric])
        for row in rows
        if row["model"] == model_name
        and row["order"] == baseline_order
        and row.get(metric) is not None
    ]
    compared_values = [
        float(row[metric])
        for row in rows
        if row["model"] == model_name
        and row["order"] == compared_order
        and row.get(metric) is not None
    ]
    baseline_successes = sum(1 for value in baseline_values if value == 1.0)
    compared_successes = sum(1 for value in compared_values if value == 1.0)
    baseline_n = len(baseline_values)
    compared_n = len(compared_values)
    baseline_failures = baseline_n - baseline_successes
    compared_failures = compared_n - compared_successes
    baseline_mean = baseline_successes / baseline_n if baseline_n else None
    compared_mean = compared_successes / compared_n if compared_n else None
    odds_denominator = compared_failures * baseline_successes

    return {
        "model": model_name,
        "metric": metric,
        "test": "fisher_exact_conditional_proportions",
        "baseline_order": baseline_order,
        "compared_order": compared_order,
        "baseline_n": baseline_n,
        "compared_n": compared_n,
        "baseline_successes": baseline_successes,
        "baseline_failures": baseline_failures,
        "compared_successes": compared_successes,
        "compared_failures": compared_failures,
        "baseline_mean": baseline_mean,
        "compared_mean": compared_mean,
        "risk_difference": (
            compared_mean - baseline_mean
            if baseline_mean is not None and compared_mean is not None
            else None
        ),
        "odds_ratio": (
            None
            if odds_denominator == 0
            else (compared_successes * baseline_failures) / odds_denominator
        ),
        "odds_ratio_haldane": (
            (compared_successes + 0.5)
            * (baseline_failures + 0.5)
            / ((compared_failures + 0.5) * (baseline_successes + 0.5))
        ),
        "p_value": fisher_exact_two_sided_p(
            baseline_successes,
            baseline_failures,
            compared_successes,
            compared_failures,
        ),
        "requires_both_executable": False,
    }


def numeric_result(
    rows: list[dict[str, Any]],
    model_name: str,
    metric: str,
    baseline_order: str,
    compared_order: str,
) -> dict[str, Any]:
    pairs = build_pairs(rows, model_name, metric, baseline_order, compared_order)
    baseline_values = [baseline for baseline, _ in pairs]
    compared_values = [compared for _, compared in pairs]
    differences = [compared - baseline for baseline, compared in pairs]
    t_result = paired_t_test(differences)
    permutation_p = sign_flip_permutation_p(differences)
    baseline_mean = statistics.mean(baseline_values) if baseline_values else None
    compared_mean = statistics.mean(compared_values) if compared_values else None
    mean_diff = t_result["mean_diff"]
    return {
        "model": model_name,
        "metric": metric,
        "test": "paired_t_and_sign_flip_permutation",
        "baseline_order": baseline_order,
        "compared_order": compared_order,
        "n_pairs": len(pairs),
        "baseline_mean": baseline_mean,
        "compared_mean": compared_mean,
        "mean_difference": mean_diff,
        "percent_difference_vs_baseline": (
            mean_diff / baseline_mean
            if mean_diff is not None and baseline_mean not in (None, 0)
            else None
        ),
        "sd_difference": t_result["sd_diff"],
        "t_statistic": t_result["t"],
        "degrees_of_freedom": t_result["df"],
        "p_value_t_test": t_result["p_value"],
        "p_value_sign_flip_permutation": permutation_p,
        "cohens_dz": t_result["cohens_dz"],
    }


def problem_level_result(
    rows: list[dict[str, Any]],
    model_name: str,
    metric: str,
    baseline_order: str,
    compared_order: str,
    *,
    require_both_executable: bool = False,
) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["model"] != model_name:
            continue
        if row["order"] not in {baseline_order, compared_order}:
            continue
        by_key[(row["problem"], row["order"])].append(row)

    problem_records: list[dict[str, Any]] = []
    differences: list[float] = []
    baseline_values: list[float] = []
    compared_values: list[float] = []

    for problem_id in PROBLEM_IDS:
        baseline_rows = by_key.get((problem_id, baseline_order), [])
        compared_rows = by_key.get((problem_id, compared_order), [])
        if not baseline_rows or not compared_rows:
            continue

        if require_both_executable:
            executable_run_ids = {
                row["run"]
                for row in baseline_rows
                if row["executability"] == 1.0
            } & {
                row["run"]
                for row in compared_rows
                if row["executability"] == 1.0
            }
            baseline_rows = [row for row in baseline_rows if row["run"] in executable_run_ids]
            compared_rows = [row for row in compared_rows if row["run"] in executable_run_ids]

        baseline_metric_values = [
            float(row[metric])
            for row in baseline_rows
            if row.get(metric) is not None
        ]
        compared_metric_values = [
            float(row[metric])
            for row in compared_rows
            if row.get(metric) is not None
        ]
        if not baseline_metric_values or not compared_metric_values:
            continue

        baseline_mean = sum(baseline_metric_values) / len(baseline_metric_values)
        compared_mean = sum(compared_metric_values) / len(compared_metric_values)
        difference = compared_mean - baseline_mean
        problem_records.append(
            {
                "problem": problem_id,
                "baseline_mean": baseline_mean,
                "compared_mean": compared_mean,
                "difference": difference,
                "baseline_count": len(baseline_metric_values),
                "compared_count": len(compared_metric_values),
            }
        )
        differences.append(difference)
        baseline_values.append(baseline_mean)
        compared_values.append(compared_mean)

    ci_low, ci_high = bootstrap_mean_ci(differences)
    mean_difference = statistics.mean(differences) if differences else None
    return {
        "model": model_name,
        "metric": metric,
        "test": "problem_level_paired_sign_flip_with_bootstrap_ci",
        "baseline_order": baseline_order,
        "compared_order": compared_order,
        "n_problems": len(differences),
        "baseline_mean": statistics.mean(baseline_values) if baseline_values else None,
        "compared_mean": statistics.mean(compared_values) if compared_values else None,
        "mean_difference": mean_difference,
        "percent_difference_vs_baseline": (
            mean_difference / statistics.mean(baseline_values)
            if mean_difference is not None and baseline_values and statistics.mean(baseline_values) != 0
            else None
        ),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value_sign_flip_permutation": sign_flip_permutation_p(differences),
        "problem_records": problem_records,
        "requires_both_executable": require_both_executable,
    }


def problem_level_order_summary(
    rows: list[dict[str, Any]],
    model_name: str,
    metric: str,
    order_name: str,
) -> dict[str, Any]:
    values_by_problem: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["model"] != model_name or row["order"] != order_name:
            continue
        value = row.get(metric)
        if value is not None:
            values_by_problem[row["problem"]].append(float(value))

    problem_means = [
        sum(values_by_problem[problem_id]) / len(values_by_problem[problem_id])
        for problem_id in PROBLEM_IDS
        if values_by_problem.get(problem_id)
    ]
    return {
        "model": model_name,
        "metric": metric,
        "order": order_name,
        "n_problems": len(problem_means),
        "problem_level_mean": statistics.mean(problem_means)
        if problem_means
        else None,
    }


def empirical_best_order(
    rows: list[dict[str, Any]],
    model_name: str,
    metric: str,
    orders: list[str],
) -> dict[str, Any]:
    summaries = [
        problem_level_order_summary(rows, model_name, metric, order_name)
        for order_name in orders
    ]
    available = [
        summary for summary in summaries if summary["problem_level_mean"] is not None
    ]
    if not available:
        return {
            "order": None,
            "problem_level_mean": None,
            "summaries": summaries,
        }

    order_position = {order_name: index for index, order_name in enumerate(VARIANT_NAMES)}
    best = max(
        available,
        key=lambda summary: (
            summary["problem_level_mean"],
            -order_position.get(summary["order"], len(VARIANT_NAMES)),
        ),
    )
    return {
        "order": best["order"],
        "problem_level_mean": best["problem_level_mean"],
        "summaries": summaries,
    }


def disp3_problem_level_results(
    rows: list[dict[str, Any]],
    model_name: str,
    baseline_orders: list[str],
    available_orders: list[str],
) -> list[dict[str, Any]]:
    best = empirical_best_order(rows, model_name, DISP3_METRIC, available_orders)
    results = [
        problem_level_result(
            rows,
            model_name,
            DISP3_METRIC,
            baseline_order,
            DISP3_ORDER,
        )
        for baseline_order in baseline_orders
    ]
    apply_holm_adjustment(
        results,
        "p_value_sign_flip_permutation",
        "p_value_sign_flip_permutation_holm",
    )
    for result in results:
        result["comparison_family"] = "disp3_vs_all_problem_level"
        result["empirical_best_order"] = best["order"]
        result["empirical_best_mean"] = best["problem_level_mean"]
        result["is_empirical_best_comparison"] = (
            result["baseline_order"] == best["order"]
        )
    return results


def disp3_run_level_mcnemar_results(
    rows: list[dict[str, Any]],
    model_name: str,
    baseline_orders: list[str],
) -> list[dict[str, Any]]:
    results = [
        mcnemar_result(
            rows,
            model_name,
            DISP3_METRIC,
            baseline_order,
            DISP3_ORDER,
        )
        for baseline_order in baseline_orders
    ]
    apply_holm_adjustment(results, "p_value", "p_value_holm")
    for result in results:
        result["comparison_family"] = "disp3_vs_all_run_level_mcnemar"
    return results


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def strip_problem_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in result.items() if key != "problem_records"}
        for result in rows
    ]


def comparison_label(result: dict[str, Any]) -> str:
    return f"{result['baseline_order']} -> {result['compared_order']}"


def markdown_report(model_name: str, payload: dict[str, Any]) -> str:
    lines = [
        f"# Statistical Tests: {model_name}",
        "",
        "Primary baseline order: `canonical`.",
        "Canonical compared orders: "
        + ", ".join(f"`{order_name}`" for order_name in payload["compared_orders"])
        + ".",
        "",
        "Pairing unit for McNemar and numeric tests: `(problem, run)` within this model. Conditional reachability is summarized per order among executable plans only.",
        "",
        "## Binary Metrics",
        "",
        "Exact McNemar test is used for binary outcomes. `b` means compared order succeeds while baseline fails; `c` means baseline succeeds while compared order fails. Effect size is reported as risk difference and matched odds ratio.",
        "",
    ]
    for result in payload["binary_tests"]:
        lines.append(
            "| metric | comparison | n | baseline | compared | b | c | risk diff | matched OR | p | p Holm |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        break
    if payload["binary_tests"]:
        for result in payload["binary_tests"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        result["metric"],
                        comparison_label(result),
                        str(result["n_pairs"]),
                        format_number(result["baseline_mean"]),
                        format_number(result["compared_mean"]),
                        str(result["n01_compared_only_success"]),
                        str(result["n10_baseline_only_success"]),
                        format_number(result["risk_difference"]),
                        format_number(result["matched_odds_ratio"]),
                        format_number(result["p_value"], 6),
                        format_number(result["p_value_holm"], 6),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Conditional Binary Metrics",
            "",
            "`conditional_reachability` is computed as goal reached among executable plans for each order separately. Non-executable plans are excluded from that order's denominator. The comparison table uses Fisher's exact test on those executable-plan counts.",
            "",
        ]
    )
    if payload["conditional_binary_tests"]:
        lines.append(
            "| metric | comparison | baseline n | compared n | baseline | compared | baseline success/fail | compared success/fail | risk diff | OR | p | p Holm |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for result in payload["conditional_binary_tests"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        result["metric"],
                        comparison_label(result),
                        str(result["baseline_n"]),
                        str(result["compared_n"]),
                        format_number(result["baseline_mean"]),
                        format_number(result["compared_mean"]),
                        f"{result['baseline_successes']}/{result['baseline_failures']}",
                        f"{result['compared_successes']}/{result['compared_failures']}",
                        format_number(result["risk_difference"]),
                        format_number(result["odds_ratio"]),
                        format_number(result["p_value"], 6),
                        format_number(result["p_value_holm"], 6),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Numeric Metrics",
            "",
            "Numeric metrics use paired t-test plus paired sign-flip permutation p-value. Effect size is Cohen's dz: mean paired difference divided by the standard deviation of paired differences.",
            "",
        ]
    )
    if payload["numeric_tests"]:
        lines.append(
            "| metric | comparison | n | baseline mean | compared mean | mean diff | % diff | dz | p t-test | p perm | p perm Holm |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for result in payload["numeric_tests"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        result["metric"],
                        comparison_label(result),
                        str(result["n_pairs"]),
                        format_number(result["baseline_mean"]),
                        format_number(result["compared_mean"]),
                        format_number(result["mean_difference"]),
                        format_number(result["percent_difference_vs_baseline"]),
                        format_number(result["cohens_dz"]),
                        format_number(result["p_value_t_test"], 6),
                        format_number(result["p_value_sign_flip_permutation"], 6),
                        format_number(result["p_value_sign_flip_permutation_holm"], 6),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Problem-Level Tests",
            "",
            "Runs are averaged within each problem first. The test unit is the problem, not an individual run. `mean diff` is compared minus baseline, with a paired sign-flip permutation p-value and a bootstrap 95% CI over problems.",
            "",
        ]
    )
    if payload["problem_level_tests"]:
        lines.append(
            "| metric | comparison | n problems | baseline mean | compared mean | mean diff | 95% CI | p perm |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for result in payload["problem_level_tests"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        result["metric"],
                        comparison_label(result),
                        str(result["n_problems"]),
                        format_number(result["baseline_mean"]),
                        format_number(result["compared_mean"]),
                        format_number(result["mean_difference"]),
                        f"[{format_number(result['ci95_low'])}, {format_number(result['ci95_high'])}]",
                        format_number(result["p_value_sign_flip_permutation"], 6),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Disp 3 vs All Orders",
            "",
            "`reachability` is the primary metric. Each non-`disp_3` order is used as the baseline and `disp_3` is the compared order; negative mean differences mean lower reachability for `disp_3`.",
            "",
            "Primary test: problem-level paired sign-flip permutation test with bootstrap 95% CI over problems. Holm adjustment is applied across the six problem-level comparisons within this model.",
            "",
        ]
    )
    if payload["disp3_problem_level_tests"]:
        lines.append(
            "| comparison | n problems | baseline mean | disp_3 mean | mean diff | 95% CI | p perm | p perm Holm | empirical best baseline |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        )
        for result in payload["disp3_problem_level_tests"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        comparison_label(result),
                        str(result["n_problems"]),
                        format_number(result["baseline_mean"]),
                        format_number(result["compared_mean"]),
                        format_number(result["mean_difference"]),
                        f"[{format_number(result['ci95_low'])}, {format_number(result['ci95_high'])}]",
                        format_number(result["p_value_sign_flip_permutation"], 6),
                        format_number(result["p_value_sign_flip_permutation_holm"], 6),
                        "yes" if result["is_empirical_best_comparison"] else "",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Additional analysis: run-level exact McNemar tests on paired `(problem, run)` reachability outcomes.",
            "",
        ]
    )
    if payload["disp3_run_level_mcnemar_tests"]:
        lines.append(
            "| comparison | n | baseline | disp_3 | b | c | risk diff | matched OR | p | p Holm |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for result in payload["disp3_run_level_mcnemar_tests"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        comparison_label(result),
                        str(result["n_pairs"]),
                        format_number(result["baseline_mean"]),
                        format_number(result["compared_mean"]),
                        str(result["n01_compared_only_success"]),
                        str(result["n10_baseline_only_success"]),
                        format_number(result["risk_difference"]),
                        format_number(result["matched_odds_ratio"]),
                        format_number(result["p_value"], 6),
                        format_number(result["p_value_holm"], 6),
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def build_model_payload(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    model_rows = [row for row in rows if row["model"] == model_name]
    run_ids = sorted({row["run"] for row in model_rows})
    available_orders = {row["order"] for row in model_rows}
    compared_orders = [
        order_name
        for order_name in COMPARED_ORDERS
        if order_name in available_orders
    ]
    available_order_list = [
        order_name for order_name in VARIANT_NAMES if order_name in available_orders
    ]
    disp3_baseline_orders = [
        order_name
        for order_name in DISP3_BASELINE_ORDERS
        if DISP3_ORDER in available_orders and order_name in available_orders
    ]

    binary_tests: list[dict[str, Any]] = []
    for metric in BINARY_METRICS:
        metric_results = [
            mcnemar_result(rows, model_name, metric, BASELINE_ORDER, order_name)
            for order_name in compared_orders
        ]
        apply_holm_adjustment(metric_results, "p_value", "p_value_holm")
        binary_tests.extend(metric_results)

    conditional_binary_tests: list[dict[str, Any]] = []
    for metric in CONDITIONAL_BINARY_METRICS:
        metric_results = [
            conditional_binary_result(
                rows,
                model_name,
                metric,
                BASELINE_ORDER,
                order_name,
            )
            for order_name in compared_orders
        ]
        apply_holm_adjustment(metric_results, "p_value", "p_value_holm")
        conditional_binary_tests.extend(metric_results)

    numeric_tests: list[dict[str, Any]] = []
    for metric in NUMERIC_METRICS:
        metric_results = [
            numeric_result(rows, model_name, metric, BASELINE_ORDER, order_name)
            for order_name in compared_orders
        ]
        apply_holm_adjustment(metric_results, "p_value_t_test", "p_value_t_test_holm")
        apply_holm_adjustment(
            metric_results,
            "p_value_sign_flip_permutation",
            "p_value_sign_flip_permutation_holm",
        )
        numeric_tests.extend(metric_results)

    problem_level_tests: list[dict[str, Any]] = []
    for metric in BINARY_METRICS:
        problem_level_tests.extend(
            problem_level_result(
                rows,
                model_name,
                metric,
                BASELINE_ORDER,
                order_name,
            )
            for order_name in compared_orders
        )

    disp3_problem_level_tests = disp3_problem_level_results(
        rows,
        model_name,
        disp3_baseline_orders,
        available_order_list,
    )
    disp3_run_level_mcnemar_tests = disp3_run_level_mcnemar_results(
        rows,
        model_name,
        disp3_baseline_orders,
    )
    for metric in CONDITIONAL_BINARY_METRICS:
        problem_level_tests.extend(
            problem_level_result(
                rows,
                model_name,
                metric,
                BASELINE_ORDER,
                order_name,
            )
            for order_name in compared_orders
        )
    for metric in NUMERIC_METRICS:
        problem_level_tests.extend(
            problem_level_result(
                rows,
                model_name,
                metric,
                BASELINE_ORDER,
                order_name,
            )
            for order_name in compared_orders
        )

    return {
        "model": model_name,
        "generated_from": str(DATA_ROOT),
        "row_count": len(model_rows),
        "problem_count": len({row["problem"] for row in model_rows}),
        "run_ids": run_ids,
        "orders": VARIANT_NAMES,
        "baseline_order": BASELINE_ORDER,
        "compared_orders": compared_orders,
        "disp3_order": DISP3_ORDER,
        "disp3_baseline_orders": disp3_baseline_orders,
        "method_summary": {
            "binary": "Exact McNemar test on paired binary outcomes.",
            "conditional_binary": "Conditional reachability uses per-order executable-plan denominators and Fisher's exact test on those counts.",
            "numeric": "Paired t-test and paired sign-flip permutation test.",
            "problem_level": "Runs are averaged per problem before a paired sign-flip permutation test; bootstrap CI resamples problems.",
            "disp3_vs_all": "Primary reachability test compares each non-disp_3 order against disp_3 with problem-level sign-flip tests and Holm adjustment; run-level McNemar is included as additional evidence.",
            "multiple_comparisons": "Holm adjustment is applied across canonical-baseline comparisons for each model/metric/test family.",
        },
        "metric_definitions": {
            **BINARY_METRICS,
            **CONDITIONAL_BINARY_METRICS,
            **NUMERIC_METRICS,
        },
        "binary_tests": binary_tests,
        "conditional_binary_tests": conditional_binary_tests,
        "numeric_tests": numeric_tests,
        "problem_level_tests": problem_level_tests,
        "disp3_problem_level_tests": disp3_problem_level_tests,
        "disp3_run_level_mcnemar_tests": disp3_run_level_mcnemar_tests,
    }


def main() -> None:
    rows = load_rows()
    (OUTPUT_DIR / "input_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    index_lines = [
        "# Statistical Testing",
        "",
        "This folder contains reproducible order-effect tests for the saved local `llm_result.json` files. No model/API calls are made.",
        "",
        "McNemar and numeric tests are paired within each model by `(problem, run)`: each compared order is matched with `canonical`.",
        "Problem-level tests average runs inside each problem before testing the paired problem differences.",
        "A separate `disp_3` degradation check uses `reachability` only: each non-`disp_3` order is compared against `disp_3`, with the problem-level sign-flip test as the primary test and run-level McNemar as additional evidence.",
        "",
        "Binary metrics use exact McNemar tests. Conditional reachability uses executable-plan denominators per order and Fisher's exact test. Numeric metrics use paired t-tests and sign-flip permutation tests.",
        "",
        "## Metric Semantics",
        "",
        "- `parsable`: strict VAL parse-quality flag saved in `metrics.strict`; it is not itself a statistical-test metric.",
        "- `reachability`: 1 when strict VAL reports `Plan valid`, otherwise 0.",
        "- `executability`: 1 when strict VAL reports successful execution or a valid plan, otherwise 0.",
        "- `non_executable_failure`: 1 when strict validation records a parse error, state execution error, or validator timeout; otherwise 0.",
        "- `conditional_reachability`: reachability among executable plans only. Non-executable plans are excluded from that order's denominator.",
        "- `plan_length`: analysis length used by tests and graphs. For `parsable=true` and `reachability=true`, this is strict VAL plan length from the saved log. For `parsable=false`, `reachability=false`, or missing validation metrics, this is missing and excluded from numeric summaries.",
        "- `optimality_ratio`: legacy VAL cost divided by reference optimal cost; only defined for plans that reached the goal.",
        "- `execution_progress`: strict validation progress. Parse errors are 0; state-execution errors use `first_failure_step / (plan_length + 1)` when available; other strict outcomes with a known plan length are 1.",
        "- `first_failure_step`: first failed execution step parsed from strict VAL output; only defined for state execution failures where VAL reports the step.",
        "- `prompt_tokens`, `completion_tokens`, `reasoning_completion_tokens`, `raw_completion_tokens`, `total_tokens`: provider token usage normalized from the saved response payload.",
        "- `duration_sec`: saved model-call duration in seconds, when available.",
        "",
        "Files:",
    ]

    for model_name in MODEL_NAMES:
        payload = build_model_payload(rows, model_name)
        slug = model_output_dir_name(model_name)
        json_path = OUTPUT_DIR / f"{slug}_stats.json"
        md_path = OUTPUT_DIR / f"{slug}_stats.md"
        csv_path = OUTPUT_DIR / f"{slug}_tests.csv"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        md_path.write_text(markdown_report(model_name, payload), encoding="utf-8")
        write_csv(
            csv_path,
            payload["binary_tests"]
            + payload["conditional_binary_tests"]
            + payload["numeric_tests"],
        )
        write_csv(
            OUTPUT_DIR / f"{slug}_problem_level_tests.csv",
            strip_problem_records(payload["problem_level_tests"]),
        )
        write_csv(
            OUTPUT_DIR / f"{slug}_disp3_vs_all_tests.csv",
            strip_problem_records(payload["disp3_problem_level_tests"])
            + payload["disp3_run_level_mcnemar_tests"],
        )
        index_lines.append(
            f"- `{json_path.name}` / `{md_path.name}` / `{csv_path.name}` / "
            f"`{slug}_problem_level_tests.csv` / `{slug}_disp3_vs_all_tests.csv`"
        )

    (OUTPUT_DIR / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Wrote statistical test outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
