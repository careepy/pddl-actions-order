import argparse
import json
import os
import shutil
import statistics
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:
    fcntl = None

from checker import build_metrics, wrap_plan_text_lines
from billing import PRICING_AS_OF, estimate_cost_usd
from token_usage import build_token_usage_from_payload


def model_output_dir_name(model_name: str) -> str:
    return model_name.split("/")[-1].replace(":", "-").replace(".", "-")


def atomic_write_text(path: Path, content: str) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def update_status(status_path: Path, stage: str, **extra: Any) -> None:
    atomic_write_json(
        status_path,
        {
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            **extra,
        },
    )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def serialize_exception(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }

    for attr_name in ("status_code", "code", "param", "request_id"):
        attr_value = getattr(exc, attr_name, None)
        if attr_value is not None:
            payload[attr_name] = _json_safe(attr_value)

    body = getattr(exc, "body", None)
    if body is not None:
        payload["body"] = _json_safe(body)

    response = getattr(exc, "response", None)
    if response is not None:
        response_status_code = getattr(response, "status_code", None)
        if response_status_code is not None:
            payload["response_status_code"] = _json_safe(response_status_code)
        response_text = getattr(response, "text", None)
        if response_text:
            payload["response_text"] = str(response_text)[:4000]

    return payload


def append_spending(response: dict[str, Any], model: str, domain_path: Path, problem_path: Path) -> None:
    spendings_path = Path("spendings.json")
    lock_path = spendings_path.with_suffix(".lock")
    token_usage = build_token_usage_from_payload(response)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "domain": str(domain_path),
        "problem": str(problem_path),
        "prompt_tokens": token_usage["prompt_tokens"],
        "completion_tokens": token_usage["completion_tokens"],
        "total_tokens": token_usage["total_tokens"],
        "reasoning_completion_tokens": token_usage["reasoning_completion_tokens"],
        "raw_completion_tokens": token_usage["raw_completion_tokens"],
        "completion_token_breakdown_source": token_usage["completion_token_breakdown_source"],
        "duration_sec": response.get("duration_sec"),
        "estimated_cost_usd": estimate_cost_usd(response, model),
        "pricing_as_of": PRICING_AS_OF,
    }

    with open(lock_path, "w", encoding="utf-8") as lock_handle:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if spendings_path.exists():
                try:
                    payload = json.loads(spendings_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = []
            else:
                payload = []

            payload.append(entry)
            atomic_write_json(spendings_path, payload)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _metrics_record_from_payload(payload: dict[str, Any]) -> dict[str, float | None]:
    metrics = payload.get("metrics") or {}
    strict = metrics.get("strict") or {}
    legacy = metrics.get("legacy") or {}
    token_usage = build_token_usage_from_payload(payload)

    executability = bool(strict.get("executability"))
    reachability = bool(strict.get("reachability"))
    return {
        "plan_length": strict.get("plan_length"),
        "executability": float(executability),
        "reachability": float(reachability),
        "conditional_reachability": float(reachability) if executability else None,
        "optimality_ratio": legacy.get("optimality_ratio") if reachability else None,
        "execution_progress": strict.get("execution_progress"),
        "first_failure_step": strict.get("first_failure_step"),
        "non_executable_failure": float(strict.get("non_executable_failure") is not None),
        "prompt_tokens": token_usage["prompt_tokens"],
        "completion_tokens": token_usage["completion_tokens"],
        "total_tokens": token_usage["total_tokens"],
        "reasoning_completion_tokens": token_usage["reasoning_completion_tokens"],
        "raw_completion_tokens": token_usage["raw_completion_tokens"],
    }


def _summarize_metric(values: list[float | None]) -> dict[str, float | int | None]:
    present_values = [float(value) for value in values if value is not None]
    if not present_values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
        }

    return {
        "count": len(present_values),
        "mean": sum(present_values) / len(present_values),
        "std": statistics.pstdev(present_values) if len(present_values) > 1 else 0.0,
    }


def refresh_aggregate_for_model(order_dir: Path, model: str) -> None:
    model_dir_name = model_output_dir_name(model)
    run_dirs = sorted(
        [child for child in order_dir.iterdir() if child.is_dir() and child.name.isdigit()],
        key=lambda child: int(child.name),
    )

    payloads: list[dict[str, Any]] = []
    run_ids: list[int] = []
    for run_dir in run_dirs:
        payload = load_json_dict(run_dir / model_dir_name / "llm_result.json")
        if payload is None:
            continue
        payloads.append(payload)
        run_ids.append(int(run_dir.name))

    if not payloads:
        return

    metric_records = [_metrics_record_from_payload(payload) for payload in payloads]
    aggregate_payload = {
        "model": model,
        "updated_at": datetime.now().isoformat(),
        "run_count": len(run_ids),
        "runs": run_ids,
        "metrics": {
            metric_name: _summarize_metric([record[metric_name] for record in metric_records])
            for metric_name in metric_records[0]
        },
    }

    aggregate_dir = order_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(aggregate_dir / f"{model_dir_name}.json", aggregate_payload)


def build_result_payload(
    *,
    model: str,
    plan_path: Path,
    metrics: dict[str, Any] | None,
    response: dict[str, Any] | None = None,
    postprocess_error: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "plan_file": str(plan_path),
        "metrics": metrics,
    }
    if response is not None:
        payload.update(
            {
                key: value
                for key, value in response.items()
                if key not in {"model", "plan", "plan_file", "metrics", "postprocess_error"}
            }
        )
    if postprocess_error is not None:
        payload["postprocess_error"] = postprocess_error
    return payload


def safe_build_metrics(
    domain_path: Path,
    problem_path: Path,
    plan_path: Path,
    optimal_plan_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        metrics = build_metrics(domain_path, problem_path, plan_path, optimal_plan_path)
        strict = metrics.get("strict") or {}
        plan_text = plan_path.read_text(encoding="utf-8")
        if strict.get("executability") and strict.get("plan_length") == 0 and plan_text.strip():
            normalized_plan = wrap_plan_text_lines(plan_text)
            if normalized_plan is not None:
                atomic_write_text(plan_path, normalized_plan)
                metrics = build_metrics(domain_path, problem_path, plan_path, optimal_plan_path)
        return metrics, None
    except Exception as exc:
        return None, {
            "type": type(exc).__name__,
            "message": str(exc),
        }


def run_model(
    domain_path: Path,
    problem_path: Path,
    optimal_plan_path: Path,
    output_dir: Path,
    model: str,
    force: bool,
) -> None:
    plan_path = output_dir / "llm.plan"
    result_path = output_dir / "llm_result.json"
    status_path = output_dir / "run_status.json"

    if plan_path.exists() and not force:
        return

    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from api_call import call_openrouter

    update_status(status_path, "calling_model", model=model)
    try:
        response = call_openrouter(
            domain_path=domain_path,
            problem_path=problem_path,
            model=model,
            reasoning_enabled=True,
        )
    except KeyboardInterrupt as exc:
        update_status(
            status_path,
            "interrupted",
            model=model,
            failed_stage="calling_model",
            error=serialize_exception(exc),
        )
        raise
    except Exception as exc:
        update_status(
            status_path,
            "failed",
            model=model,
            failed_stage="calling_model",
            error=serialize_exception(exc),
        )
        raise

    atomic_write_text(plan_path, str(response["plan"]))
    update_status(status_path, "plan_written", plan_file=str(plan_path))

    metrics, postprocess_error = safe_build_metrics(domain_path, problem_path, plan_path, optimal_plan_path)
    payload = build_result_payload(
        model=model,
        plan_path=plan_path,
        metrics=metrics,
        response=response,
        postprocess_error=postprocess_error,
    )
    atomic_write_json(result_path, payload)
    update_status(status_path, "result_written", result_file=str(result_path))
    refresh_aggregate_for_model(output_dir.parent.parent, model)

    try:
        append_spending(response, model, domain_path, problem_path)
        update_status(status_path, "completed", spendings_logged=True)
    except Exception as exc:
        update_status(
            status_path,
            "completed_with_spending_error",
            spendings_logged=False,
            spending_error={"type": type(exc).__name__, "message": str(exc)},
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one model for one domain/problem/variant combination.")
    parser.add_argument("--domain-path", required=True)
    parser.add_argument("--problem-path", required=True)
    parser.add_argument("--optimal-plan-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_model(
        domain_path=Path(args.domain_path),
        problem_path=Path(args.problem_path),
        optimal_plan_path=Path(args.optimal_plan_path),
        output_dir=Path(args.output_dir),
        model=args.model,
        force=args.force,
    )


if __name__ == "__main__":
    main()
