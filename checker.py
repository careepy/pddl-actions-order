import os
import re
import signal
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


VALIDATE_TIMEOUT_SEC = 120
PLAN_COST_PATTERNS = (
    r"Optimal cost:\s*([0-9]+(?:\.[0-9]+)?)",
    r"Plan cost:\s*([0-9]+(?:\.[0-9]+)?)",
    r"Final value:\s*([0-9]+(?:\.[0-9]+)?)",
    r"Value:\s*([0-9]+(?:\.[0-9]+)?)",
)
PLAN_LENGTH_PATTERN = re.compile(r"Plan size:\s*(\d+)")
FIRST_FAILURE_PATTERNS = (
    re.compile(r"unsatisfied precondition at time\s+(\d+)", re.IGNORECASE),
    re.compile(r"Checking next happening \(time\s+(\d+)\)\s*[\r\n]+Plan failed", re.IGNORECASE),
)
ACTION_LINE_PATTERN = re.compile(r"^\s*\(([^;].*)\)\s*$")
VALIDATION_ACTION_LINE_PATTERN = re.compile(r"^\s*\([^();\s]+(?:\s+[^();\s]+)*\)\s*$")
CALL_STYLE_ACTION_PATTERN = re.compile(r"^([^()\s]+)\((.*)\)$")


def _validator_binary() -> str:
    """Return the VAL executable configured for this process.

    ``VAL_BIN`` permits a project run to use a locally built VAL binary without
    requiring a system-wide symlink.  The historical ``validate``-on-PATH
    behaviour remains the default.
    """
    return os.environ.get("VAL_BIN", "validate")


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _extract_numeric_value(text: str) -> float | None:
    for pattern in PLAN_COST_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _read_plan_actions(plan_path: str | Path) -> list[str]:
    actions: list[str] = []
    for raw_line in _read_text(plan_path).splitlines():
        line = raw_line.strip()
        if ACTION_LINE_PATTERN.match(line):
            actions.append(" ".join(line.lower().split()))
    return actions


def wrap_plan_text_lines(plan_text: str) -> str | None:
    action_lines: list[str] = []

    for raw_line in plan_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        call_style_match = CALL_STYLE_ACTION_PATTERN.match(line)
        if call_style_match:
            line = f"{call_style_match.group(1)} {call_style_match.group(2).strip()}"
        if line.startswith("(") and line.endswith(")"):
            action_lines.append(line)
        else:
            action_lines.append(f"({line})")

    if not action_lines:
        return None
    return "\n".join(action_lines) + "\n"


def _sanitize_plan_text_for_validation(plan_text: str) -> str | None:
    action_lines: list[str] = []
    saw_non_action_content = False
    for raw_line in plan_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if VALIDATION_ACTION_LINE_PATTERN.match(line):
            action_lines.append(line)
        else:
            saw_non_action_content = True

    if saw_non_action_content and action_lines:
        return "\n".join(action_lines) + "\n"
    return None


def _extract_plan_length(output: str, plan_path: str | Path | None = None) -> int | None:
    match = PLAN_LENGTH_PATTERN.search(output)
    if match:
        return int(match.group(1))
    if plan_path is not None:
        try:
            return len(_read_plan_actions(plan_path))
        except FileNotFoundError:
            return None
    return None


def _extract_first_failure_step(output: str) -> int | None:
    for pattern in FIRST_FAILURE_PATTERNS:
        match = pattern.search(output)
        if match:
            return int(match.group(1))
    return None


def _execution_progress(
    failure_code: str | None,
    first_failure_step: int | None,
    plan_length: int | None,
) -> float | None:
    if failure_code == "parse_error":
        return 0.0
    if plan_length is None:
        return None
    if failure_code == "state_execution_error":
        if first_failure_step is None:
            return None
        return first_failure_step / (plan_length + 1)
    return 1.0


def _run_validator(
    flag: str,
    domain_path: str | Path,
    problem_path: str | Path,
    plan_path: str | Path,
) -> tuple[str, bool]:
    validation_plan_path = str(plan_path)
    sanitized_plan_handle = None
    timed_out = False
    sanitized_plan_text = _sanitize_plan_text_for_validation(_read_text(plan_path))
    if sanitized_plan_text is not None:
        sanitized_plan_handle = tempfile.NamedTemporaryFile(mode="w+", suffix=".plan", encoding="utf-8")
        sanitized_plan_handle.write(sanitized_plan_text)
        sanitized_plan_handle.flush()
        validation_plan_path = sanitized_plan_handle.name

    try:
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as output_handle:
            process = subprocess.Popen(
                [_validator_binary(), flag, str(domain_path), str(problem_path), validation_plan_path],
                stdout=output_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

            deadline = time.monotonic() + VALIDATE_TIMEOUT_SEC
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.kill()
                    break
                time.sleep(0.25)

            output_handle.flush()
            output_handle.seek(0)
            output = output_handle.read()
    finally:
        if sanitized_plan_handle is not None:
            sanitized_plan_handle.close()

    if validation_plan_path != str(plan_path):
        output = output.replace(validation_plan_path, str(plan_path))

    if timed_out:
        output = f"{output}\nValidator timed out after {VALIDATE_TIMEOUT_SEC} seconds.\n"
    return output, timed_out


def strict_validation(domain_path: str | Path, problem_path: str | Path, plan_path: str | Path) -> dict[str, Any]:
    output, timed_out = _run_validator("-v", domain_path, problem_path, plan_path)
    parsable = not bool(re.search(r"Bad plan description|parse error|failed to parse", output, re.IGNORECASE))
    executability = "Plan executed successfully - checking goal" in output or "Plan valid" in output
    reachability = "Plan valid" in output

    if timed_out:
        failure_code = "validator_timeout"
    elif not parsable:
        failure_code = "parse_error"
    elif not executability:
        failure_code = "state_execution_error"
    else:
        failure_code = None

    plan_length = _extract_plan_length(output, plan_path=plan_path)
    first_failure_step = _extract_first_failure_step(output) if failure_code == "state_execution_error" else None

    return {
        "parsable": parsable,
        "plan_length": plan_length,
        "executability": executability,
        "reachability": reachability,
        "execution_progress": _execution_progress(failure_code, first_failure_step, plan_length),
        "first_failure_step": first_failure_step,
        "non_executable_failure": failure_code,
        "strict_final_value": _extract_numeric_value(output) if reachability else None,
        "validator_timed_out": timed_out,
        "validator_stdout_strict": output,
    }


def legacy_validation(domain_path: str | Path, problem_path: str | Path, plan_path: str | Path) -> dict[str, Any]:
    output, timed_out = _run_validator("-c", domain_path, problem_path, plan_path)
    return {
        "cost": _extract_numeric_value(output),
        "goal_reached": "Plan valid" in output,
        "validator_timed_out": timed_out,
        "validator_stdout_legacy": output,
    }


@lru_cache(maxsize=None)
def _load_reference_plan_stats(
    domain_path: str,
    problem_path: str,
    optimal_plan_path: str,
) -> dict[str, Any]:
    optimal_path = Path(optimal_plan_path)
    optimal_text = _read_text(optimal_path)
    optimal_cost = _extract_numeric_value(optimal_text)
    if optimal_cost is None:
        optimal_cost = legacy_validation(domain_path, problem_path, optimal_path)["cost"]

    return {
        "optimal_cost": optimal_cost,
        "optimal_plan_length": len(_read_plan_actions(optimal_path)),
    }


def build_metrics(
    domain_path: str | Path,
    problem_path: str | Path,
    plan_path: str | Path,
    optimal_plan_path: str | Path | None = None,
) -> dict[str, Any]:
    strict = strict_validation(domain_path, problem_path, plan_path)
    reachability = strict["reachability"]

    if reachability:
        legacy_raw = legacy_validation(domain_path, problem_path, plan_path)
    else:
        legacy_raw = {
            "cost": None,
            "goal_reached": False,
            "validator_timed_out": False,
            "validator_stdout_legacy": None,
            "skipped_because_strict_not_reached": True,
        }

    optimal_cost = None
    optimal_plan_length = None
    if optimal_plan_path and Path(optimal_plan_path).exists():
        reference = _load_reference_plan_stats(
            str(Path(domain_path)),
            str(Path(problem_path)),
            str(Path(optimal_plan_path)),
        )
        optimal_cost = reference["optimal_cost"]
        optimal_plan_length = reference["optimal_plan_length"]

    legacy_cost = legacy_raw["cost"] if reachability else None
    optimality_ratio = None
    if reachability and legacy_cost is not None and optimal_cost not in (None, 0):
        optimality_ratio = legacy_cost / optimal_cost

    return {
        "strict": strict,
        "legacy": {
            "cost": legacy_cost,
            "optimality_ratio": optimality_ratio,
            "validator_stdout_legacy": legacy_raw["validator_stdout_legacy"],
            "validator_timed_out": legacy_raw.get("validator_timed_out", False),
            "skipped_because_strict_not_reached": legacy_raw.get("skipped_because_strict_not_reached", False),
        },
        "reference": {
            "optimal_cost": optimal_cost,
            "optimal_plan_length": optimal_plan_length,
        },
    }
