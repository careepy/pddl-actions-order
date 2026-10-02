#!/usr/bin/env python3
"""Recheck saved plans whose original strict VAL run hit its 120 s limit.

No model calls or changes to original ``llm_result.json`` files.  A plan with
an unknown action, malformed action line, or wrong number of arguments is
classified as a plan-format error before invoking VAL.  All other plans are
rerun with the same domain/problem and validator, recording its exit status.
Unexplained crashes and repeated timeouts remain unresolved, never labelled
as execution errors merely because VAL produced no output.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checker import _extract_first_failure_step, _extract_numeric_value, _sanitize_plan_text_for_validation
from experiment_config import PROBLEM_IDS
from shuffler import _extract_action_blocks
from materials.stats.run_order_audit import DATA, MODELS, ORDERS, source_directory

DEFAULT_OUTPUT = ROOT / "materials/stats/validator_recheck.json"
ACTION_LINE = re.compile(r"^\(([a-zA-Z0-9_\-]+)(?:\s+([^()]*?))?\)$")
PARAMETERS = re.compile(r":parameters\s*\(([^()]*)\)", re.IGNORECASE)
PARSE_ERROR = re.compile(
    r"Bad plan description|parse error|failed to parse|Object with unknown type|Error in type-checking",
    re.IGNORECASE,
)


def action_arities(domain: Path) -> dict[str, int]:
    _, blocks, _, _ = _extract_action_blocks(domain.read_text(encoding="utf-8"))
    result = {}
    for name, block in blocks.items():
        match = PARAMETERS.search(block)
        if not match:
            raise ValueError(f"Missing parameters for {name} in {domain}")
        result[name.lower()] = len(re.findall(r"\?[\w-]+", match.group(1)))
    return result


def format_error(plan_text: str, arities: dict[str, int]) -> str | None:
    lines = [line.strip() for line in plan_text.splitlines()
             if line.strip() and not line.lstrip().startswith(";")]
    if not lines:
        return "empty plan"
    for number, line in enumerate(lines, 1):
        match = ACTION_LINE.fullmatch(line)
        if not match:
            return f"line {number}: not a PDDL action"
        name = match.group(1).lower()
        if name not in arities:
            return f"line {number}: unknown action {name}"
        count = len((match.group(2) or "").split())
        if count != arities[name]:
            return f"line {number}: {name} has {count} arguments, expected {arities[name]}"
    return None


def classify_stdout(output: str, returncode: int, plan_length: int) -> dict:
    if returncode < 0 and not output.strip():
        return {"outcome": "unresolved", "reason": f"validator signal {-returncode} with no output"}
    if "Plan valid" in output:
        return {"outcome": "goal_reached", "plan_length": plan_length,
                "strict_final_value": _extract_numeric_value(output)}
    if "Plan executed successfully - checking goal" in output:
        return {"outcome": "executable_no_goal", "plan_length": plan_length}
    if PARSE_ERROR.search(output):
        return {"outcome": "parse_error", "plan_length": plan_length,
                "execution_progress": 0.0}
    step = _extract_first_failure_step(output)
    if step is not None:
        return {"outcome": "execution_error", "plan_length": plan_length,
                "first_failure_step": step,
                "execution_progress": step / (plan_length + 1)}
    return {"outcome": "unresolved", "reason": f"validator exit {returncode} without recognizable result"}


def recheck(path: Path, domain: Path, problem: Path, arities: dict[str, int],
            validator: Path, limit: int) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    strict = raw["metrics"]["strict"]
    if not strict.get("validator_timed_out") and strict.get("non_executable_failure") != "validator_timeout":
        raise ValueError(f"Not an original strict timeout: {path}")
    plan = path.with_name("llm.plan")
    text = plan.read_text(encoding="utf-8")
    sanitized = _sanitize_plan_text_for_validation(text)
    validated_text = sanitized if sanitized is not None else text
    error = format_error(validated_text, arities)
    result = {"path": str(path.relative_to(ROOT)), "plan_length": strict.get("plan_length")}
    if error:
        result.update(outcome="parse_error", reason=error,
                      execution_progress=0.0, method="plan_schema_precheck")
        return result
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".plan", encoding="utf-8") as temp:
        temp.write(validated_text)
        temp.flush()
        try:
            run = subprocess.run([str(validator), "-v", str(domain), str(problem), temp.name],
                                 capture_output=True, text=True, timeout=limit, check=False)
        except subprocess.TimeoutExpired:
            result.update(outcome="unresolved", reason=f"repeated timeout after {limit} s",
                          method="VAL_recheck")
            return result
    output = run.stdout + run.stderr
    result.update(classify_stdout(output, run.returncode, len([line for line in validated_text.splitlines()
                                                               if line.strip() and not line.lstrip().startswith(";")])))
    result.update(method="VAL_recheck", returncode=run.returncode,
                  validator_output_tail=output[-1500:])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.validator.is_file() or args.limit < 1:
        parser.error("Existing VAL binary and positive limit required")
    records = []
    arity_cache = {}
    for task in PROBLEM_IDS:
        for model in MODELS:
            for order in ORDERS:
                base = DATA / task / source_directory(model, order)
                domain = base / "domain.pddl"
                for run in range(1, 16):
                    path = base / str(run) / model.split("/")[-1] / "llm_result.json"
                    if not path.is_file():
                        continue
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    strict = (raw.get("metrics") or {}).get("strict") or {}
                    if not strict.get("validator_timed_out") and strict.get("non_executable_failure") != "validator_timeout":
                        continue
                    if domain not in arity_cache:
                        arity_cache[domain] = action_arities(domain)
                    record = recheck(path, domain, DATA / task / f"{task}.pddl",
                                     arity_cache[domain], args.validator, args.limit)
                    records.append(record)
                    print(f"{len(records)}: {record['outcome']} {record['path']} {record.get('reason', '')}", flush=True)
    if len(records) != 73:
        raise ValueError(f"Expected 73 strict timeouts, found {len(records)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"validator": str(args.validator), "limit_seconds": args.limit,
                                       "cases": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from collections import Counter
    print(dict(Counter(r["outcome"] for r in records)))
    print(args.output)


if __name__ == "__main__":
    main()
