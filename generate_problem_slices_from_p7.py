#!/usr/bin/env python3
"""Regenerate logistics alpha problem slices from the current p7 source.

The current p1-p20 set is derived from p7 as fixed slices of the p7 optimal
plan.  Most non-prefix problems use all facts that become true during the
slice as the goal.  The prefix/milestone problems keep manually reduced goal
sets, and p10 keeps an alternate valid ordering for its 10-step plan.

By default this script does not write into materials/.  It compares generated
files with the current tree.  Use --output-root to write a separate generated
copy, or --write-in-place to refresh materials/logistics/alpha explicitly.
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DOMAIN_PATH = PROJECT_ROOT / "materials" / "logistics" / "domain.pddl"
DEFAULT_ALPHA_DIR = PROJECT_ROOT / "materials" / "logistics" / "alpha"
SOURCE_PROBLEM_ID = "p7"


@dataclass(frozen=True)
class ProblemSpec:
    problem_id: str
    start_action: int
    length: int
    goal_mode: str = "delta"
    goal_facts: tuple[tuple[str, ...], ...] = ()
    plan_override: tuple[str, ...] = ()
    exclude_goal_facts: tuple[tuple[str, ...], ...] = ()

    @property
    def label(self) -> str:
        return f"s{self.start_action:02d}_l{self.length:02d}"


def fact(text: str) -> tuple[str, ...]:
    return tuple(text.strip("()").split())


PROBLEM_SPECS: tuple[ProblemSpec, ...] = (
    ProblemSpec(
        "p1",
        1,
        8,
        goal_mode="manual",
        goal_facts=tuple(
            fact(item)
            for item in [
                "(candidate_prepared shift_candidate_alpha)",
                "(candidate_created destination_availability_alpha)",
                "(candidate_created shift_candidate_beta)",
                "(vehicle_allocated origin_availability_alpha)",
                "(candidate_vehicle_linked origin_availability_alpha vehicle_unit_alpha)",
            ]
        ),
    ),
    ProblemSpec(
        "p2",
        1,
        15,
        goal_mode="manual",
        goal_facts=tuple(
            fact(item)
            for item in [
                "(dispatch_ready shift_candidate_alpha)",
                "(operator_available driver_operator_alpha)",
                "(candidate_created destination_availability_alpha)",
                "(candidate_created shift_candidate_beta)",
                "(vehicle_allocated origin_availability_alpha)",
                "(candidate_vehicle_linked origin_availability_alpha vehicle_unit_alpha)",
            ]
        ),
    ),
    ProblemSpec("p3", 18, 22),
    ProblemSpec("p4", 24, 30),
    ProblemSpec(
        "p5",
        1,
        35,
        goal_mode="manual",
        goal_facts=tuple(
            fact(item)
            for item in [
                "(dispatched shift_candidate_alpha)",
                "(dispatched origin_availability_alpha)",
                "(candidate_created shift_candidate_beta)",
                "(compliance_certificate_available compliance_certificate_alpha)",
                "(route_task_profile_available route_task_profile_alpha)",
            ]
        ),
    ),
    ProblemSpec(
        "p6",
        1,
        47,
        goal_mode="manual",
        goal_facts=tuple(
            fact(item)
            for item in [
                "(dispatched shift_candidate_alpha)",
                "(dispatched origin_availability_alpha)",
                "(dispatched destination_availability_alpha)",
                "(candidate_staged_for_loading shift_candidate_beta)",
                "(candidate_operator_linked shift_candidate_beta driver_operator_alpha)",
                "(candidate_skill_certificate_linked shift_candidate_beta skill_certificate_alpha)",
                "(compliance_certificate_available compliance_certificate_alpha)",
            ]
        ),
    ),
    ProblemSpec(
        "p7",
        1,
        53,
        goal_mode="manual",
        goal_facts=tuple(
            fact(item)
            for item in [
                "(dispatched origin_availability_alpha)",
                "(dispatched destination_availability_alpha)",
                "(dispatched shift_candidate_alpha)",
                "(dispatched shift_candidate_beta)",
            ]
        ),
    ),
    ProblemSpec("p8", 10, 6),
    ProblemSpec("p9", 43, 9),
    ProblemSpec(
        "p10",
        9,
        10,
        plan_override=(
            "(assign_operator_resource shift_candidate_alpha driver_operator_alpha)",
            "(assign_compliance_certificate shift_candidate_alpha compliance_certificate_alpha)",
            "(apply_operational_constraint shift_candidate_alpha operational_constraint_alpha route_task_profile_alpha)",
            "(confirm_operator_alignment shift_candidate_alpha driver_operator_alpha)",
            "(verify_compliance_certificate shift_candidate_alpha compliance_certificate_alpha)",
            "(finalize_candidate_after_compliance_validation shift_candidate_alpha)",
            "(allocate_dispatch_slot_for_candidate shift_candidate_alpha dispatch_slot_alpha route_task_profile_alpha)",
            "(allocate_vehicle_unit destination_availability_alpha vehicle_unit_beta)",
            "(execute_schedule_dispatch shift_candidate_alpha vehicle_unit_gamma dispatch_slot_alpha)",
            "(detach_task_profile shift_candidate_alpha route_task_profile_alpha)",
        ),
        exclude_goal_facts=(
            fact("(candidate_operator_linked shift_candidate_alpha driver_operator_alpha)"),
        ),
    ),
    ProblemSpec("p11", 18, 12),
    ProblemSpec("p12", 18, 14),
    ProblemSpec("p13", 22, 17),
    ProblemSpec("p14", 19, 19),
    ProblemSpec("p15", 19, 24),
    ProblemSpec("p16", 27, 27),
    ProblemSpec("p17", 16, 32),
    ProblemSpec("p18", 16, 38),
    ProblemSpec("p19", 10, 41),
    ProblemSpec("p20", 9, 45),
)


def strip_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def tokenize(text: str) -> list[str]:
    return re.findall(r"\(|\)|[^\s()]+", strip_comments(text))


def parse_sexpr(text: str) -> Any:
    tokens = tokenize(text)
    position = 0

    def parse_one() -> Any:
        nonlocal position
        if tokens[position] == "(":
            position += 1
            values = []
            while tokens[position] != ")":
                values.append(parse_one())
            position += 1
            return values

        token = tokens[position]
        position += 1
        return token

    parsed = []
    while position < len(tokens):
        parsed.append(parse_one())
    return parsed[0] if len(parsed) == 1 else parsed


def facts_from_goal(expr: list[Any]) -> set[tuple[str, ...]]:
    if expr and expr[0] == "and":
        return {tuple(item) for item in expr[1:]}
    return {tuple(expr)}


def parse_problem(path: Path) -> tuple[str, set[tuple[str, ...]], set[tuple[str, ...]]]:
    root = parse_sexpr(path.read_text(encoding="utf-8"))
    name = root[1]
    init: set[tuple[str, ...]] = set()
    goal: set[tuple[str, ...]] = set()
    for part in root[2:]:
        if not isinstance(part, list) or not part:
            continue
        if part[0] == ":init":
            init = {tuple(item) for item in part[1:]}
        elif part[0] == ":goal":
            goal = facts_from_goal(part[1])
    return name, init, goal


def parse_domain_actions(path: Path) -> dict[str, tuple[list[str], list[tuple[str, ...]], list[tuple[str, ...]]]]:
    root = parse_sexpr(path.read_text(encoding="utf-8"))
    actions: dict[str, tuple[list[str], list[tuple[str, ...]], list[tuple[str, ...]]]] = {}
    for part in root:
        if not isinstance(part, list) or not part or part[0] != ":action":
            continue

        name = part[1]
        parameters: list[Any] = []
        effect: Any = None
        index = 2
        while index < len(part):
            if part[index] == ":parameters":
                parameters = part[index + 1]
                index += 2
            elif part[index] == ":effect":
                effect = part[index + 1]
                index += 2
            else:
                index += 1

        variables = [
            token for token in parameters if isinstance(token, str) and token.startswith("?")
        ]
        add_effects: list[tuple[str, ...]] = []
        delete_effects: list[tuple[str, ...]] = []

        def collect(expr: Any) -> None:
            if isinstance(expr, str) or not expr:
                return
            if expr[0] == "and":
                for child in expr[1:]:
                    collect(child)
            elif expr[0] == "not":
                delete_effects.append(tuple(expr[1]))
            else:
                add_effects.append(tuple(expr))

        collect(effect)
        actions[name] = (variables, add_effects, delete_effects)
    return actions


def parse_plan_lines(lines: list[str]) -> list[tuple[str, list[str], str]]:
    actions: list[tuple[str, list[str], str]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        match = re.fullmatch(r"\(([^()\s]+)(.*?)\)", line)
        if not match:
            raise ValueError(f"cannot parse plan action: {raw_line}")
        actions.append((match.group(1), match.group(2).split(), line))
    return actions


def parse_plan(path: Path) -> list[tuple[str, list[str], str]]:
    return parse_plan_lines(path.read_text(encoding="utf-8").splitlines())


def ground(pattern: tuple[str, ...], bindings: dict[str, str]) -> tuple[str, ...]:
    return tuple(bindings.get(token, token) for token in pattern)


def apply_action(
    state: set[tuple[str, ...]],
    action: tuple[str, list[str], str],
    action_schemas: dict[str, tuple[list[str], list[tuple[str, ...]], list[tuple[str, ...]]]],
) -> set[tuple[str, ...]]:
    action_name, args, _line = action
    variables, add_effects, delete_effects = action_schemas[action_name]
    bindings = dict(zip(variables, args))
    next_state = set(state)
    for effect in delete_effects:
        next_state.discard(ground(effect, bindings))
    for effect in add_effects:
        next_state.add(ground(effect, bindings))
    return next_state


def simulate_states(
    initial_state: set[tuple[str, ...]],
    plan: list[tuple[str, list[str], str]],
    action_schemas: dict[str, tuple[list[str], list[tuple[str, ...]], list[tuple[str, ...]]]],
) -> list[set[tuple[str, ...]]]:
    states = [set(initial_state)]
    for action in plan:
        states.append(apply_action(states[-1], action, action_schemas))
    return states


def format_fact(item: tuple[str, ...]) -> str:
    return "(" + " ".join(item) + ")"


def extract_source_blocks(source_problem_text: str) -> tuple[str, str, str]:
    objects_start = source_problem_text.index("  (:objects")
    init_start = source_problem_text.index("  (:init", objects_start)
    objects_block = source_problem_text[objects_start:init_start].rstrip()

    init_body_start = source_problem_text.index("\n", init_start) + 1
    init_body_end = source_problem_text.index("\n  )\n  (:goal", init_body_start)
    init_body = source_problem_text[init_body_start:init_body_end]

    domain_match = re.search(r"^\s*\(:domain\s+([^)]+)\)", source_problem_text, re.MULTILINE)
    if not domain_match:
        raise ValueError("source problem does not contain a :domain line")
    domain_name = domain_match.group(1).strip()
    return domain_name, objects_block, init_body


def problem_name(spec: ProblemSpec, full_plan_length: int) -> str:
    if spec.problem_id in {"p3", "p4"}:
        middle = f"slice_{spec.length}"
    elif spec.start_action == 1 and spec.length == full_plan_length:
        middle = f"full_{spec.length}"
    elif spec.start_action == 1:
        middle = f"prefix_{spec.length}"
    else:
        middle = spec.label
    return f"logistics_route_dispatch_alpha_{middle}_{spec.problem_id}"


def generated_plan_lines(
    spec: ProblemSpec,
    full_plan: list[tuple[str, list[str], str]],
) -> list[str]:
    if spec.plan_override:
        return list(spec.plan_override)
    start_index = spec.start_action - 1
    return [line for *_prefix, line in full_plan[start_index : start_index + spec.length]]


def generated_goal(
    spec: ProblemSpec,
    start_state: set[tuple[str, ...]],
    plan: list[tuple[str, list[str], str]],
    action_schemas: dict[str, tuple[list[str], list[tuple[str, ...]], list[tuple[str, ...]]]],
) -> tuple[tuple[str, ...], ...]:
    if spec.goal_mode == "manual":
        return spec.goal_facts

    end_state = simulate_states(start_state, plan, action_schemas)[-1]
    excluded = set(spec.exclude_goal_facts)
    return tuple(sorted((end_state - start_state) - excluded))


def render_problem(
    spec: ProblemSpec,
    domain_name: str,
    objects_block: str,
    source_init_body: str,
    initial_state: set[tuple[str, ...]],
    goal_facts: tuple[tuple[str, ...], ...],
    full_plan_length: int,
) -> str:
    if spec.start_action == 1:
        init_body = source_init_body
    else:
        init_body = "\n".join(f"    {format_fact(item)}" for item in sorted(initial_state))

    goal_body = "\n".join(f"      {format_fact(item)}" for item in goal_facts)
    return (
        f"(define (problem {problem_name(spec, full_plan_length)})\n"
        f"  (:domain {domain_name})\n"
        f"{objects_block}\n"
        "  (:init\n"
        f"{init_body}\n"
        "  )\n"
        "  (:goal\n"
        "    (and\n"
        f"{goal_body}\n"
        "    )\n"
        "  )\n"
        ")\n"
    )


def build_generated_files(domain_path: Path, alpha_dir: Path) -> dict[Path, str]:
    source_problem_path = alpha_dir / SOURCE_PROBLEM_ID / f"{SOURCE_PROBLEM_ID}.pddl"
    source_plan_path = alpha_dir / SOURCE_PROBLEM_ID / f"{SOURCE_PROBLEM_ID}.plan"
    source_problem_text = source_problem_path.read_text(encoding="utf-8")
    domain_name, objects_block, source_init_body = extract_source_blocks(source_problem_text)

    _source_name, source_init, _source_goal = parse_problem(source_problem_path)
    full_plan = parse_plan(source_plan_path)
    action_schemas = parse_domain_actions(domain_path)
    full_states = simulate_states(source_init, full_plan, action_schemas)

    generated: dict[Path, str] = {}
    for spec in PROBLEM_SPECS:
        start_state = full_states[spec.start_action - 1]
        plan_lines = generated_plan_lines(spec, full_plan)
        parsed_plan = parse_plan_lines(plan_lines)
        goal = generated_goal(spec, start_state, parsed_plan, action_schemas)

        problem_rel = Path(spec.problem_id) / f"{spec.problem_id}.pddl"
        plan_rel = Path(spec.problem_id) / f"{spec.problem_id}.plan"
        generated[problem_rel] = render_problem(
            spec,
            domain_name,
            objects_block,
            source_init_body,
            start_state,
            goal,
            len(full_plan),
        )
        generated[plan_rel] = "\n".join(plan_lines) + "\n"
    return generated


def compare_generated(generated: dict[Path, str], alpha_dir: Path) -> int:
    mismatches = 0
    for relative_path, expected_text in sorted(generated.items()):
        current_path = alpha_dir / relative_path
        current_text = current_path.read_text(encoding="utf-8")
        if current_text == expected_text:
            continue
        mismatches += 1
        print(f"DIFF {relative_path}")
        for line in difflib.unified_diff(
            current_text.splitlines(),
            expected_text.splitlines(),
            fromfile=str(current_path),
            tofile=f"generated/{relative_path}",
            lineterm="",
        ):
            print(line)
    return mismatches


def write_generated(generated: dict[Path, str], output_root: Path) -> None:
    for relative_path, text in generated.items():
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate fixed alpha p1-p20 slices from materials/logistics/alpha/p7."
    )
    parser.add_argument("--domain-path", type=Path, default=DEFAULT_DOMAIN_PATH)
    parser.add_argument("--alpha-dir", type=Path, default=DEFAULT_ALPHA_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Write generated p*/p*.pddl and p*/p*.plan under this directory.",
    )
    parser.add_argument(
        "--write-in-place",
        action="store_true",
        help="Write generated files into --alpha-dir. Use only when intentionally refreshing materials.",
    )
    parser.add_argument(
        "--clean-output-root",
        action="store_true",
        help="Remove --output-root before writing generated files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Accepted for provenance. Generation is deterministic and does not use randomness.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the success message when generated files match current materials.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output_root and args.write_in_place:
        raise SystemExit("choose either --output-root or --write-in-place, not both")
    if args.clean_output_root and not args.output_root:
        raise SystemExit("--clean-output-root requires --output-root")

    generated = build_generated_files(args.domain_path, args.alpha_dir)

    if args.output_root:
        if args.clean_output_root and args.output_root.exists():
            shutil.rmtree(args.output_root)
        write_generated(generated, args.output_root)
        if not args.quiet:
            print(f"wrote generated files to {args.output_root}")
        return 0

    if args.write_in_place:
        write_generated(generated, args.alpha_dir)
        if not args.quiet:
            print(f"wrote generated files to {args.alpha_dir}")
        return 0

    mismatches = compare_generated(generated, args.alpha_dir)
    if mismatches:
        print(f"{mismatches} generated files differ from current materials")
        return 1
    if not args.quiet:
        seed_note = "" if args.seed is None else f" (seed {args.seed} ignored: deterministic)"
        print(f"generated files match current materials{seed_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
