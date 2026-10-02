#!/usr/bin/env python3
"""Create action-order variants relative to a reference-plan-based baseline.

The existing experiment's ``canonical`` order is the original declaration
order of actions in the PDDL domain. This script leaves that experiment
untouched and creates a separate four-order family for every selected task:

* ``reference``: unique actions in first-use order in the reference plan,
  followed by actions absent from that plan in original declaration order;
* ``reference_disp_1`` and ``reference_disp_2``: one- and two-thirds of the
  maximum Kendall-tau distance from ``reference``;
* ``reference_reverse``: the complete reversal of ``reference``.

No model calls are made. Existing run directories are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import PROBLEM_IDS
from shuffler import (
    _build_dispersion_order,
    _build_plan_front_order,
    _extract_action_blocks,
    _extract_plan_action_order,
    _write_domain_variant,
    kendall_tau_distance,
)


DEFAULT_DOMAIN = "logistics"
DEFAULT_TASK = "alpha"
VARIANT_NAMES = (
    "reference",
    "reference_disp_1",
    "reference_disp_2",
    "reference_reverse",
)


def _has_saved_runs(variant_dir: Path) -> bool:
    return any(
        child.is_dir() and child.name.isdigit() for child in variant_dir.iterdir()
    )


def _write_variant_safely(
    *,
    header: str,
    action_blocks: dict[str, str],
    footer: str,
    action_order: list[str],
    variant_dir: Path,
    force: bool,
) -> str:
    """Write a domain only when this cannot invalidate saved model runs."""

    domain_path = variant_dir / "domain.pddl"
    if domain_path.exists():
        existing_order = _extract_action_blocks(domain_path.read_text(encoding="utf-8"))[3]
        if existing_order == action_order:
            return "unchanged"
        if _has_saved_runs(variant_dir):
            raise RuntimeError(
                f"refusing to replace {domain_path}: it already has saved run directories. "
                "Choose a new output prefix or move those results first."
            )
        if not force:
            raise RuntimeError(
                f"{domain_path} exists with a different order. Re-run with --force "
                "only if it has no saved model runs."
            )

    _write_domain_variant(header, action_blocks, footer, action_order, variant_dir)
    return "written"


def generate_for_problem(
    *,
    domain_path: Path,
    problem_dir: Path,
    problem_id: str,
    task_name: str,
    prefix: str,
    force: bool,
) -> dict[str, str]:
    """Create reference-relative domains and metadata for one planning task."""

    reference_plan_path = problem_dir / f"{problem_id}.plan"
    if not domain_path.is_file():
        raise FileNotFoundError(f"missing source domain: {domain_path}")
    if not reference_plan_path.is_file():
        raise FileNotFoundError(f"missing reference plan: {reference_plan_path}")

    header, action_blocks, footer, canonical_order = _extract_action_blocks(
        domain_path.read_text(encoding="utf-8")
    )
    if not canonical_order:
        raise ValueError(f"no action schemas found in {domain_path}")

    plan_action_order = _extract_plan_action_order(
        reference_plan_path.read_text(encoding="utf-8"), canonical_order
    )
    reference_order = _build_plan_front_order(canonical_order, plan_action_order)
    variants = {
        f"{prefix}": reference_order,
        f"{prefix}_disp_1": _build_dispersion_order(reference_order, 1, 3),
        f"{prefix}_disp_2": _build_dispersion_order(reference_order, 2, 3),
        f"{prefix}_reverse": list(reversed(reference_order)),
    }

    expected_actions = set(canonical_order)
    if len(expected_actions) != len(canonical_order):
        raise ValueError("source domain has duplicate action-schema names")
    for variant_name, order in variants.items():
        if len(order) != len(canonical_order) or set(order) != expected_actions:
            raise ValueError(f"{variant_name} is not a permutation of the source actions")

    statuses: dict[str, str] = {}
    for variant_name, order in variants.items():
        statuses[variant_name] = _write_variant_safely(
            header=header,
            action_blocks=action_blocks,
            footer=footer,
            action_order=order,
            variant_dir=problem_dir / variant_name,
            force=force,
        )

    max_distance = kendall_tau_distance(reference_order, list(reversed(reference_order)))
    metadata = {
        "domain": domain_path.name,
        "task": task_name,
        "problem_id": problem_id,
        "reference_plan": reference_plan_path.name,
        "definition": (
            "reference = unique action schemas in first-use order in the reference "
            "plan, followed by schemas absent from that plan in source declaration order"
        ),
        "source_action_order": canonical_order,
        "reference_plan_action_order": plan_action_order,
        "variants": list(variants),
        "variant_orders": variants,
        "kendall_tau_distance_from_reference": {
            name: kendall_tau_distance(reference_order, order)
            for name, order in variants.items()
        },
        "maximum_kendall_tau_distance": max_distance,
    }
    metadata_path = problem_dir / f"{prefix}_order_meta.json"
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    if metadata_path.exists() and metadata_path.read_text(encoding="utf-8") != metadata_text:
        if not force:
            raise RuntimeError(
                f"{metadata_path} exists with different content. Re-run with --force "
                "after confirming that the new definition is intended."
            )
    metadata_path.write_text(metadata_text, encoding="utf-8")
    return statuses


def parse_problem_ids(tokens: list[str] | None) -> list[str]:
    if not tokens:
        return PROBLEM_IDS[:]
    problem_ids: list[str] = []
    for token in tokens:
        normalized = f"p{int(token.removeprefix('p'))}"
        if normalized not in PROBLEM_IDS:
            raise ValueError(f"unknown problem: {token}")
        if normalized not in problem_ids:
            problem_ids.append(normalized)
    return problem_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate safe reference-plan-relative PDDL action orders."
    )
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--problems",
        nargs="*",
        metavar="P",
        help="Problem IDs, e.g. p1 p2. Defaults to p1-p20.",
    )
    parser.add_argument(
        "--prefix",
        default="reference",
        help="Directory-name prefix for variants (default: reference).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace generated domain files only when their variant directories "
            "contain no saved numeric run directories."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.prefix.replace("_", "").isalnum():
        raise ValueError("--prefix may contain only letters, digits, and underscores")

    problem_ids = parse_problem_ids(args.problems)
    domain_dir = PROJECT_ROOT / "materials" / args.domain
    domain_path = domain_dir / "domain.pddl"
    for problem_id in problem_ids:
        problem_dir = domain_dir / args.task / problem_id
        statuses = generate_for_problem(
            domain_path=domain_path,
            problem_dir=problem_dir,
            problem_id=problem_id,
            task_name=args.task,
            prefix=args.prefix,
            force=args.force,
        )
        summary = ", ".join(f"{name}: {status}" for name, status in statuses.items())
        print(f"{args.task}/{problem_id}: {summary}")


if __name__ == "__main__":
    main()
