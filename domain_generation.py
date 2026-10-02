import json
import shutil
from pathlib import Path

from experiment_config import DOMAIN_TYPES, PROBLEM_REFS, ProblemRef
from shuffler import VARIANT_NAMES, shuffle


def _normalize_problem_refs(problems: list[ProblemRef] | None = None) -> list[ProblemRef]:
    if problems is None:
        return PROBLEM_REFS[:]
    deduplicated: list[ProblemRef] = []
    seen: set[ProblemRef] = set()
    for problem_ref in problems:
        if problem_ref in seen:
            continue
        deduplicated.append(problem_ref)
        seen.add(problem_ref)
    return deduplicated


def _cleanup_problem_variants(problem_dir: Path) -> None:
    for child in problem_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name in VARIANT_NAMES or (child / "domain.pddl").exists():
            shutil.rmtree(child)


def _variants_are_up_to_date(problem_dir: Path, problem_ref: ProblemRef | None = None) -> bool:
    metadata_path = problem_dir / "shuffle_meta.json"
    if not metadata_path.exists():
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    if metadata.get("variants") != VARIANT_NAMES:
        return False
    if problem_ref is not None:
        if metadata.get("task") != problem_ref.task:
            return False
        if metadata.get("problem_id") != problem_ref.problem:
            return False

    variant_orders = metadata.get("variant_orders")
    if not isinstance(variant_orders, dict):
        return False
    if list(variant_orders) != VARIANT_NAMES:
        return False
    if not all(isinstance(order, list) and order for order in variant_orders.values()):
        return False

    plan_action_order = metadata.get("plan_action_order")
    if not isinstance(plan_action_order, list):
        return False
    if not all(isinstance(action_name, str) for action_name in plan_action_order):
        return False

    return all((problem_dir / variant_name / "domain.pddl").exists() for variant_name in VARIANT_NAMES)


def generate_paths(
    domains: list[str] | None = None,
    problems: list[ProblemRef] | None = None,
    force: bool = False,
) -> None:
    domains = DOMAIN_TYPES[:] if domains is None else domains
    problem_refs = _normalize_problem_refs(problems)

    for domain_name in domains:
        source_dir = Path("ipc2023-dataset-main/opt") / domain_name
        target_dir = Path("materials") / domain_name
        target_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            required_paths = [target_dir / "domain.pddl"]
            for problem_ref in problem_refs:
                problem_dir = target_dir / problem_ref.task / problem_ref.problem
                required_paths.extend(
                    [
                        problem_dir / f"{problem_ref.problem}.pddl",
                        problem_dir / f"{problem_ref.problem}.plan",
                    ]
                )

            missing_paths = [path for path in required_paths if not path.exists()]
            if missing_paths:
                missing = "\n".join(str(path) for path in missing_paths[:10])
                raise FileNotFoundError(
                    f"missing prepared materials for domain '{domain_name}'. "
                    "Expected existing files under materials/ because no IPC source "
                    f"directory was found at {source_dir}.\n{missing}"
                )
            continue

        domain_target = target_dir / "domain.pddl"
        if force or not domain_target.exists():
            shutil.copy(source_dir / "domain.pddl", domain_target)

        for problem_ref in problem_refs:
            problem_dir = target_dir / problem_ref.task / problem_ref.problem
            problem_dir.mkdir(parents=True, exist_ok=True)

            for suffix in ("pddl", "plan"):
                source_path = source_dir / problem_ref.task / f"{problem_ref.problem}.{suffix}"
                target_path = problem_dir / f"{problem_ref.problem}.{suffix}"
                if force or not target_path.exists():
                    shutil.copy(source_path, target_path)


def process_domains(
    domains: list[str] | None = None,
    problems: list[ProblemRef] | None = None,
    force: bool = False,
    shuffle_seed: int = 52,
) -> None:
    domains = DOMAIN_TYPES[:] if domains is None else domains
    problem_refs = _normalize_problem_refs(problems)

    for domain_name in domains:
        domain_dir = Path("materials") / domain_name
        domain_path = domain_dir / "domain.pddl"

        for problem_ref in problem_refs:
            problem_dir = domain_dir / problem_ref.task / problem_ref.problem
            if force:
                _cleanup_problem_variants(problem_dir)
                shuffle_meta_path = problem_dir / "shuffle_meta.json"
                if shuffle_meta_path.exists():
                    shuffle_meta_path.unlink()

            if not force and _variants_are_up_to_date(problem_dir, problem_ref):
                continue

            shuffle(
                domain_path=domain_path,
                problem_path=problem_dir / f"{problem_ref.problem}.pddl",
                optimal_plan_path=problem_dir / f"{problem_ref.problem}.plan",
                save_dir=problem_dir,
                seed=shuffle_seed,
                problem_id=problem_ref.problem,
                task_name=problem_ref.task,
            )
