import json
import re
from pathlib import Path


VARIANT_NAMES = [
    "canonical",
    "disp_1",
    "disp_2",
    "disp_3",
    "plan_front",
    "plan_back",
    "plan_scatter",
]
DISPERSION_LEVELS = {
    "disp_1": (1, 3),
    "disp_2": (2, 3),
}

ACTION_PATTERN = re.compile(r"\(\s*:action\s+([^\s)]+)", re.IGNORECASE)
PLAN_ACTION_PATTERN = re.compile(r"^\s*\(\s*([^;\s()]+)", re.IGNORECASE)


def _extract_action_blocks(domain_text: str) -> tuple[str, dict[str, str], str, list[str]]:
    action_blocks: dict[str, str] = {}
    action_order: list[str] = []
    cursor = 0
    text_length = len(domain_text)

    while cursor < text_length:
        start = domain_text.lower().find("(:action", cursor)
        if start == -1:
            break

        block_start = start
        while block_start > 0 and domain_text[block_start - 1] in " \t\n":
            block_start -= 1

        depth = 0
        block_end = start
        while block_end < text_length:
            if domain_text[block_end] == "(":
                depth += 1
            elif domain_text[block_end] == ")":
                depth -= 1
                if depth == 0:
                    block_end += 1
                    break
            block_end += 1

        block = domain_text[block_start:block_end]
        match = ACTION_PATTERN.search(block)
        if match:
            action_name = match.group(1)
            action_blocks[action_name] = block
            action_order.append(action_name)

        cursor = block_end

    if not action_order:
        return domain_text, {}, "", []

    first_block = action_blocks[action_order[0]]
    last_block = action_blocks[action_order[-1]]
    header = domain_text[:domain_text.find(first_block)].rstrip()
    footer_start = domain_text.rfind(last_block) + len(last_block)
    footer = domain_text[footer_start:].lstrip("\n")
    return header, action_blocks, footer, action_order


def build_even_inversion_vector(n: int, inversions: int) -> list[int]:
    if n < 0:
        raise ValueError("n must be non-negative")

    capacities = [n - index - 1 for index in range(n)]
    max_inversions = sum(capacities)
    if inversions < 0 or inversions > max_inversions:
        raise ValueError(f"inversions must be in [0, {max_inversions}]")

    vector = [0] * n
    index = 0
    while inversions > 0:
        if vector[index] < capacities[index]:
            vector[index] += 1
            inversions -= 1
        index = (index + 1) % n
    return vector


def _round_half_up_fraction(value: int, numerator: int, denominator: int) -> int:
    return (2 * value * numerator + denominator) // (2 * denominator)


def _order_from_inversion_vector(items: list[str], inversion_vector: list[int]) -> list[str]:
    if len(items) != len(inversion_vector):
        raise ValueError("items and inversion_vector must have the same length")

    available_positions = list(range(len(items)))
    ordered_items = [""] * len(items)

    for item, inversion_count in zip(items, inversion_vector):
        if inversion_count > len(available_positions) - 1:
            raise ValueError("invalid inversion vector")
        position = available_positions.pop(inversion_count)
        ordered_items[position] = item

    return ordered_items


def kendall_tau_distance(reference_order: list[str], candidate_order: list[str]) -> int:
    if len(reference_order) != len(candidate_order):
        raise ValueError("orders must have the same length")
    if set(reference_order) != set(candidate_order):
        raise ValueError("orders must contain the same items")

    positions = {item: index for index, item in enumerate(reference_order)}
    normalized_candidate = [positions[item] for item in candidate_order]

    distance = 0
    for left_index, left_value in enumerate(normalized_candidate):
        for right_value in normalized_candidate[left_index + 1 :]:
            if left_value > right_value:
                distance += 1
    return distance


def _build_dispersion_order(canonical_order: list[str], numerator: int, denominator: int) -> list[str]:
    reverse_order = list(reversed(canonical_order))
    max_distance = kendall_tau_distance(canonical_order, reverse_order)
    target_distance = _round_half_up_fraction(max_distance, numerator, denominator)
    inversion_vector = build_even_inversion_vector(len(canonical_order), target_distance)
    dispersion_order = _order_from_inversion_vector(canonical_order, inversion_vector)

    if kendall_tau_distance(canonical_order, dispersion_order) != target_distance:
        raise ValueError("constructed dispersion order does not match target Kendall tau distance")

    return dispersion_order


def _extract_plan_action_order(plan_text: str, action_order: list[str]) -> list[str]:
    action_lookup = {action_name.lower(): action_name for action_name in action_order}
    plan_action_order: list[str] = []
    seen: set[str] = set()

    for raw_line in plan_text.splitlines():
        match = PLAN_ACTION_PATTERN.match(raw_line)
        if not match:
            continue

        raw_action_name = match.group(1)
        action_name = action_lookup.get(raw_action_name.lower())
        if action_name is None:
            raise ValueError(f"plan action not found in domain: {raw_action_name}")
        if action_name in seen:
            continue

        plan_action_order.append(action_name)
        seen.add(action_name)

    return plan_action_order


def _build_plan_front_order(action_order: list[str], plan_action_order: list[str]) -> list[str]:
    plan_actions = set(plan_action_order)
    return plan_action_order[:] + [
        action_name
        for action_name in action_order
        if action_name not in plan_actions
    ]


def _build_plan_back_order(action_order: list[str], plan_action_order: list[str]) -> list[str]:
    plan_actions = set(plan_action_order)
    background_order = [
        action_name
        for action_name in action_order
        if action_name not in plan_actions
    ]
    return background_order + list(reversed(plan_action_order))


def _build_plan_scatter_order(action_order: list[str], plan_action_order: list[str]) -> list[str]:
    if not plan_action_order:
        return action_order[:]

    plan_actions = set(plan_action_order)
    background_order = [
        action_name
        for action_name in action_order
        if action_name not in plan_actions
    ]
    scattered_plan_order = list(reversed(plan_action_order))

    scattered_order: list[str] = []
    background_cursor = 0
    for plan_index, action_name in enumerate(scattered_plan_order, start=1):
        target_cursor = _round_half_up_fraction(
            len(background_order),
            plan_index,
            len(scattered_plan_order) + 1,
        )
        scattered_order.extend(background_order[background_cursor:target_cursor])
        scattered_order.append(action_name)
        background_cursor = target_cursor

    scattered_order.extend(background_order[background_cursor:])
    return scattered_order


def _build_variants(action_order: list[str], plan_action_order: list[str]) -> dict[str, list[str]]:
    variants = {
        "canonical": action_order[:],
    }
    for variant_name, (numerator, denominator) in DISPERSION_LEVELS.items():
        variants[variant_name] = _build_dispersion_order(action_order, numerator, denominator)
    variants["disp_3"] = list(reversed(action_order))
    variants["plan_front"] = _build_plan_front_order(action_order, plan_action_order)
    variants["plan_back"] = _build_plan_back_order(action_order, plan_action_order)
    variants["plan_scatter"] = _build_plan_scatter_order(action_order, plan_action_order)
    return variants


def _write_domain_variant(
    header: str,
    action_blocks: dict[str, str],
    footer: str,
    action_order: list[str],
    variant_dir: Path,
) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    domain_path = variant_dir / "domain.pddl"
    content = [header, "\n"]
    for action_name in action_order:
        content.append(action_blocks[action_name])
    content.append(footer)
    domain_path.write_text("".join(content), encoding="utf-8")


def shuffle(
    domain_path: str | Path,
    problem_path: str | Path,
    optimal_plan_path: str | Path,
    save_dir: str | Path,
    seed: int = 52,
    problem_id: str | None = None,
    task_name: str | None = None,
) -> None:
    domain_path = Path(domain_path)
    problem_path = Path(problem_path)
    optimal_plan_path = Path(optimal_plan_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    domain_text = domain_path.read_text(encoding="utf-8")
    optimal_plan_text = optimal_plan_path.read_text(encoding="utf-8")
    header, action_blocks, footer, action_order = _extract_action_blocks(domain_text)
    plan_action_order = _extract_plan_action_order(optimal_plan_text, action_order)

    variants = _build_variants(action_order, plan_action_order)

    for variant_name, variant_order in variants.items():
        _write_domain_variant(header, action_blocks, footer, variant_order, save_dir / variant_name)

    metadata = {
        "seed": seed,
        "problem_id": problem_id or problem_path.stem,
        "task": task_name,
        "variants": list(variants),
        "plan_action_order": plan_action_order,
        "variant_orders": variants,
    }
    (save_dir / "shuffle_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
