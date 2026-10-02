from dataclasses import dataclass


DOMAIN_TYPES = ["logistics"]
TASK_NAMES = ["alpha"]
PROBLEM_IDS = [f"p{index}" for index in range(1, 21)]
MODEL_NAMES = [
    "deepseek/deepseek-v4-flash",
    "gpt-oss-120b",
    "nemotron-3-super",
]


@dataclass(frozen=True, order=True)
class ProblemRef:
    task: str
    problem: str

    @property
    def label(self) -> str:
        return f"{self.task}/{self.problem}"


@dataclass(frozen=True)
class PlanLengthGroup:
    group_id: str
    min_actions: int
    max_actions: int

    def contains(self, action_count: int) -> bool:
        return self.min_actions <= action_count <= self.max_actions


PROBLEM_TYPE_BY_ID = {
    "p1": "s01_l08",
    "p2": "s01_l15",
    "p3": "s18_l22",
    "p4": "s24_l30",
    "p5": "s01_l35",
    "p6": "s01_l47",
    "p7": "s01_l53",
    "p8": "s10_l06",
    "p9": "s43_l09",
    "p10": "s09_l10",
    "p11": "s18_l12",
    "p12": "s18_l14",
    "p13": "s22_l17",
    "p14": "s19_l19",
    "p15": "s19_l24",
    "p16": "s27_l27",
    "p17": "s16_l32",
    "p18": "s16_l38",
    "p19": "s10_l41",
    "p20": "s09_l45",
}
PROBLEM_TYPE_ORDER = [PROBLEM_TYPE_BY_ID[problem_id] for problem_id in PROBLEM_IDS]
PROBLEM_TYPE_LABELS = {
    problem_type: problem_type.upper()
    for problem_type in PROBLEM_TYPE_ORDER
}
REFERENCE_PLAN_ACTION_COUNTS_BY_ID = {
    "p1": 8,
    "p2": 15,
    "p3": 22,
    "p4": 30,
    "p5": 35,
    "p6": 47,
    "p7": 53,
    "p8": 6,
    "p9": 9,
    "p10": 10,
    "p11": 12,
    "p12": 14,
    "p13": 17,
    "p14": 19,
    "p15": 24,
    "p16": 27,
    "p17": 32,
    "p18": 38,
    "p19": 41,
    "p20": 45,
}
PLAN_LENGTH_GROUPS = (
    PlanLengthGroup("1", 0, 16),
    PlanLengthGroup("2", 17, 34),
    PlanLengthGroup("3", 35, 53),
)


def plan_length_group_for_problem(problem_id: str) -> PlanLengthGroup:
    action_count = REFERENCE_PLAN_ACTION_COUNTS_BY_ID[problem_id]
    for group in PLAN_LENGTH_GROUPS:
        if group.contains(action_count):
            return group
    raise ValueError(f"no plan-length group configured for {problem_id} ({action_count} actions)")


def problem_ids_in_plan_length_group(group: PlanLengthGroup) -> tuple[str, ...]:
    return tuple(
        problem_id
        for problem_id in PROBLEM_IDS
        if group.contains(REFERENCE_PLAN_ACTION_COUNTS_BY_ID[problem_id])
    )


PROBLEM_REFS = [
    ProblemRef(task_name, problem_id)
    for task_name in TASK_NAMES
    for problem_id in PROBLEM_IDS
]
