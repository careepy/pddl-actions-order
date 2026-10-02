import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def add_local_venv_site_packages() -> None:
    venv_lib = Path(__file__).with_name("venv") / "lib"
    if not venv_lib.exists():
        return

    current_python = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = venv_lib / current_python / "site-packages"
    if not site_packages.exists():
        return

    site_packages_path = str(site_packages)
    if site_packages_path not in sys.path:
        sys.path.insert(0, site_packages_path)


add_local_venv_site_packages()

from domain_generation import generate_paths, process_domains
from experiment_config import DOMAIN_TYPES, MODEL_NAMES, PROBLEM_IDS, PROBLEM_REFS, TASK_NAMES, ProblemRef
from billing import PRICING_AS_OF, estimate_cost_usd
from manual_model_run import model_output_dir_name
from shuffler import VARIANT_NAMES

DEFAULT_DOMAIN = DOMAIN_TYPES[0]
REFERENCE_ORDER_NAMES = (
    "reference",
    "reference_disp_1",
    "reference_disp_2",
    "reference_reverse",
)
RUN_ORDER_NAMES = [*VARIANT_NAMES, *REFERENCE_ORDER_NAMES]
ORDER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class ModelRunFailure(RuntimeError):
    pass


def normalize_problem_id(problem_id: str) -> str:
    token = problem_id.strip()
    if not token:
        raise ValueError("problem id cannot be empty")
    if token.startswith("p"):
        token = token[1:]
    normalized = f"p{int(token)}"
    if normalized not in PROBLEM_IDS:
        raise ValueError(f"unknown problem id: {problem_id}")
    return normalized


def normalize_problem_refs(tokens: list[str] | None) -> list[ProblemRef]:
    if not tokens:
        return PROBLEM_REFS[:]

    selected: list[ProblemRef] = []
    current_task: str | None = None
    current_task_problem_count = 0

    def add_all_for_current_task() -> None:
        if current_task is None:
            return
        selected.extend(ProblemRef(current_task, problem_id) for problem_id in PROBLEM_IDS)

    for raw_token in tokens:
        token = raw_token.strip().lower()
        if not token:
            continue

        if token in TASK_NAMES:
            if current_task is not None and current_task_problem_count == 0:
                add_all_for_current_task()
            current_task = token
            current_task_problem_count = 0
            continue

        if current_task is None:
            raise ValueError(
                f"problem id '{raw_token}' must follow a task name: "
                f"{', '.join(TASK_NAMES)}"
            )
        selected.append(ProblemRef(current_task, normalize_problem_id(token)))
        current_task_problem_count += 1

    if current_task is not None and current_task_problem_count == 0:
        add_all_for_current_task()

    deduplicated: list[ProblemRef] = []
    seen: set[ProblemRef] = set()
    for problem_ref in selected:
        if problem_ref in seen:
            continue
        deduplicated.append(problem_ref)
        seen.add(problem_ref)
    return deduplicated


def normalize_problem_ids(tokens: list[str] | None) -> list[ProblemRef]:
    return normalize_problem_refs(tokens)


def normalize_order_names(tokens: list[str]) -> list[str]:
    """Accept built-in and separately generated action-order directories."""

    normalized: list[str] = []
    for raw_token in tokens:
        name = raw_token.strip()
        if not name or not ORDER_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"invalid order name: {raw_token!r}; use only letters, digits, and underscores"
            )
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("at least one order is required")
    return normalized


def build_run_commands(
    problem_refs: list[ProblemRef],
    models: list[str],
    orders: list[str],
    runs: int,
    force: bool,
    start_run: int = 1,
) -> list[list[str]]:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if start_run < 1:
        raise ValueError("start_run must be at least 1")

    commands: list[list[str]] = []
    script_path = Path(__file__).with_name("manual_model_run.py")

    for problem_ref in problem_refs:
        problem_dir = Path("materials") / DEFAULT_DOMAIN / problem_ref.task / problem_ref.problem
        problem_path = problem_dir / f"{problem_ref.problem}.pddl"
        optimal_plan_path = problem_dir / f"{problem_ref.problem}.plan"

        for order_name in orders:
            order_dir = problem_dir / order_name
            domain_path = order_dir / "domain.pddl"
            if not domain_path.exists():
                raise FileNotFoundError(
                    f"missing variant domain: {domain_path}. Run `python3 main.py --force` first."
                )

            for run_id in range(start_run, start_run + runs):
                run_dir = order_dir / str(run_id)

                for model_name in models:
                    output_dir = run_dir / model_output_dir_name(model_name)
                    plan_path = output_dir / "llm.plan"
                    if plan_path.exists() and not force:
                        continue

                    command = [
                        sys.executable,
                        str(script_path),
                        "--domain-path",
                        str(domain_path),
                        "--problem-path",
                        str(problem_path),
                        "--optimal-plan-path",
                        str(optimal_plan_path),
                        "--output-dir",
                        str(output_dir),
                        "--model",
                        model_name,
                    ]
                    if force:
                        command.append("--force")
                    commands.append(command)
    return commands


def _status_path_for_command(command: list[str]) -> Path | None:
    try:
        output_dir_index = command.index("--output-dir") + 1
        return Path(command[output_dir_index]) / "run_status.json"
    except (ValueError, IndexError):
        return None


def _value_after_flag(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _output_dir_for_command(command: list[str]) -> Path | None:
    value = _value_after_flag(command, "--output-dir")
    return Path(value) if value is not None else None


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class RunProgress:
    """Console progress and cost estimate for a batch of model calls."""

    def __init__(self, total: int, *, budget_usd: float | None = None) -> None:
        self.total = total
        self.budget_usd = budget_usd
        self.started_at = time.monotonic()
        self.finished = 0
        self.failed = 0
        self.cost_by_model: dict[str, float] = defaultdict(float)
        self.unpriced_completed = 0

    def record(self, command: list[str], *, succeeded: bool) -> None:
        self.finished += 1
        if not succeeded:
            self.failed += 1
        else:
            model = _value_after_flag(command, "--model")
            output_dir = _output_dir_for_command(command)
            result_path = output_dir / "llm_result.json" if output_dir else None
            if model is None or result_path is None or not result_path.exists():
                self.unpriced_completed += 1
            else:
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    cost = estimate_cost_usd(payload, model)
                except (OSError, json.JSONDecodeError):
                    cost = None
                if cost is None:
                    self.unpriced_completed += 1
                else:
                    self.cost_by_model[model] += cost
        self.print()

    def print(self) -> None:
        elapsed = time.monotonic() - self.started_at
        fraction = self.finished / self.total if self.total else 1.0
        filled = int(round(20 * fraction))
        bar = "█" * filled + "░" * (20 - filled)
        eta = elapsed / self.finished * (self.total - self.finished) if self.finished else 0.0
        total_cost = sum(self.cost_by_model.values())
        cost_text = f"~${total_cost:.4f}"
        if self.budget_usd is not None:
            cost_text += f" / ${self.budget_usd:.2f} budget"
        model_costs = ", ".join(
            f"{model}: ${cost:.4f}" for model, cost in sorted(self.cost_by_model.items())
        )
        suffix = f" | {model_costs}" if model_costs else ""
        if self.unpriced_completed:
            suffix += f" | unpriced: {self.unpriced_completed}"
        if self.failed:
            suffix += f" | failed: {self.failed}"
        print(
            f"[{bar}] {self.finished}/{self.total} ({fraction:.1%}) "
            f"| elapsed {_format_elapsed(elapsed)} | ETA {_format_elapsed(eta)} "
            f"| {cost_text}{suffix}",
            flush=True,
        )


def _format_failed_command(index: int, total: int, command: list[str], error: BaseException) -> str:
    status_path = _status_path_for_command(command)
    status_hint = f" (status: {status_path})" if status_path is not None else ""
    return f"[{index}/{total}] ✗ {' '.join(command[1:])} → {error}{status_hint}"


def run_models(
    problem_refs: list[ProblemRef],
    models: list[str],
    orders: list[str],
    runs: int,
    start_run: int = 1,
    jobs: int = 1,
    force: bool = False,
    budget_usd: float | None = None,
) -> None:
    commands = build_run_commands(problem_refs, models, orders, runs, force, start_run=start_run)
    total = len(commands)
    if total == 0:
        print("Nothing to run.")
        return

    progress = RunProgress(total, budget_usd=budget_usd)
    print(f"Cost estimates use configured provider list prices as of {PRICING_AS_OF}.", flush=True)

    if jobs == 1:
        for index, command in enumerate(commands, start=1):
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                progress.record(command, succeeded=False)
                print(_format_failed_command(index, total, command, exc), file=sys.stderr)
                raise
            progress.record(command, succeeded=True)
            print(f"[{index}/{total}] {' '.join(command[1:])}")
    else:
        failures: list[tuple[list[str], BaseException]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_cmd = {
                executor.submit(subprocess.run, cmd, check=True): cmd
                for cmd in commands
            }
            for index, future in enumerate(concurrent.futures.as_completed(future_to_cmd), start=1):
                cmd = future_to_cmd[future]
                try:
                    future.result()
                except Exception as e:
                    failures.append((cmd, e))
                    progress.record(cmd, succeeded=False)
                    print(_format_failed_command(index, total, cmd, e), file=sys.stderr)
                else:
                    progress.record(cmd, succeeded=True)
        if failures:
            raise ModelRunFailure(
                f"{len(failures)} of {total} model jobs failed; inspect each run_status.json for details."
            )


def prepare_with_force(force: bool) -> None:
    generate_paths(DOMAIN_TYPES, PROBLEM_REFS, force=force)
    process_domains(DOMAIN_TYPES, PROBLEM_REFS, force=force)


def report() -> None:
    from materials.stats.run_statistical_tests import main as build_statistical_tests
    from plot_metrics import build_reports

    build_reports(DOMAIN_TYPES, PROBLEM_REFS)
    build_statistical_tests()


def graphs() -> None:
    from plot_metrics import build_reports

    build_reports(DOMAIN_TYPES, PROBLEM_REFS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal planning pipeline.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="When no command is supplied, run the prepare step with force enabled.",
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare materials and domain variants.")
    prepare_parser.add_argument("--force", action="store_true")

    models_run_parser = subparsers.add_parser("models-run", help="Run all requested model jobs.")
    models_run_parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, required=True)
    models_run_parser.add_argument(
        "--problems",
        nargs="*",
        metavar="PROBLEM_SELECTION",
        help="Task-scoped problem list, e.g. `alpha p1 p2`; task alone selects all p1-p20.",
    )
    models_run_parser.add_argument(
        "--orders",
        nargs="+",
        metavar="ORDER",
        required=True,
        help="Action-order directory names; built-in or generated reference-order prefixes.",
    )
    models_run_parser.add_argument("--runs", type=int, default=1)
    models_run_parser.add_argument(
        "--start-run",
        type=int,
        default=1,
        help="First run id to write (default: 1). Use `--start-run 6 --runs 5` for runs 6-10.",
    )
    models_run_parser.add_argument("--jobs", type=int, default=1, help="Number of parallel processes (default: 1)")
    models_run_parser.add_argument(
        "--budget-usd",
        type=float,
        help="Optional informational budget shown in the live progress display.",
    )
    models_run_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("graphs", help="Build technical and Russian graph sets without statistical reports.")
    subparsers.add_parser("report", help="Build graph sets and statistical reports.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        prepare_with_force(force=args.force)
        return

    if args.command == "prepare":
        prepare_with_force(force=args.force)
        return

    if args.command == "models-run":
        if args.runs < 1:
            parser.error("--runs must be at least 1")
        if args.start_run < 1:
            parser.error("--start-run must be at least 1")
        if args.budget_usd is not None and args.budget_usd <= 0:
            parser.error("--budget-usd must be positive")
        try:
            problem_refs = normalize_problem_refs(args.problems)
        except ValueError as exc:
            parser.error(str(exc))
        try:
            orders = normalize_order_names(args.orders)
        except ValueError as exc:
            parser.error(str(exc))
        try:
            run_models(
                problem_refs,
                args.models,
                orders,
                args.runs,
                start_run=args.start_run,
                jobs=args.jobs,
                force=args.force,
                budget_usd=args.budget_usd,
            )
        except ModelRunFailure as exc:
            parser.exit(1, f"{exc}\n")
        except subprocess.CalledProcessError as exc:
            parser.exit(exc.returncode or 1, f"{exc}\n")
        return

    if args.command == "report":
        report()
        return

    if args.command == "graphs":
        graphs()
        return


if __name__ == "__main__":
    main()
