import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from token_usage import build_token_usage_from_response


load_dotenv()


MODEL_ALIASES = {
    "deepseek/deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b",
}
MODEL_CONFIG = {
    "deepseek/deepseek-v4-flash": {
        "max_tokens": None,
        "supports_reasoning": True,
        "reasoning_effort": "medium",
        "temperature": 1.0,
        "top_p": 1.0,
        "provider": {
            "order": ["deepinfra"],
            "allow_fallbacks": False,
            "quantizations": ["fp8"],
        },
    },
    "openai/gpt-oss-120b": {
        "max_tokens": None,
        "supports_reasoning": True,
        "reasoning_effort": "medium",
        "temperature": 1.0,
        "top_p": 1.0,
        "provider": {
            "order": ["DeepInfra"],
            "allow_fallbacks": False,
            "quantizations": ["bf16"],
        },
    },
    "nvidia/nemotron-3-super-120b-a12b": {
        "max_tokens": None,
        "supports_reasoning": True,
        "reasoning_effort": "medium",
        "temperature": 1.0,
        "top_p": 0.95,
        "provider": {
            "order": ["deepinfra"],
            "allow_fallbacks": False,
            "quantizations": ["bf16"],
        },
    },
}
RETRYABLE_EXCEPTIONS = (Exception,)
RATE_LIMIT_COOLDOWN_SEC = 60.0
# After a 429, stagger retries for one model so parallel workers do not create
# another simultaneous burst as soon as the cooldown ends.
RATE_LIMIT_RETRY_SPACING_SEC = 2.0
RATE_LIMIT_STATE_FILE = Path(tempfile.gettempdir()) / "paper-action-order-rate-limit.json"


def _model_from_retry_state(retry_state: Any) -> str:
    model = retry_state.kwargs.get("model")
    if isinstance(model, str):
        return model
    if len(retry_state.args) >= 3 and isinstance(retry_state.args[2], str):
        return retry_state.args[2]
    return "default"


@contextmanager
def _locked_rate_limit_state() -> Iterator[dict[str, dict[str, float]]]:
    """Load and atomically update cooldown state shared by worker processes."""

    # VAL runs on Linux/WSL, where flock provides process-wide coordination.
    import fcntl

    RATE_LIMIT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RATE_LIMIT_STATE_FILE.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            try:
                state = json.load(handle)
            except (json.JSONDecodeError, OSError):
                state = {}
            if not isinstance(state, dict):
                state = {}
            yield state
            handle.seek(0)
            handle.truncate()
            json.dump(state, handle)
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _activate_rate_limit_cooldown(model: str) -> None:
    """Block new calls for ``model`` for at least one minute after a 429."""

    now = time.time()
    with _locked_rate_limit_state() as state:
        entry = state.get(model, {})
        existing = float(entry.get("cooldown_until", 0.0)) if isinstance(entry, dict) else 0.0
        cooldown_until = max(existing, now + RATE_LIMIT_COOLDOWN_SEC)
        state[model] = {
            "cooldown_until": cooldown_until,
            "next_request_at": cooldown_until,
        }


def _wait_for_rate_limit_gate(model: str) -> None:
    """Wait for a model-specific cooldown and reserve a staggered retry slot."""

    while True:
        now = time.time()
        with _locked_rate_limit_state() as state:
            entry = state.get(model)
            if not isinstance(entry, dict):
                return
            cooldown_until = float(entry.get("cooldown_until", 0.0))
            next_request_at = float(entry.get("next_request_at", 0.0))
            ready_at = max(cooldown_until, next_request_at)
            if now >= ready_at:
                entry["next_request_at"] = now + RATE_LIMIT_RETRY_SPACING_SEC
                state[model] = entry
                return
        time.sleep(min(ready_at - now, 5.0))


def _before_openrouter_attempt(retry_state: Any) -> None:
    _wait_for_rate_limit_gate(_model_from_retry_state(retry_state))


def _rate_limit_aware_wait(retry_state: Any) -> float:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exception, RateLimitError):
        model = _model_from_retry_state(retry_state)
        _activate_rate_limit_cooldown(model)
        print(
            f"Rate-limited for {model}; pausing new requests for {int(RATE_LIMIT_COOLDOWN_SEC)}s.",
            flush=True,
        )
        return RATE_LIMIT_COOLDOWN_SEC
    return min(45.0, max(8.0, 2.0**retry_state.attempt_number))


def fix_plan_format(plan_text: str) -> str:
    if not plan_text.strip():
        return plan_text

    fixed_lines = []
    for raw_line in plan_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("(") and line.endswith(")"):
            fixed_lines.append(line)
        else:
            fixed_lines.append(f"({line})")
    return "\n".join(fixed_lines)


def _resolve_provider_model(model: str) -> str:
    if model not in MODEL_ALIASES:
        raise ValueError(f"unsupported model: {model}")
    return MODEL_ALIASES[model]


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


@retry(
    stop=stop_after_attempt(5),
    wait=_rate_limit_aware_wait,
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before=_before_openrouter_attempt,
    reraise=True,
)
def call_openrouter(
    domain_path: str | Path,
    problem_path: str | Path,
    model: str = "glm-4.7-flash",
    reasoning_enabled: bool = True,
    fix_plan_format_enabled: bool = False,
) -> dict[str, object]:
    provider_model = _resolve_provider_model(model)
    config = MODEL_CONFIG[provider_model]

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        timeout=1500.0,
    )

    prompt = f"""You are a deterministic PDDL planner.

Goal: produce a valid plan that achieves the goal state with minimal number of actions.

Rules:
- Output ONLY the plan.
- One action per line.
- No extra text.
- No empty lines.
- The plan MUST be complete.
- Do NOT stop early.
- Continue until the goal is achieved.
- Stop immediately after the last action.

Constraints:
- The plan MUST be valid under the domain and problem.
- The plan SHOULD be optimal.
- Use only actions defined in the domain.
- Respect all preconditions and effects.

Domain:
{_read_text(domain_path)}

Problem:
{_read_text(problem_path)}

Return ONLY the plan.
You must give an answer.
Each line must contain exactly one action in PDDL format. Also use brackets "()":
(action-name arg1 arg2 ...)
"""

    extra_body = {}
    reasoning_used = reasoning_enabled and config.get("supports_reasoning", True)
    if reasoning_used:
        extra_body["reasoning"] = {"enabled": True}
        effort = config.get("reasoning_effort")
        if effort is not None:
            extra_body["reasoning"]["effort"] = effort

    create_kwargs = {
        "model": provider_model,
        "messages": [{"role": "user", "content": prompt}],
        "extra_body": extra_body,
        "temperature": config.get("temperature", 0.0),
        "top_p": config.get("top_p", 1.0),
    }
    max_tokens = config.get("max_tokens")
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    for key in ("presence_penalty", "frequency_penalty"):
        if key in config:
            create_kwargs[key] = config[key]

    for key in ("top_k", "min_p", "repetition_penalty", "provider"):
        if key in config:
            extra_body[key] = config[key]

    started_at = time.time()
    response = client.chat.completions.create(**create_kwargs)
    duration = round(time.time() - started_at, 2)

    message = response.choices[0].message
    raw_response = message.content or ""
    plan = fix_plan_format(raw_response) if fix_plan_format_enabled else raw_response

    reasoning_text = getattr(message, "reasoning", "") or ""
    if not reasoning_text and hasattr(message, "reasoning_details"):
        fragments = [
            item.get("text", "")
            for item in message.reasoning_details
            if isinstance(item, dict) and item.get("text")
        ]
        reasoning_text = "\n".join(fragments)
    token_usage = build_token_usage_from_response(
        response=response,
        reasoning_text=reasoning_text,
        raw_response=raw_response,
    )

    return {
        "raw_response": raw_response,
        "plan": plan,
        "reasoning": reasoning_text,
        "reasoning_enabled": reasoning_used,
        "model": model,
        "duration_sec": duration,
        **token_usage,
    }
