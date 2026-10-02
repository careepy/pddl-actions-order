import math
import re
from typing import Any


TOKENISH_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(round(value))
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            return int(token)
        except ValueError:
            return None
    return None


def get_field(source: Any, *path: str) -> Any:
    current = source
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
            continue
        current = getattr(current, key, None)
    return current


def estimate_text_token_count(text: str | None) -> int:
    if not text:
        return 0
    return len(TOKENISH_PATTERN.findall(text))


def _normalize_completion_split(
    *,
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    total_tokens: Any = None,
    reasoning_completion_tokens: Any = None,
    raw_completion_tokens: Any = None,
    completion_token_breakdown_source: str | None = None,
) -> dict[str, Any]:
    prompt_tokens = coerce_int(prompt_tokens)
    completion_tokens = coerce_int(completion_tokens)
    total_tokens = coerce_int(total_tokens)
    reasoning_completion_tokens = coerce_int(reasoning_completion_tokens)
    raw_completion_tokens = coerce_int(raw_completion_tokens)

    if reasoning_completion_tokens is not None and reasoning_completion_tokens < 0:
        reasoning_completion_tokens = 0
    if raw_completion_tokens is not None and raw_completion_tokens < 0:
        raw_completion_tokens = 0

    if completion_tokens is None:
        if reasoning_completion_tokens is not None and raw_completion_tokens is not None:
            completion_tokens = reasoning_completion_tokens + raw_completion_tokens
        elif reasoning_completion_tokens is not None:
            completion_tokens = reasoning_completion_tokens
        elif raw_completion_tokens is not None:
            completion_tokens = raw_completion_tokens

    if completion_tokens is not None:
        if reasoning_completion_tokens is not None and raw_completion_tokens is None:
            raw_completion_tokens = max(completion_tokens - reasoning_completion_tokens, 0)
        if raw_completion_tokens is not None and reasoning_completion_tokens is None:
            reasoning_completion_tokens = max(completion_tokens - raw_completion_tokens, 0)

        if reasoning_completion_tokens is not None and raw_completion_tokens is not None:
            split_total = reasoning_completion_tokens + raw_completion_tokens
            if split_total != completion_tokens:
                if split_total == 0:
                    reasoning_completion_tokens = 0
                    raw_completion_tokens = completion_tokens
                else:
                    scale = completion_tokens / split_total
                    reasoning_completion_tokens = min(
                        completion_tokens,
                        max(0, int(round(reasoning_completion_tokens * scale))),
                    )
                    raw_completion_tokens = max(completion_tokens - reasoning_completion_tokens, 0)

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_completion_tokens": reasoning_completion_tokens,
        "raw_completion_tokens": raw_completion_tokens,
        "completion_token_breakdown_source": completion_token_breakdown_source,
    }


def _estimate_completion_split_from_text(
    *,
    completion_tokens: int | None,
    reasoning_text: str | None,
    raw_response: str | None,
) -> dict[str, Any]:
    reasoning_estimate = estimate_text_token_count(reasoning_text)
    raw_estimate = estimate_text_token_count(raw_response)
    estimated_total = reasoning_estimate + raw_estimate

    if completion_tokens is not None:
        if estimated_total > 0:
            reasoning_completion_tokens = int(round(completion_tokens * reasoning_estimate / estimated_total))
            reasoning_completion_tokens = min(completion_tokens, max(reasoning_completion_tokens, 0))
            return _normalize_completion_split(
                completion_tokens=completion_tokens,
                reasoning_completion_tokens=reasoning_completion_tokens,
                raw_completion_tokens=completion_tokens - reasoning_completion_tokens,
                completion_token_breakdown_source="estimated_text_ratio",
            )

        if reasoning_text and not raw_response:
            return _normalize_completion_split(
                completion_tokens=completion_tokens,
                reasoning_completion_tokens=completion_tokens,
                raw_completion_tokens=0,
                completion_token_breakdown_source="estimated_reasoning_only",
            )

        if raw_response and not reasoning_text:
            return _normalize_completion_split(
                completion_tokens=completion_tokens,
                reasoning_completion_tokens=0,
                raw_completion_tokens=completion_tokens,
                completion_token_breakdown_source="estimated_raw_only",
            )

        return _normalize_completion_split(completion_tokens=completion_tokens)

    if estimated_total == 0:
        return _normalize_completion_split()

    return _normalize_completion_split(
        completion_tokens=estimated_total,
        reasoning_completion_tokens=reasoning_estimate,
        raw_completion_tokens=raw_estimate,
        completion_token_breakdown_source="estimated_text_only",
    )


def build_token_usage_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_completion_split(
        prompt_tokens=payload.get("prompt_tokens"),
        completion_tokens=payload.get("completion_tokens"),
        total_tokens=payload.get("total_tokens"),
        reasoning_completion_tokens=payload.get("reasoning_completion_tokens", payload.get("reasoning_tokens")),
        raw_completion_tokens=payload.get("raw_completion_tokens"),
        completion_token_breakdown_source=payload.get("completion_token_breakdown_source"),
    )

    if (
        normalized["reasoning_completion_tokens"] is not None
        and normalized["raw_completion_tokens"] is not None
    ):
        if normalized["completion_token_breakdown_source"] is None:
            normalized["completion_token_breakdown_source"] = "provider_usage"
        return normalized

    estimated = _estimate_completion_split_from_text(
        completion_tokens=normalized["completion_tokens"],
        reasoning_text=payload.get("reasoning"),
        raw_response=payload.get("raw_response"),
    )
    for key, value in estimated.items():
        if normalized.get(key) is None and value is not None:
            normalized[key] = value
    if normalized["completion_token_breakdown_source"] is None:
        normalized["completion_token_breakdown_source"] = estimated.get("completion_token_breakdown_source")
    return normalized


def build_token_usage_from_response(
    *,
    response: Any,
    reasoning_text: str | None,
    raw_response: str | None,
) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    reasoning_tokens = None
    for path in (
        ("completion_tokens_details", "reasoning_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
        ("reasoning_tokens",),
    ):
        reasoning_tokens = coerce_int(get_field(usage, *path))
        if reasoning_tokens is not None:
            break

    raw_completion_tokens = None
    for path in (
        ("completion_tokens_details", "text_tokens"),
        ("output_tokens_details", "text_tokens"),
    ):
        raw_completion_tokens = coerce_int(get_field(usage, *path))
        if raw_completion_tokens is not None:
            break

    normalized = _normalize_completion_split(
        prompt_tokens=get_field(usage, "prompt_tokens") or get_field(usage, "input_tokens"),
        completion_tokens=get_field(usage, "completion_tokens") or get_field(usage, "output_tokens"),
        total_tokens=get_field(usage, "total_tokens"),
        reasoning_completion_tokens=reasoning_tokens,
        raw_completion_tokens=raw_completion_tokens,
        completion_token_breakdown_source="provider_usage" if reasoning_tokens is not None or raw_completion_tokens is not None else None,
    )

    if (
        normalized["reasoning_completion_tokens"] is not None
        and normalized["raw_completion_tokens"] is not None
    ):
        return normalized

    estimated = _estimate_completion_split_from_text(
        completion_tokens=normalized["completion_tokens"],
        reasoning_text=reasoning_text,
        raw_response=raw_response,
    )
    for key, value in estimated.items():
        if normalized.get(key) is None and value is not None:
            normalized[key] = value
    if normalized["completion_token_breakdown_source"] is None:
        normalized["completion_token_breakdown_source"] = estimated.get("completion_token_breakdown_source")
    return normalized
