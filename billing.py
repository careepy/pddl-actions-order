"""Local, transparent estimates of OpenRouter request costs.

These are configured-provider list prices in USD per million tokens, captured
on 2026-09-17. They are estimates: OpenRouter billing can differ when a
provider changes prices or prompt caching is applied.
"""

from __future__ import annotations

from typing import Any

from token_usage import build_token_usage_from_payload


PRICING_AS_OF = "2026-09-17"
MODEL_PRICES_USD_PER_MILLION = {
    "deepseek/deepseek-v4-flash": {"input": 0.05, "output": 0.12},
    "gpt-oss-120b": {"input": 0.037, "output": 0.17},
    "nemotron-3-super": {"input": 0.085, "output": 0.40},
}


def estimate_cost_usd(payload: dict[str, Any], model: str) -> float | None:
    """Estimate a request's billed cost from its saved token-usage payload."""

    prices = MODEL_PRICES_USD_PER_MILLION.get(model)
    if prices is None:
        return None

    usage = build_token_usage_from_payload(payload)
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    return (
        int(prompt_tokens) * prices["input"]
        + int(completion_tokens) * prices["output"]
    ) / 1_000_000


def pricing_label(model: str) -> str:
    prices = MODEL_PRICES_USD_PER_MILLION.get(model)
    if prices is None:
        return "pricing unavailable"
    return f'${prices["input"]}/M input, ${prices["output"]}/M output (as of {PRICING_AS_OF})'
