"""Claude client (Anthropic API) implementing :class:`~src.ai.llm.LLMClient`.

Uses the official ``anthropic`` SDK. Credentials come from the environment
(``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile), never from config.

Every request opts into server-side refusal fallbacks (``fallbacks="default"``)
so a rare policy decline is retried on a fallback model inside the same call,
and caches the system prompt, which carries the large, stable track knowledge.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import structlog

_log = structlog.get_logger("ai.anthropic")

_FALLBACK_BETA = "server-side-fallback-2026-07-01"

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# USD per million tokens: (input, output, cache read, 5-minute cache write).
# First-party list prices as of 2026-11; used only to *estimate* spend in logs.
PRICES_PER_MTOK: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}


@dataclass
class Usage:
    """Token counts for one or more requests, with an estimated cost."""

    calls: int = 0
    input_tokens: int = 0  # uncached input
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0  # includes thinking
    cost_usd: float = 0.0

    def add(self, other: Usage) -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cost_usd"] = round(self.cost_usd, 5)
        return data


def estimate_usage(model: str, usage: Any, *, fast: bool = False) -> Usage:
    """Build a :class:`Usage` (with estimated cost) from an API ``usage`` object."""

    def count(name: str) -> int:
        return int(getattr(usage, name, 0) or 0)

    out = Usage(
        calls=1,
        input_tokens=count("input_tokens"),
        cache_read_tokens=count("cache_read_input_tokens"),
        cache_write_tokens=count("cache_creation_input_tokens"),
        output_tokens=count("output_tokens"),
    )
    prices = PRICES_PER_MTOK.get(model)
    if prices is not None:
        p_in, p_out, p_read, p_write = prices
        multiplier = 2.0 if fast else 1.0
        out.cost_usd = (
            multiplier
            * (
                out.input_tokens * p_in
                + out.output_tokens * p_out
                + out.cache_read_tokens * p_read
                + out.cache_write_tokens * p_write
            )
            / 1_000_000
        )
    return out


class ClaudeRefusalError(RuntimeError):
    """Claude (and its fallback) declined the request."""


class AnthropicClient:
    """Async Claude client with a cached system prompt.

    Args:
        model: Claude model id, e.g. ``claude-opus-5-5``.
        effort: Default ``output_config.effort`` for requests.
        speed: ``"standard"`` or ``"fast"`` (fast mode, premium pricing).
        timeout_s: Per-request timeout in seconds.
        client: Optional pre-built ``anthropic.AsyncAnthropic`` (tests inject a fake).
    """

    def __init__(
        self,
        *,
        model: str,
        effort: Effort = "low",
        speed: str = "standard",
        timeout_s: float = 45.0,
        client: Any | None = None,
    ) -> None:
        if client is None:  # pragma: no cover - real network client
            import anthropic

            client = anthropic.AsyncAnthropic(timeout=timeout_s, max_retries=2)
        self._client = client
        self.model = model
        self.effort: Effort = effort
        self.speed = speed
        self.total = Usage()
        self.last: Usage | None = None

    async def _create(
        self,
        system: str,
        user: str,
        *,
        effort: Effort | None,
        max_tokens: int,
        json_schema: dict[str, Any] | None,
    ) -> str:
        output_config: dict[str, Any] = {"effort": effort or self.effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "betas": [_FALLBACK_BETA],
            "fallbacks": "default",
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        if self.speed == "fast":
            kwargs["speed"] = "fast"
        response = await self._client.beta.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            raise ClaudeRefusalError(str(getattr(response, "stop_details", None)))
        served_by = str(getattr(response, "model", self.model))
        used = estimate_usage(
            served_by if served_by in PRICES_PER_MTOK else self.model,
            getattr(response, "usage", None),
            fast=self.speed == "fast",
        )
        self.last = used
        self.total.add(used)
        _log.info("claude-response", model=served_by, stop=response.stop_reason, **used.as_dict())
        return "".join(block.text for block in response.content if block.type == "text")

    async def complete(self, system: str, user: str) -> str:
        """Plain-text completion (the :class:`LLMClient` protocol)."""
        return await self._create(system, user, effort=None, max_tokens=8000, json_schema=None)

    async def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        effort: Effort | None = None,
        max_tokens: int = 8000,
    ) -> dict[str, Any]:
        """Completion constrained to *schema*; returns the parsed JSON object."""
        text = await self._create(
            system, user, effort=effort, max_tokens=max_tokens, json_schema=schema
        )
        return cast(dict[str, Any], json.loads(text))
