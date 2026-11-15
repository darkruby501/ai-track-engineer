"""Claude client (Anthropic API) implementing :class:`~src.ai.llm.LLMClient`.

Uses the official ``anthropic`` SDK. Credentials come from the environment
(``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile), never from config.

Every request opts into server-side refusal fallbacks (``fallbacks="default"``)
so a rare policy decline is retried on a fallback model inside the same call,
and caches the system prompt, which carries the large, stable track knowledge.
"""

from __future__ import annotations

import json
from typing import Any, Literal, cast

import structlog

_log = structlog.get_logger("ai.anthropic")

_FALLBACK_BETA = "server-side-fallback-2026-07-01"

Effort = Literal["low", "medium", "high", "xhigh", "max"]


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
        usage = getattr(response, "usage", None)
        _log.info(
            "claude-response",
            model=getattr(response, "model", self.model),
            stop=response.stop_reason,
            input_tokens=getattr(usage, "input_tokens", None),
            cache_read=getattr(usage, "cache_read_input_tokens", None),
            cache_write=getattr(usage, "cache_creation_input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )
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
