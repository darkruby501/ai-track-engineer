"""The live coach: lap bookkeeping, timed corner calls, and spoken feedback.

Feed it one :class:`Sample` at a time (live or from a recording). When a lap
ends it reads out the lap time immediately, then asks the planner (Claude) in
the background what to work on. Corner calls from the latest plan are spoken
once per lap, a few seconds before the reference braking point for that
corner; the lap comment waits for the next stretch of straight.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.coach.metrics import CornerComparison, CornerMetrics, compare, measure_lap
from src.coach.planner import CoachPlan, CornerCall
from src.coach.reference import Reference
from src.coach.samples import LapTrace, Sample
from src.coach.speaker import LogSpeaker
from src.knowledge.models import Corner

_log = structlog.get_logger("coach.live")

PlanFn = Callable[[int, float, list[CornerComparison], list[str]], Awaitable[CoachPlan]]


def say_lap_time(lap_s: float, ref_s: float) -> str:
    tenths_total = int(round(lap_s * 10))
    minutes, rest = divmod(tenths_total, 600)
    whole, tenth = divmod(rest, 10)
    delta = lap_s - ref_s
    sign = "plus" if delta >= 0 else "minus"
    return f"{int(minutes)} {whole:02d} point {tenth}, {sign} {abs(delta):.1f}"


def heuristic_plan(rows: list[CornerComparison]) -> CoachPlan:
    """A no-LLM plan: the single corner with the biggest loss and its likely cause."""
    if not rows:
        return CoachPlan(lap_comment="")
    worst = max(rows, key=lambda r: r.time_lost_s)
    if worst.time_lost_s < 0.15:
        return CoachPlan(lap_comment="Close to the reference everywhere. Keep it up.")
    if worst.brake_diff_m is not None and worst.brake_diff_m < -15:
        say = f"{worst.name}: brake a little later"
    elif worst.min_speed_diff < -3:
        say = f"{worst.name}: carry more speed to the apex"
    elif worst.throttle_diff_m is not None and worst.throttle_diff_m > 15:
        say = f"{worst.name}: back to throttle earlier"
    else:
        say = f"{worst.name}: smooth and tidy"
    return CoachPlan(
        lap_comment=f"Biggest loss {worst.name}, {worst.time_lost_s:.1f} seconds.",
        calls=[CornerCall(corner=worst.name, say=say, why="heuristic")],
    )


@dataclass
class _Trigger:
    corner: str
    at_m: float


@dataclass
class LiveCoach:
    corners: tuple[Corner, ...]
    reference: Reference
    speaker: LogSpeaker
    plan_fn: PlanFn | None = None
    call_lead_s: float = 4.5
    log_path: Path | None = None
    plan_timeout_s: float = 40.0

    _samples: list[Sample] = field(default_factory=list)
    _ref_metrics: list[CornerMetrics] = field(default_factory=list)
    _triggers: dict[str, _Trigger] = field(default_factory=dict)
    _active: dict[str, CornerCall] = field(default_factory=dict)
    _fired: set[str] = field(default_factory=set)
    _pending_comment: str | None = None
    _tasks: set[asyncio.Task[None]] = field(default_factory=set)
    _laps_seen: int = 0
    plans: list[CoachPlan] = field(default_factory=list)
    usage: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._ref_metrics = measure_lap(self.reference.lap, self.corners)
        self._triggers = {m.name: self._trigger_for(m) for m in self._ref_metrics}

    @property
    def length_m(self) -> float:
        return self.reference.lap.length_m

    def _trigger_for(self, m: CornerMetrics) -> _Trigger:
        lap = self.reference.lap
        target = m.brake_m if m.brake_m is not None else (m.lift_m or m.start_m)
        speed_ms = max(lap.speed_at(target), 30.0) / 3.6
        at = target - self.call_lead_s * speed_ms
        previous_ends = [o.end_m for o in self._ref_metrics if o.end_m <= m.start_m]
        if previous_ends:
            at = max(at, max(previous_ends) + 5.0)
        return _Trigger(m.name, at % self.length_m)

    # -- public API ---------------------------------------------------------
    async def feed(self, sample: Sample) -> None:
        prev = self._samples[-1] if self._samples else None
        if prev is not None and sample.lap != prev.lap:
            finished = self._samples
            self._samples = [sample]
            self._fired.clear()
            self._on_lap_complete(finished, sample.last_ms)
        else:
            self._samples.append(sample)
        if prev is not None and sample.lap == prev.lap:
            self._check_calls(prev, sample)
            self._maybe_speak_comment(sample)

    async def drain(self) -> None:
        """Wait for in-flight planner calls (tests, replay, shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # -- internals ----------------------------------------------------------
    def _check_calls(self, prev: Sample, cur: Sample) -> None:
        if cur.speed_kmh < 30 or cur.in_pit:
            return
        a, b = prev.pos * self.length_m, cur.pos * self.length_m
        if b < a:  # spline wrapped without a lap change; ignore this step
            return
        for name, call in list(self._active.items()):
            trig = self._triggers.get(name)
            if trig is None or name in self._fired:
                continue
            if a < trig.at_m <= b:
                self._fired.add(name)
                self.speaker.say(call.say)
                self._event("call", corner=name, say=call.say, at_m=round(b))

    def _in_quiet_zone(self, sample: Sample) -> bool:
        s = sample.pos * self.length_m
        speed_ms = max(sample.speed_kmh, 40.0) / 3.6
        for trig in self._triggers.values():
            ahead = (trig.at_m - s) % self.length_m
            if ahead < 3.5 * speed_ms:  # a call point is less than ~3.5 s away
                return False
        return not any(m.start_m - 40 <= s <= m.end_m for m in self._ref_metrics)

    def _maybe_speak_comment(self, sample: Sample) -> None:
        if self._pending_comment and self._in_quiet_zone(sample):
            self.speaker.say(self._pending_comment)
            self._event("comment", say=self._pending_comment)
            self._pending_comment = None

    def _on_lap_complete(self, samples: list[Sample], lap_ms: int) -> None:
        self._laps_seen += 1
        lap_no = samples[-1].lap + 1
        if lap_ms <= 0 or len(samples) < 50:
            return
        lap_s = lap_ms / 1000.0
        ref_s = self.reference.lap_time_s
        flags: list[str] = []
        if any(s.in_pit for s in samples):
            flags.append("lap included the pit lane (out-lap or in-lap)")
        if self._laps_seen == 1:
            flags.append("first lap seen this session (may be partial or an out-lap)")
        if lap_s > ref_s + 5:
            flags.append("slow or messy lap (more than 5 s off the reference)")
        self.speaker.say(say_lap_time(lap_s, ref_s), urgent=True)
        try:
            trace = LapTrace(samples, length_m=self.length_m, lap_time_s=lap_s)
            rows = compare(measure_lap(trace, self.corners), self._ref_metrics)
        except ValueError:
            _log.warning("lap-unusable", lap=lap_no)
            return
        self._event(
            "lap",
            lap=lap_no,
            time_s=lap_s,
            flags=flags,
            corners=[
                {
                    "corner": r.name,
                    "lost_s": round(r.time_lost_s, 3),
                    "brake_diff_m": None if r.brake_diff_m is None else round(r.brake_diff_m),
                    "min_diff_kmh": round(r.min_speed_diff, 1),
                    "throttle_diff_m": (
                        None if r.throttle_diff_m is None else round(r.throttle_diff_m)
                    ),
                }
                for r in rows
            ],
        )
        if any("pit lane" in f for f in flags):
            return
        task = asyncio.get_running_loop().create_task(self._plan(lap_no, lap_s, rows, flags))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _plan(
        self, lap_no: int, lap_s: float, rows: list[CornerComparison], flags: list[str]
    ) -> None:
        started = time.monotonic()
        plan: CoachPlan
        if self.plan_fn is None:
            plan = heuristic_plan(rows)
        else:
            try:
                plan = await asyncio.wait_for(
                    self.plan_fn(lap_no, lap_s, rows, flags), self.plan_timeout_s
                )
            except Exception as exc:  # noqa: BLE001 - never let coaching kill the session
                _log.warning("planner-failed", error=repr(exc))
                plan = heuristic_plan(rows)
        self.plans.append(plan)
        for key, value in (plan.usage or {}).items():
            self.usage[key] = self.usage.get(key, 0) + value
        self._active = {c.corner: c for c in plan.calls}
        self._pending_comment = plan.lap_comment or None
        self._event(
            "plan",
            lap=lap_no,
            latency_s=round(time.monotonic() - started, 2),
            comment=plan.lap_comment,
            calls=[{"corner": c.corner, "say": c.say, "why": c.why} for c in plan.calls],
            notes=plan.notes,
            usage=plan.usage,
        )

    def usage_summary(self) -> str:
        """One line: Claude calls, tokens and estimated cost so far."""
        u = self.usage
        calls = int(u.get("calls", 0))
        if not calls:
            return "Claude calls: 0"
        cost = u.get("cost_usd", 0.0)
        return (
            f"Claude calls: {calls} | input {int(u.get('input_tokens', 0))} + cache read "
            f"{int(u.get('cache_read_tokens', 0))} + cache write "
            f"{int(u.get('cache_write_tokens', 0))} | output (incl. thinking) "
            f"{int(u.get('output_tokens', 0))} | est. ${cost:.4f} total, "
            f"${cost / calls:.4f} per lap"
        )

    def finish(self) -> None:
        """Log the session's Claude usage (call at the end of a session)."""
        if self.usage:
            self._event("session-usage", **{k: round(v, 5) for k, v in self.usage.items()})

    def _event(self, kind: str, **data: Any) -> None:
        _log.info(kind, **data)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": round(time.time(), 2), "event": kind, **data}) + "\n")
