"""Claude as the driving coach: per-lap focus and spoken corner calls.

After every lap the coach sends Claude a compact, corner-by-corner comparison
against the reference lap plus a short history of the session. Claude answers
with a structured plan: one sentence to say right away, and at most a couple
of short corner calls that are spoken on the approach to those corners on the
*next* lap - so the model's latency never matters on track.

The large, stable material (track guide, driver notes, reference numbers) goes
in the system prompt, which is cached across laps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.coach.metrics import CornerComparison, CornerMetrics, comparison_table, reference_table

_COACH_RULES = """\
You are a calm, precise driving coach on the radio with an amateur track-day \
driver who is practising in Assetto Corsa. Everything you write in "say" fields \
is read aloud by text-to-speech while they drive, so it must be short, plain and \
unambiguous.

After each lap you receive a corner-by-corner comparison against a reference \
lap in the same car, setup and track. You answer with:
- lap_comment: spoken right after the lap ends. At most 18 words. Name what \
improved (if anything), then the focus for the next lap.
- calls: zero to {max_calls} corner calls, each spoken about 4 seconds before \
that corner's braking point on the NEXT lap. At most 9 words each, imperative, \
one idea each, e.g. "Off Ramp: stay left, brake at the clip".
- notes: a short private rationale (not spoken) for the session log.

How to coach:
- Pick focus corners by the biggest time loss that has a clear cause in the \
data: braking point, entry speed, minimum speed, or throttle pickup. Use the \
"time lost s" column, not just speeds.
- Change one thing per corner at a time, in small steps: brake 10 to 20 metres \
later per lap, not 80. Carry "a little more" speed, not a target 15 km/h higher.
- Keep the same focus corners for consecutive laps until the loss there is \
under about 0.15 s or the driver clearly fixed the cause; then move on. Say so \
when they fixed it.
- If a lap was messy (more than 5 s off the reference, an off or a spin, or \
the out-lap), give one simple, encouraging call and do not pile on.
- Use the official corner names. Use landmarks only when the track guide \
mentions them; otherwise give distances relative to the reference ("brake 20 \
metres later", "throttle earlier on exit"), gears, and "more or less speed".
- The reference lap is AC's AI. It is fast but not perfect: it short-shifts \
around 6000 rpm, over-slows Cotton Corners left, and exits Sweeper poorly into \
the Esses. Where the driver is already faster than the reference, do not coach \
them toward it; say they are good there.
- Never invent numbers. Only use what is in the data and the reference material.
- Units: km/h and metres. Vary your wording from lap to lap.
"""


@dataclass
class CornerCall:
    corner: str
    say: str
    why: str = ""


@dataclass
class CoachPlan:
    lap_comment: str
    calls: list[CornerCall] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any], *, valid_corners: set[str]) -> CoachPlan:
        calls = [
            CornerCall(corner=c["corner"], say=c["say"].strip(), why=c.get("why", ""))
            for c in data.get("calls", [])
            if c.get("corner") in valid_corners and c.get("say", "").strip()
        ]
        return cls(
            lap_comment=str(data.get("lap_comment", "")).strip(),
            calls=calls,
            notes=str(data.get("notes", "")),
        )


@dataclass
class LapRecord:
    """What the coach remembers about one lap."""

    lap_no: int
    lap_time_s: float
    worst: list[tuple[str, float]]
    plan: CoachPlan | None = None

    def summary(self) -> str:
        losses = ", ".join(f"{name} {lost:+.2f}s" for name, lost in self.worst)
        calls = "; ".join(f"{c.corner}: {c.say}" for c in self.plan.calls) if self.plan else ""
        return (
            f"lap {self.lap_no}: {self.lap_time_s:.3f}s | biggest losses: {losses} | "
            f"your calls for the following lap: {calls or 'none'}"
        )


class JSONLLM(Protocol):
    async def complete_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]: ...


def plan_schema(corner_names: list[str], max_calls: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "lap_comment": {"type": "string"},
            "calls": {
                "type": "array",
                "maxItems": max_calls,
                "items": {
                    "type": "object",
                    "properties": {
                        "corner": {"type": "string", "enum": corner_names},
                        "say": {"type": "string"},
                        "why": {"type": "string"},
                    },
                    "required": ["corner", "say", "why"],
                    "additionalProperties": False,
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["lap_comment", "calls", "notes"],
        "additionalProperties": False,
    }


def build_system_prompt(
    *,
    track_name: str,
    car: str,
    reference_time_s: float,
    reference_metrics: list[CornerMetrics],
    context_files: list[Path],
    max_calls: int,
) -> str:
    parts = [_COACH_RULES.format(max_calls=max_calls)]
    parts.append(
        f"# Session\nTrack: {track_name}. Car: {car}. Reference lap: "
        f"{reference_time_s:.3f}s (AC AI, same car and setup).\n"
        "Distances are metres from the start/finish line along the racing line."
    )
    parts.append("# Reference lap, corner by corner\n" + reference_table(reference_metrics))
    for path in context_files:
        if path.is_file():
            parts.append(f"# Reference material: {path.name}\n" + path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def build_lap_message(
    *,
    lap_no: int,
    lap_time_s: float,
    reference_time_s: float,
    rows: list[CornerComparison],
    history: list[LapRecord],
    flags: list[str],
) -> str:
    lines = [
        f"Lap {lap_no} just finished: {lap_time_s:.3f}s "
        f"({lap_time_s - reference_time_s:+.3f}s vs reference)."
    ]
    if flags:
        lines.append("Flags: " + "; ".join(flags))
    if history:
        lines.append("Earlier laps this session (oldest first):")
        lines.extend("- " + h.summary() for h in history)
    lines.append("This lap vs the reference:")
    lines.append(comparison_table(rows))
    return "\n".join(lines)


class CoachPlanner:
    """Turns a lap comparison into a :class:`CoachPlan` using Claude."""

    def __init__(
        self,
        llm: JSONLLM,
        *,
        system_prompt: str,
        corner_names: list[str],
        reference_time_s: float,
        max_calls: int = 2,
        history_laps: int = 6,
    ) -> None:
        self._llm = llm
        self._system = system_prompt
        self._corners = corner_names
        self._schema = plan_schema(corner_names, max_calls)
        self._ref_time = reference_time_s
        self._history_laps = history_laps
        self.history: list[LapRecord] = []

    async def plan(
        self,
        *,
        lap_no: int,
        lap_time_s: float,
        rows: list[CornerComparison],
        flags: list[str] | None = None,
    ) -> CoachPlan:
        message = build_lap_message(
            lap_no=lap_no,
            lap_time_s=lap_time_s,
            reference_time_s=self._ref_time,
            rows=rows,
            history=self.history[-self._history_laps :],
            flags=flags or [],
        )
        data = await self._llm.complete_json(self._system, message, self._schema)
        plan = CoachPlan.from_json(data, valid_corners=set(self._corners))
        worst = sorted(((r.name, r.time_lost_s) for r in rows), key=lambda x: -x[1])[:3]
        self.history.append(LapRecord(lap_no, lap_time_s, worst, plan))
        return plan
