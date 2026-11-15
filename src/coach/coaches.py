"""The coach's three Claude roles.

* :class:`CornerAdvisor` - a fast model called as the driver exits each corner.
  It returns a few words to say now ("better, on the mark") and that corner's
  notes for the next lap (approach sentence, apex note, which cue words).
* :class:`HeadCoach` - a frontier model called once per lap. It sets the
  session's focus: a short spoken lap comment and an objective for up to
  three corners, which the corner advisor then works towards.
* :class:`NoteWriter` - the frontier model once, before driving: initial
  notes for every corner, built from the track brief.

All three share one context block (track brief, driver brief, reference and
target numbers) which is sent as the cached system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.coach.metrics import CornerComparison, CornerMetrics, comparison_table, reference_table
from src.coach.pacenotes import CornerNote, PassRecord

SAFETY = """\
Safety: the driver will drive this track for real next weekend in their own \
car. Keep all four wheels on the pavement; never suggest dropping wheels in the \
dirt, jumping kerbs the brief calls risky, or braking past the braking mark \
(the marks already include a safety margin). Where going off would be costly \
(Bus Stop exit, Phil Hill crest, Sweeper exit and the Esses) favour a tidy, \
repeatable line over the last tenth."""


class JSONLLM(Protocol):
    async def complete_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]: ...


@dataclass
class CoachContext:
    """Shared, stable reference material for the system prompts."""

    track_name: str
    car: str
    reference_time_s: float
    reference_metrics: list[CornerMetrics]
    marks: dict[str, float]
    target_min_kmh: dict[str, float]
    context_files: list[Path] = field(default_factory=list)

    def text(self) -> str:
        parts = [
            f"# Session\nTrack: {self.track_name}. Car: {self.car}. Reference lap: "
            f"{self.reference_time_s:.3f}s (AC AI, same car and setup). Distances are metres "
            "from the start/finish line along the racing line. 'Inside'/'outside' gaps are "
            "metres between the car and that track edge.",
            "# Reference lap (AC AI), corner by corner\n" + reference_table(self.reference_metrics),
            "# Target (conservative: AI line kept inside safety margins, ~5% slower corners, "
            "gentler braking)\ncorner | braking mark @m | target min km/h\n"
            + "\n".join(
                f"{m.name} | {self.marks.get(m.name, float('nan')):.0f} | "
                f"{self.target_min_kmh.get(m.name, float('nan')):.0f}"
                for m in self.reference_metrics
            ),
        ]
        for path in self.context_files:
            if path.is_file():
                parts.append(f"# {path.name}\n" + path.read_text(encoding="utf-8"))
        return "\n\n".join(parts)


# -- the fast per-corner advisor ------------------------------------------------
ADVISOR_RULES = """\
You are the co-driver voice of a live driving coach in Assetto Corsa. Each time \
the driver exits a corner you get that pass's numbers. Everything in "now", \
"say" and "apex_say" is read aloud by text-to-speech while they drive.

Return:
- now: at most 9 words, spoken at the next quiet moment (possibly a few \
seconds later, so start with the corner's short spoken name). How this pass \
went compared with their previous pass, and the one thing that made the \
difference. E.g. "Phil Hill: better, braked on the mark." / "Sunrise: two \
tenths slower, turned in early." No numbers except tenths of a second. \
Return "" when nothing changed that is worth the driver's attention.
- say: at most 10 words, spoken a few seconds before this corner's braking mark \
next lap. One concrete instruction. Use landmarks, kerbs, bumps and line \
placement from the track brief where useful. Keep the previous wording when it \
is still right - repetition helps a driver learn.
- apex_say: at most 4 words spoken just before the apex, or "" (use for a kerb, \
bump or line reminder that matters, e.g. "Kerb", "Wait for it", "Off the kerb").
- call_turn / call_throttle: true to also call "Turn in" / "Throttle" at the \
reference points - only when turn-in or throttle timing is this driver's issue here.

Follow the head coach's objective for this corner when there is one. Use only \
the numbers given. Be calm and plain.
"""


def advisor_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "now": {"type": "string"},
            "say": {"type": "string"},
            "apex_say": {"type": "string"},
            "call_turn": {"type": "boolean"},
            "call_throttle": {"type": "boolean"},
        },
        "required": ["now", "say", "apex_say", "call_turn", "call_throttle"],
        "additionalProperties": False,
    }


def _m(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}"


def _g(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def pass_message(
    row: CornerComparison,
    *,
    lap: int,
    mark: float | None,
    target_min: float | None,
    history: list[PassRecord],
    note: CornerNote,
) -> str:
    d, r = row.driver, row.ref
    lines = [f"Corner: {d.name} ({d.direction or '?'}). Lap {lap}."]
    times = f"Section time {d.section_s:.2f}s (AI {r.section_s:.2f}s"
    if history:
        times += f"; your previous {history[-1].row.driver.section_s:.2f}s"
        times += f"; your best {min(h.row.driver.section_s for h in history):.2f}s"
    lines.append(times + ").")
    if d.brake_m is not None or mark is not None:
        miss = "" if d.brake_m is None or mark is None else f" -> {d.brake_m - mark:+.0f} m vs mark"
        lines.append(
            f"Braking started @{_m(d.brake_m)} m; mark @{_m(mark)} m (AI {_m(r.brake_m)}){miss}."
        )
    lines.append(
        f"Turn-in @{_m(d.turn_in_m)} (AI {_m(r.turn_in_m)}). Entry {d.entry_kmh:.0f} km/h "
        f"(AI {r.entry_kmh:.0f}). Minimum {d.min_kmh:.0f} km/h (target {_m(target_min)}, "
        f"AI {r.min_kmh:.0f}) in gear {d.min_gear} (AI {r.min_gear}). Throttle back "
        f"@{_m(d.throttle_m)} (AI {_m(r.throttle_m)})."
    )
    lines.append(
        f"Line: apex {_g(d.apex_inside_m)} m from inside (AI {_g(r.apex_inside_m)}), exit "
        f"{_g(d.exit_outside_m)} m from outside (AI {_g(r.exit_outside_m)})."
    )
    for h in history[-3:]:
        hd = h.row.driver
        lines.append(
            f"Earlier, lap {h.lap}: {hd.section_s:.2f}s, braked @{_m(hd.brake_m)}, min "
            f"{hd.min_kmh:.0f}; you then said: {h.said_after or '-'}"
        )
    cues = ", ".join(
        f"{k} {'on' if v else 'off'}"
        for k, v in (
            ("brake", note.call_brake),
            ("turn", note.call_turn),
            ("throttle", note.call_throttle),
        )
    )
    lines.append(f'Current notes: approach "{note.say}", apex "{note.apex_say}", cues {cues}.')
    lines.append(f"Head coach objective here: {note.objective or 'none set'}.")
    return "\n".join(lines)


@dataclass
class Advice:
    now: str
    note: CornerNote


class CornerAdvisor:
    def __init__(self, llm: JSONLLM, context: CoachContext) -> None:
        self._llm = llm
        self._system = ADVISOR_RULES + "\n" + SAFETY + "\n\n" + context.text()
        self._schema = advisor_schema()

    async def advise(self, message: str, note: CornerNote) -> Advice:
        data = await self._llm.complete_json(self._system, message, self._schema)
        updated = CornerNote(
            corner=note.corner,
            say=str(data.get("say", note.say)).strip() or note.say,
            apex_say=str(data.get("apex_say", "")).strip(),
            call_brake=note.call_brake,
            call_turn=bool(data.get("call_turn", note.call_turn)),
            call_throttle=bool(data.get("call_throttle", note.call_throttle)),
            objective=note.objective,
        )
        return Advice(now=str(data.get("now", "")).strip(), note=updated)


# -- the per-lap head coach -----------------------------------------------------
HEAD_RULES = """\
You are the head coach of a live driving coach in Assetto Corsa. After each lap \
you get a corner-by-corner comparison with the reference lap and the session \
history. A fast co-driver model handles each corner's spoken notes; your job is \
the plan:
- lap_comment: at most 18 words, read aloud at the next quiet moment. What \
improved, and what to work on now.
- focus: up to three corners to work on next, each with an objective of at most \
20 words for the co-driver (not spoken), e.g. "Brake at the mark, release \
earlier, carry 3-4 km/h more to the apex; keep left on entry."
- notes: a short private rationale for the session log.
Pick corners by the biggest repeatable loss with a clear cause in the data. \
Keep a focus until it is fixed (loss under about 0.15 s) or three laps pass. \
The reference AI is fast but imperfect: it short-shifts around 6000 rpm, \
over-slows Cotton Corners left and exits Sweeper poorly. Use only the data given.
"""


def head_schema(corner_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "lap_comment": {"type": "string"},
            "focus": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "corner": {"type": "string", "enum": corner_names},
                        "objective": {"type": "string"},
                    },
                    "required": ["corner", "objective"],
                    "additionalProperties": False,
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["lap_comment", "focus", "notes"],
        "additionalProperties": False,
    }


@dataclass
class LapPlan:
    lap_comment: str
    focus: dict[str, str]
    notes: str = ""
    usage: dict[str, Any] | None = None


class HeadCoach:
    def __init__(self, llm: JSONLLM, context: CoachContext, corner_names: list[str]) -> None:
        self._llm = llm
        self._system = HEAD_RULES + "\n" + SAFETY + "\n\n" + context.text()
        self._schema = head_schema(corner_names)
        self._names = set(corner_names)
        self._ref_time = context.reference_time_s
        self.history: list[str] = []

    async def plan(
        self, *, lap: int, lap_time_s: float, rows: list[CornerComparison], flags: list[str]
    ) -> LapPlan:
        message = [
            f"Lap {lap} just finished: {lap_time_s:.3f}s "
            f"({lap_time_s - self._ref_time:+.3f}s vs the reference)."
        ]
        if flags:
            message.append("Flags: " + "; ".join(flags))
        if self.history:
            message.append("Earlier laps (oldest first):\n" + "\n".join(self.history[-6:]))
        message.append("This lap vs the reference:\n" + comparison_table(rows))
        data = await self._llm.complete_json(self._system, "\n".join(message), self._schema)
        focus = {
            f["corner"]: str(f.get("objective", "")).strip()
            for f in data.get("focus", [])
            if f.get("corner") in self._names
        }
        focus = dict(list(focus.items())[:3])
        worst = sorted(rows, key=lambda r: -r.time_lost_s)[:3]
        self.history.append(
            f"lap {lap}: {lap_time_s:.3f}s; biggest losses "
            + ", ".join(f"{r.name} {r.time_lost_s:+.2f}s" for r in worst)
            + f"; focus set: {', '.join(focus) or 'none'}"
        )
        last = getattr(self._llm, "last", None)
        return LapPlan(
            lap_comment=str(data.get("lap_comment", "")).strip(),
            focus=focus,
            notes=str(data.get("notes", "")),
            usage=last.as_dict() if last is not None else None,
        )


# -- initial notes ---------------------------------------------------------------
WRITER_RULES = """\
You write the initial pace notes for a live driving coach in Assetto Corsa. For \
every corner give:
- say: at most 10 words, read aloud a few seconds before the corner's braking \
mark (or turn-in, if it is not braked for). The most useful thing to know \
going in: braking reference, line placement, a landmark, kerb or bump from the \
track brief, or the gear.
- apex_say: at most 4 words read just before the apex, or "" if nothing matters there.
- call_turn / call_throttle: whether to call "Turn in" / "Throttle" at the \
reference points (use sparingly; start with true only where timing is critical).
Write for a driver learning the track. Plain, calm words.
"""


def writer_schema(corner_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "corner": {"type": "string", "enum": corner_names},
                        "say": {"type": "string"},
                        "apex_say": {"type": "string"},
                        "call_turn": {"type": "boolean"},
                        "call_throttle": {"type": "boolean"},
                    },
                    "required": ["corner", "say", "apex_say", "call_turn", "call_throttle"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["notes"],
        "additionalProperties": False,
    }


async def write_initial_notes(
    llm: JSONLLM, context: CoachContext, existing: dict[str, CornerNote]
) -> dict[str, CornerNote]:
    names = [m.name for m in context.reference_metrics]
    system = WRITER_RULES + "\n" + SAFETY + "\n\n" + context.text()
    data = await llm.complete_json(
        system, "Write the notes for every corner, in lap order.", writer_schema(names)
    )
    out = dict(existing)
    for item in data.get("notes", []):
        name = item.get("corner")
        if name not in existing:
            continue
        base = existing[name]
        out[name] = CornerNote(
            corner=name,
            say=str(item.get("say", "")).strip() or base.say,
            apex_say=str(item.get("apex_say", "")).strip(),
            call_brake=base.call_brake,
            call_turn=bool(item.get("call_turn", False)),
            call_throttle=bool(item.get("call_throttle", False)),
            objective=base.objective,
        )
    return out
