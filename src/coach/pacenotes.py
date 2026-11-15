"""Pace notes: what to say where, for every corner.

Each corner has a :class:`CornerNote` - the sentence spoken at the corner's
trigger point on approach, an optional short apex note ("Kerb", "Off the
kerb"), and which of the "Brake" / "Turn in" / "Throttle" cue words to call.
A per-corner braking *mark* (metres from the start/finish line) says where
"Brake" is called; marks start conservative and are moved later as the
driver hits them.

Cue positions are computed from the reference lap: speech is started a little
before each point to absorb text-to-speech latency, and the approach sentence
is spoken a few seconds before the mark so it finishes before "Brake".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.coach.metrics import CornerComparison, CornerMetrics
from src.coach.samples import LapTrace

GEAR_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}


@dataclass
class CornerNote:
    corner: str
    say: str = ""  # spoken at the trigger point on approach
    apex_say: str = ""  # spoken just before the apex (a few words)
    call_brake: bool = True
    call_turn: bool = False
    call_throttle: bool = False
    objective: str = ""  # the head coach's current goal here (not spoken)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CornerNote:
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


@dataclass(frozen=True)
class Cue:
    corner: str
    kind: str  # approach | brake | turn | apex | throttle
    at_m: float
    text: str
    urgent: bool


@dataclass
class PassRecord:
    """One pass through a corner (kept for verdicts and the prompt history)."""

    lap: int
    row: CornerComparison
    mark_m: float | None
    said_after: str = ""


def template_note(ref: CornerMetrics) -> CornerNote:
    gear = GEAR_WORDS.get(ref.min_gear, "")
    say = f"{ref.name}, {gear}." if gear else f"{ref.name}."
    return CornerNote(corner=ref.name, say=say, call_brake=ref.brake_m is not None)


@dataclass
class PaceNotes:
    ref_metrics: list[CornerMetrics]
    ref_lap: LapTrace
    notes: dict[str, CornerNote] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)
    seed_marks: dict[str, float] = field(default_factory=dict)
    lead_s: float = 3.5
    latency_s: float = 0.35
    apex_lead_s: float = 1.2
    max_later_than_ref_m: float = 5.0

    def __post_init__(self) -> None:
        for m in self.ref_metrics:
            self.notes.setdefault(m.name, template_note(m))
            if m.brake_m is not None:
                seed = self.seed_marks.get(m.name, m.brake_m - 25.0)
                self.seed_marks.setdefault(m.name, seed)
                self.marks.setdefault(m.name, seed)

    @property
    def length_m(self) -> float:
        return self.ref_lap.length_m

    def ref(self, corner: str) -> CornerMetrics | None:
        return next((m for m in self.ref_metrics if m.name == corner), None)

    # -- marks ---------------------------------------------------------------
    def move_mark(self, corner: str, metres: float) -> float | None:
        """Move a braking mark (positive = later), within safe bounds."""
        ref = self.ref(corner)
        if corner not in self.marks or ref is None or ref.brake_m is None:
            return None
        step = max(-25.0, min(25.0, metres))
        latest = ref.brake_m + self.max_later_than_ref_m
        earliest = self.seed_marks.get(corner, ref.brake_m) - 40.0
        self.marks[corner] = max(earliest, min(latest, self.marks[corner] + step))
        return self.marks[corner]

    # -- cues ----------------------------------------------------------------
    def _v(self, s_m: float) -> float:
        return max(self.ref_lap.speed_at(s_m % self.length_m), 30.0) / 3.6

    def cues(self, *, only: set[str] | None = None) -> list[Cue]:
        """All cue points for one lap, sorted by position."""
        out: list[Cue] = []
        ordered = sorted(self.ref_metrics, key=lambda m: m.start_m)
        for i, m in enumerate(ordered):
            if only is not None and m.name not in only:
                continue
            note = self.notes.get(m.name) or template_note(m)
            prev_end = ordered[i - 1].end_m if i > 0 else ordered[-1].end_m - self.length_m
            mark = self.marks.get(m.name)
            anchor = mark if mark is not None else (m.turn_in_m or m.start_m)
            v = self._v(anchor)
            if note.say:
                at = max(anchor - self.lead_s * v, prev_end + 5.0)
                if anchor - at >= 1.5 * v:  # room to say it before the action
                    out.append(Cue(m.name, "approach", at, note.say, False))
            if mark is not None and note.call_brake:
                out.append(Cue(m.name, "brake", mark - self.latency_s * v, "Brake", True))
            if note.call_turn and m.turn_in_m is not None:
                t = m.turn_in_m - self.latency_s * self._v(m.turn_in_m)
                out.append(Cue(m.name, "turn", t, "Turn in", True))
            if note.apex_say:
                a = m.min_m - self.apex_lead_s * self._v(m.min_m)
                out.append(Cue(m.name, "apex", a, note.apex_say, True))
            if note.call_throttle and m.throttle_m is not None:
                t = m.throttle_m - self.latency_s * self._v(m.throttle_m)
                out.append(Cue(m.name, "throttle", t, "Throttle", True))
        wrapped = [Cue(c.corner, c.kind, c.at_m % self.length_m, c.text, c.urgent) for c in out]
        return sorted(wrapped, key=lambda c: c.at_m)

    # -- persistence -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "notes": {k: asdict(v) for k, v in self.notes.items()},
            "marks": {k: round(v, 1) for k, v in self.marks.items()},
            "seed_marks": {k: round(v, 1) for k, v in self.seed_marks.items()},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load_state(cls, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        return dict(json.loads(path.read_text(encoding="utf-8")))

    def apply_state(self, state: dict[str, Any]) -> None:
        for name, data in state.get("notes", {}).items():
            if self.ref(name) is not None:
                self.notes[name] = CornerNote.from_dict({**data, "corner": name})
        for name, value in state.get("seed_marks", {}).items():
            if name in self.marks:
                self.seed_marks[name] = float(value)
        for name, value in state.get("marks", {}).items():
            if name in self.marks:
                self.marks[name] = float(value)


def spoken_name(name: str) -> str:
    """Corner name as it should be read aloud ("Cotton Corners (right)" -> "Cotton right")."""
    if "(" in name and name.endswith(")"):
        base, extra = name[:-1].split("(", 1)
        return f"{base.split()[0]} {extra.strip()}"
    return name


def quick_verdict(row: CornerComparison, previous: PassRecord | None, mark: float | None) -> str:
    """An instant few-word verdict on a pass, or "" if there is nothing worth saying."""
    parts: list[str] = []
    if previous is not None:
        delta = row.driver.section_s - previous.row.driver.section_s
        if delta <= -0.05:
            parts.append(f"better, {abs(delta):.1f} up")
        elif delta >= 0.05:
            parts.append(f"slower, {delta:.1f} down")
    if mark is not None and row.driver.brake_m is not None:
        miss = row.driver.brake_m - mark
        if miss < -10:
            parts.append("braked early")
        elif miss > 10:
            parts.append("braked past the mark")
        elif parts:
            parts.append("on the mark")
    if not parts:
        return ""
    return f"{spoken_name(row.name)}: " + ", ".join(parts) + "."
