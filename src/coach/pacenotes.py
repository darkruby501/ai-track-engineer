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
Cues are then scheduled so no phrase is cut off by the next one: notes move
earlier where there is room, and where there is none the less important cue
is dropped ("Brake" and "Turn in" are never moved or dropped for a note).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.coach.metrics import CornerComparison, CornerMetrics
from src.coach.samples import LapTrace

GEAR_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}

_WORD = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")
# Cue priority when two cannot both be said: a note never displaces "Brake" or
# "Turn in"; "Throttle" gives way to a note (notes often cover the throttle).
_PRIORITY = {"brake": 4, "turn": 4, "approach": 3, "apex": 2, "throttle": 1}
# How much earlier than planned a note may start, in seconds.
_MAX_ADVANCE_S = {"approach": 3.0, "apex": 2.0}


def estimate_speech_s(text: str) -> float:
    """Seconds to say *text*: fitted to the ElevenLabs voice at speed 1.1, plus 0.2 s."""
    return 0.65 + 0.23 * len(_WORD.findall(text)) + 0.25 * text.count(",")


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
class _Slot:
    """A cue being scheduled: when it starts (seconds into the lap) and how long it lasts."""

    cue: Cue
    t: float
    planned: float
    dur: float


@dataclass
class PassRecord:
    """One pass through a corner (kept for verdicts and the prompt history)."""

    lap: int
    row: CornerComparison
    mark_m: float | None
    said_after: str = ""
    incident: str = ""  # "spin or near-stop" / "off track" - not a fair comparison


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
    gap_s: float = 0.15  # silence between two cues
    cue_words: bool = True  # "Brake" / "Turn in" / "Throttle" (off when the line shows them)
    speech_s: Callable[[str], float] = field(default=estimate_speech_s, repr=False)

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
    def approach_text(self, corner: str) -> str:
        """The approach note as spoken: always says which corner it is about."""
        note = self.notes.get(corner)
        if note is None or not note.say:
            return ""
        name = spoken_name(corner)
        if note.say.lower().startswith(name.split()[0].lower()):
            return note.say
        return f"{name}. {note.say}"

    def spoken_texts(self) -> list[str]:
        """Everything the cues can say (to prepare the audio ahead of time)."""
        texts = [self.approach_text(c) for c in self.notes]
        texts += [n.apex_say for n in self.notes.values()]
        return [t for t in texts if t]

    def _v(self, s_m: float) -> float:
        return max(self.ref_lap.speed_at(s_m % self.length_m), 30.0) / 3.6

    def cues(self, *, only: set[str] | None = None) -> list[Cue]:
        """All cue points for one lap, sorted by position."""
        out: list[Cue] = []
        ordered = sorted(self.ref_metrics, key=lambda m: m.start_m)
        for m in ordered:
            if only is not None and m.name not in only:
                continue
            note = self.notes.get(m.name) or template_note(m)
            mark = self.marks.get(m.name)
            anchor = mark if mark is not None else (m.turn_in_m or m.start_m)
            v = self._v(anchor)
            if note.say:
                at = anchor - self.lead_s * v
                out.append(Cue(m.name, "approach", at, self.approach_text(m.name), False))
            if mark is not None and note.call_brake and self.cue_words:
                out.append(Cue(m.name, "brake", mark - self.latency_s * v, "Brake", True))
            if note.call_turn and m.turn_in_m is not None and self.cue_words:
                t = m.turn_in_m - self.latency_s * self._v(m.turn_in_m)
                out.append(Cue(m.name, "turn", t, "Turn in", True))
            if note.apex_say:
                a = m.min_m - self.apex_lead_s * self._v(m.min_m)
                out.append(Cue(m.name, "apex", a, note.apex_say, True))
            if note.call_throttle and m.throttle_m is not None and self.cue_words:
                t = m.throttle_m - self.latency_s * self._v(m.throttle_m)
                out.append(Cue(m.name, "throttle", t, "Throttle", True))
        wrapped = [Cue(c.corner, c.kind, c.at_m % self.length_m, c.text, c.urgent) for c in out]
        return self._schedule(wrapped)

    # -- scheduling ------------------------------------------------------------
    def _time(self, s_m: float) -> float:
        return self.ref_lap.time_at(s_m % self.length_m)

    def _pos(self, t: float) -> float:
        lap = self.ref_lap
        return float(np.interp(t % self.lap_s, lap.t, lap.s))

    @property
    def lap_s(self) -> float:
        return float(self.ref_lap.t[-1])

    def _schedule(self, cues: list[Cue]) -> list[Cue]:
        """Move or drop cues so each finishes before the next one starts."""
        lap = self.lap_s
        items = [
            _Slot(c, self._time(c.at_m), self._time(c.at_m), self.speech_s(c.text)) for c in cues
        ]
        gap = self.gap_s
        for _ in range(4 * len(items) + 4):
            items.sort(key=lambda it: it.t)
            n = len(items)
            if n < 2:
                break
            for i in range(n):
                a, b = items[i], items[(i + 1) % n]
                tb = b.t + (lap if i == n - 1 else 0.0)
                overlap = a.t + a.dur + gap - tb
                if overlap <= 1e-9:
                    continue
                if self._move_earlier(items, i, tb - a.dur - gap):
                    break
                if self._delay(items, (i + 1) % n, overlap):
                    break
                pa, pb = _PRIORITY[a.cue.kind], _PRIORITY[b.cue.kind]
                items.remove(b if pb <= pa else a)
                break
            else:
                break  # no overlaps left
        out = [
            Cue(
                it.cue.corner,
                it.cue.kind,
                self._pos(it.t),
                it.cue.text,
                it.cue.urgent,
            )
            for it in items
        ]
        return sorted(out, key=lambda c: c.at_m)

    def _move_earlier(self, items: list[_Slot], i: int, new_t: float) -> bool:
        it = items[i]
        limit = _MAX_ADVANCE_S.get(it.cue.kind)
        if limit is None:
            return False
        n = len(items)
        prev = items[i - 1]
        prev_end = prev.t - (self.lap_s if i == 0 else 0.0) + prev.dur + self.gap_s
        if new_t < it.planned - limit or new_t < prev_end:
            return False
        it.t = new_t % self.lap_s if n else new_t
        return True

    def _delay(self, items: list[_Slot], j: int, by: float) -> bool:
        """Start a note slightly late (at most 0.4 s) if the cue after it allows.

        A less important cue in the way (e.g. "Throttle") is dropped for it.
        """
        it = items[j]
        if it.cue.kind not in _MAX_ADVANCE_S or by > 0.4:
            return False
        n = len(items)
        nxt = items[(j + 1) % n]
        t_next = nxt.t + (self.lap_s if j == n - 1 else 0.0)
        if it.t + by + it.dur + self.gap_s > t_next:
            if n > 2 and _PRIORITY[nxt.cue.kind] < _PRIORITY[it.cue.kind]:
                items.remove(nxt)
                return True  # try again without it
            return False
        it.t = (it.t + by) % self.lap_s
        return True

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


_NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def time_words(seconds: float) -> str:
    """A time gap as a person would say it, never more precise than useful.

    "a tenth", "three tenths", "half a second", "a second", "a second and a
    half", "about four seconds", "about 25 seconds".
    """
    tenths = round(abs(seconds) * 10)
    if tenths <= 1:
        return "a tenth"
    if tenths == 5:
        return "half a second"
    if tenths < 10:
        return f"{_NUMBER_WORDS[tenths]} tenths"
    if tenths < 15:
        return "a second"
    if tenths < 20:
        return "a second and a half"
    whole = round(abs(seconds))
    return f"about {_NUMBER_WORDS[whole] if whole < 10 else whole} seconds"


def quick_verdict(row: CornerComparison, previous: PassRecord | None, mark: float | None) -> str:
    """An instant verdict on a pass in a plain sentence, or "" if there is nothing to say."""
    name = spoken_name(row.name)
    change = ""
    if previous is not None:
        delta = row.driver.section_s - previous.row.driver.section_s
        if delta <= -0.05:
            change = f"{name} was {time_words(delta)} quicker than last time"
        elif delta >= 0.05:
            change = f"{name} was {time_words(delta)} slower than last time"
    braking = ""
    if mark is not None and row.driver.brake_m is not None:
        miss = row.driver.brake_m - mark
        if miss < -10:
            braking = "you braked early"
        elif miss > 10:
            braking = "you braked late"
        elif change:
            braking = "you braked right on the mark"
    if change and braking:
        return f"{change}, and {braking}."
    if change:
        return f"{change}."
    if braking:
        return f"In {name}, {braking}."
    return ""
