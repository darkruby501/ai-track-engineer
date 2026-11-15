"""The live coach: pace-note cues, per-corner verdicts, per-lap focus.

Feed it one :class:`Sample` at a time (live or from a recording).

* Position cues from :class:`PaceNotes` are spoken as the car crosses them:
  the approach sentence a few seconds before a corner, then "Brake" at the
  braking mark, and optional "Turn in" / apex / "Throttle" calls.
* As the car exits a corner, that pass is measured against the reference and
  the corner advisor (a fast model) returns a short verdict - spoken at the
  next quiet moment - and that corner's notes for the next lap.
* As a lap ends, the lap time is queued for the next quiet moment and the
  head coach (a frontier model) sets the focus objectives for the next lap.

Nothing waits for a model on track: position cues always come from notes that
were decided at least a lap earlier.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.coach.coaches import CornerAdvisor, HeadCoach, LapPlan, pass_message
from src.coach.metrics import CornerComparison, CornerMetrics, compare, measure_lap
from src.coach.pacenotes import Cue, PaceNotes, PassRecord, quick_verdict
from src.coach.reference import Reference
from src.coach.samples import LapTrace, Sample
from src.coach.speaker import LogSpeaker
from src.knowledge.models import Corner
from src.knowledge.track_edges import TrackEdges

_log = structlog.get_logger("coach.live")

WORDS_PER_S = 2.8


def say_lap_time(lap_s: float, ref_s: float) -> str:
    tenths_total = int(round(lap_s * 10))
    minutes, rest = divmod(tenths_total, 600)
    whole, tenth = divmod(rest, 10)
    delta = lap_s - ref_s
    sign = "plus" if delta >= 0 else "minus"
    return f"{int(minutes)} {whole:02d} point {tenth}, {sign} {abs(delta):.1f}"


def speech_seconds(text: str) -> float:
    return len(text.split()) / WORDS_PER_S + 0.4


@dataclass
class _Queued:
    text: str
    created_t: float
    max_age_s: float
    kind: str


@dataclass
class LiveCoach:
    corners: tuple[Corner, ...]
    reference: Reference
    speaker: LogSpeaker
    pace: PaceNotes
    edges: TrackEdges | None = None
    advisor: CornerAdvisor | None = None
    head: HeadCoach | None = None
    target_min_kmh: dict[str, float] = field(default_factory=dict)
    log_path: Path | None = None
    state_path: Path | None = None
    advisor_timeout_s: float = 10.0
    head_timeout_s: float = 40.0
    clients: list[Any] = field(default_factory=list)  # for usage reporting
    context: Any = None  # CoachContext, used to write initial notes
    head_client: Any = None
    notes_from_claude: bool = False

    _samples: list[Sample] = field(default_factory=list)
    _ref_metrics: list[CornerMetrics] = field(default_factory=list)
    _ref_by_name: dict[str, CornerMetrics] = field(default_factory=dict)
    _cues: list[Cue] = field(default_factory=list)
    _cues_dirty: bool = True
    _fired: set[tuple[str, str]] = field(default_factory=set)
    _measured: set[str] = field(default_factory=set)
    _history: dict[str, list[PassRecord]] = field(default_factory=dict)
    _queue: list[_Queued] = field(default_factory=list)
    _tasks: set[asyncio.Task[None]] = field(default_factory=set)
    _laps_seen: int = 0
    plans: list[LapPlan] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._ref_metrics = measure_lap(self.reference.lap, self.corners, edges=self.edges)
        self._ref_by_name = {m.name: m for m in self._ref_metrics}

    @property
    def length_m(self) -> float:
        return self.reference.lap.length_m

    # -- public API -----------------------------------------------------------
    async def feed(self, sample: Sample) -> None:
        prev = self._samples[-1] if self._samples else None
        if prev is not None and sample.lap != prev.lap:
            finished = self._samples
            self._samples = [sample]
            self._fired.clear()
            self._measured.clear()
            self._on_lap_complete(finished, sample.last_ms, sample.t)
            return
        self._samples.append(sample)
        if prev is None or sample.in_pit or sample.speed_kmh < 20:
            return
        a, b = prev.pos * self.length_m, sample.pos * self.length_m
        if b < a:  # spline wrapped without a lap change
            return
        if self._cues_dirty:
            self._cues = self.pace.cues()
            self._cues_dirty = False
        self._fire_cues(a, b)
        self._check_corner_exits(b, sample.t)
        self._speak_queued(sample)

    def pending(self) -> int:
        return len(self._tasks)

    async def drain(self) -> None:
        """Wait for in-flight model calls (tests, replay, shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # -- position cues ----------------------------------------------------------
    def _fire_cues(self, a: float, b: float) -> None:
        for cue in self._cues:
            key = (cue.corner, cue.kind)
            if key in self._fired or not (a < cue.at_m <= b):
                continue
            self._fired.add(key)
            self.speaker.say(cue.text, urgent=True)
            self._event("cue", corner=cue.corner, kind=cue.kind, say=cue.text, at_m=round(b))

    # -- quiet-moment speech -----------------------------------------------------
    def _queue_say(
        self, text: str, t: float, *, max_age_s: float, kind: str, front: bool = False
    ) -> None:
        if not text:
            return
        item = _Queued(text, t, max_age_s, kind)
        if front:
            self._queue.insert(0, item)
        else:
            self._queue.append(item)

    def _seconds_to_next_cue(self, s_m: float, speed_kmh: float) -> float:
        if not self._cues:
            return float("inf")
        v = max(speed_kmh, 30.0) / 3.6
        ahead = min((c.at_m - s_m) % self.length_m for c in self._cues)
        return ahead / v

    def _speak_queued(self, sample: Sample) -> None:
        self._queue = [q for q in self._queue if sample.t - q.created_t <= q.max_age_s]
        if not self._queue:
            return
        gap = self._seconds_to_next_cue(sample.pos * self.length_m, sample.speed_kmh)
        item = self._queue[0]
        if gap >= speech_seconds(item.text) + 0.3:
            self._queue.pop(0)
            self.speaker.say(item.text)
            self._event("said", kind=item.kind, say=item.text)

    # -- per-corner passes --------------------------------------------------------
    def _lap_no(self) -> int:
        return (self._samples[0].lap + 1) if self._samples else 0

    def _check_corner_exits(self, s_m: float, t: float) -> None:
        if len(self._samples) < 20:
            return
        start_s = self._samples[0].pos * self.length_m
        for m in self._ref_metrics:
            if m.name in self._measured or s_m < m.end_m + 8:
                continue
            self._measured.add(m.name)
            if start_s > max(m.start_m - 60, 0.0) and start_s < m.end_m:
                continue  # this lap's recording began inside or just before the corner
            try:
                trace = LapTrace(self._samples, length_m=self.length_m, lap_time_s=0.0)
                got = measure_lap(trace, self.corners, edges=self.edges, only={m.name})
            except ValueError:
                continue
            if not got:
                continue
            row = CornerComparison(got[0], m)
            self._on_pass(row, t)

    def _on_pass(self, row: CornerComparison, t: float) -> None:
        name = row.name
        history = self._history.setdefault(name, [])
        mark = self.pace.marks.get(name)
        record = PassRecord(lap=self._lap_no(), row=row, mark_m=mark)
        d = row.driver
        self._event(
            "pass",
            corner=name,
            lap=record.lap,
            section_s=round(d.section_s, 3),
            lost_vs_ai_s=round(row.time_lost_s, 3),
            brake_m=None if d.brake_m is None else round(d.brake_m),
            mark_m=None if mark is None else round(mark),
            min_kmh=round(d.min_kmh, 1),
            turn_in_m=None if d.turn_in_m is None else round(d.turn_in_m),
            apex_inside_m=d.apex_inside_m,
            exit_outside_m=d.exit_outside_m,
        )
        quick = quick_verdict(row, history[-1] if history else None, mark)
        if self.advisor is None:
            self._queue_say(quick, t, max_age_s=10.0, kind="verdict")
            record.said_after = quick
            history.append(record)
            return
        message = pass_message(
            row,
            lap=record.lap,
            mark=mark,
            target_min=self.target_min_kmh.get(name),
            history=history,
            note=self.pace.notes[name],
        )
        history.append(record)
        self._spawn(self._advise(name, message, quick, record, t))

    async def _advise(
        self, name: str, message: str, quick: str, record: PassRecord, t: float
    ) -> None:
        assert self.advisor is not None
        started = time.monotonic()
        try:
            advice = await asyncio.wait_for(
                self.advisor.advise(message, self.pace.notes[name]), self.advisor_timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - coaching must never stop the session
            _log.warning("advisor-failed", corner=name, error=repr(exc))
            self._queue_say(quick, t, max_age_s=10.0, kind="verdict")
            record.said_after = quick
            return
        advice.note.objective = self.pace.notes[name].objective
        self.pace.notes[name] = advice.note
        self._cues_dirty = True
        now = quick if advice.rejected else advice.now
        record.said_after = now
        self._queue_say(now, t, max_age_s=12.0, kind="verdict")
        self._save_state()
        self._event(
            "advice",
            corner=name,
            latency_s=round(time.monotonic() - started, 2),
            now=advice.now,
            say=advice.note.say,
            apex_say=advice.note.apex_say,
            call_turn=advice.note.call_turn,
            call_throttle=advice.note.call_throttle,
        )

    # -- laps ---------------------------------------------------------------------
    def _on_lap_complete(self, samples: list[Sample], lap_ms: int, t: float) -> None:
        self._laps_seen += 1
        lap_no = samples[-1].lap + 1
        if lap_ms <= 0 or len(samples) < 50:
            return
        lap_s = lap_ms / 1000.0
        ref_s = self.reference.lap_time_s
        self._queue_say(say_lap_time(lap_s, ref_s), t, max_age_s=30.0, kind="lap-time", front=True)
        flags: list[str] = []
        pit = any(s.in_pit for s in samples)
        if pit:
            flags.append("lap included the pit lane (out-lap or in-lap)")
        if self._laps_seen == 1:
            flags.append("first lap seen this session (may be partial or an out-lap)")
        if lap_s > ref_s + 8:
            flags.append("slow or messy lap (more than 8 s off the reference)")
        try:
            trace = LapTrace(samples, length_m=self.length_m, lap_time_s=lap_s)
            rows = compare(measure_lap(trace, self.corners, edges=self.edges), self._ref_metrics)
        except ValueError:
            return
        self._event(
            "lap",
            lap=lap_no,
            time_s=lap_s,
            flags=flags,
            corners=[{"corner": r.name, "lost_s": round(r.time_lost_s, 3)} for r in rows],
        )
        if self.head is not None and not pit:
            self._spawn(self._head_plan(lap_no, lap_s, rows, flags, t))

    async def _head_plan(
        self, lap_no: int, lap_s: float, rows: list[CornerComparison], flags: list[str], t: float
    ) -> None:
        assert self.head is not None
        started = time.monotonic()
        try:
            plan = await asyncio.wait_for(
                self.head.plan(lap=lap_no, lap_time_s=lap_s, rows=rows, flags=flags),
                self.head_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("head-coach-failed", error=repr(exc))
            return
        self.plans.append(plan)
        for name, note in self.pace.notes.items():
            note.objective = plan.focus.get(name, "")
        self._queue_say(plan.lap_comment, t, max_age_s=45.0, kind="lap-comment")
        self._save_state()
        self._event(
            "plan",
            lap=lap_no,
            latency_s=round(time.monotonic() - started, 2),
            comment=plan.lap_comment,
            focus=plan.focus,
            notes=plan.notes,
            usage=plan.usage,
        )

    # -- plumbing -------------------------------------------------------------------
    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _save_state(self) -> None:
        if self.state_path is not None:
            self.pace.save(self.state_path)

    def usage_summary(self) -> str:
        parts = []
        for client in self.clients:
            total = getattr(client, "total", None)
            if total is None or not total.calls:
                continue
            parts.append(
                f"{client.model}: {total.calls} calls, in {total.input_tokens} + cache "
                f"{total.cache_read_tokens}r/{total.cache_write_tokens}w, out "
                f"{total.output_tokens}, est. ${total.cost_usd:.4f}"
            )
        laps = max(self._laps_seen, 1)
        cost = sum(c.total.cost_usd for c in self.clients if hasattr(c, "total"))
        return (
            " | ".join(parts) + f" | total ${cost:.4f} (~${cost / laps:.4f} per lap)"
            if parts
            else "Claude calls: 0"
        )

    def finish(self) -> None:
        self._event("session-usage", summary=self.usage_summary())

    def _event(self, event_name: str, /, **data: Any) -> None:
        _log.info(event_name, **data)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                record = {"t": round(time.time(), 2), "event": event_name, **data}
                fh.write(json.dumps(record) + "\n")
