"""Wiring for the ``coach`` CLI: live sessions, replays, references, debriefs."""

from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Any

import structlog

from src.ai.anthropic_client import AnthropicClient, Effort
from src.coach.coaches import CoachContext, CornerAdvisor, HeadCoach, write_initial_notes
from src.coach.live import LiveCoach
from src.coach.metrics import CornerComparison, compare, comparison_table, measure_lap
from src.coach.pacenotes import PaceNotes
from src.coach.reference import (
    Reference,
    best_lap,
    load_reference,
    reference_path,
    save_reference,
)
from src.coach.samples import CSV_COLUMNS, LapTrace, Sample, read_samples_csv, split_laps
from src.coach.speaker import LogSpeaker, make_speaker
from src.config.settings import CoachSettings
from src.knowledge.models import TrackInfo
from src.knowledge.track_edges import TrackEdges

_log = structlog.get_logger("coach.runner")


def _effort(value: str) -> Effort:
    allowed: tuple[Effort, ...] = ("low", "medium", "high", "xhigh", "max")
    for level in allowed:
        if value == level:
            return level
    raise ValueError(f"unknown effort {value!r}")


def state_stem(track: str, layout: str, car: str) -> str:
    return f"{track}__{layout}__{car}"


def load_target(settings: CoachSettings, stem: str) -> dict[str, Any]:
    path = Path(settings.notes_dir) / f"{stem}__target.json"
    return dict(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else {}


def build_coach(
    settings: CoachSettings,
    *,
    track: TrackInfo,
    reference: Reference,
    speaker: LogSpeaker,
    use_llm: bool,
    log_path: Path | None,
    edges: TrackEdges | None = None,
    head_llm: Any | None = None,
    fast_llm: Any | None = None,
) -> LiveCoach:
    """Assemble the live coach: pace notes, marks, Claude roles and speaker."""
    stem = state_stem(reference.track, reference.layout, reference.car)
    target = load_target(settings, stem)
    marks = {k: float(v) for k, v in target.get("marks", {}).items()}
    target_min = {k: float(v) for k, v in target.get("corner_min_kmh", {}).items()}
    ref_metrics = measure_lap(reference.lap, track.corners, edges=edges)
    pace = PaceNotes(
        ref_metrics=ref_metrics,
        ref_lap=reference.lap,
        marks=dict(marks),
        seed_marks=dict(marks),
        lead_s=settings.pace_lead_s,
        latency_s=settings.cue_latency_s,
    )
    state_path = Path(settings.notes_dir) / f"{stem}__notes.json"
    state = PaceNotes.load_state(state_path)
    state.pop("marks", None)  # marks always come from the current target
    state.pop("seed_marks", None)
    pace.apply_state(state)
    advisor = head = None
    clients: list[Any] = []
    context: CoachContext | None = None
    head_client: Any = None
    if use_llm:
        context = CoachContext(
            track_name=track.name,
            car=reference.car,
            reference_time_s=reference.lap_time_s,
            reference_metrics=ref_metrics,
            marks=pace.marks,
            target_min_kmh=target_min,
            context_files=[Path(p) for p in settings.context_files],
        )
        head_client = head_llm or AnthropicClient(
            model=settings.model,
            effort=_effort(settings.lap_effort),
            speed=settings.speed,
            timeout_s=settings.request_timeout_s,
        )
        fast_client = fast_llm or AnthropicClient(
            model=settings.fast_model, effort="low", timeout_s=15.0, thinking_off=True
        )
        head = HeadCoach(head_client, context, [m.name for m in ref_metrics])
        advisor = CornerAdvisor(fast_client, context)
        clients = [head_client, fast_client]
    coach = LiveCoach(
        corners=track.corners,
        reference=reference,
        speaker=speaker,
        pace=pace,
        edges=edges,
        advisor=advisor,
        head=head,
        target_min_kmh=target_min,
        log_path=log_path,
        state_path=state_path,
        head_timeout_s=settings.request_timeout_s,
        clients=clients,
        context=context,
        head_client=head_client,
        notes_from_claude=state.get("notes_source") == "claude",
    )
    return coach


async def prepare_notes(coach: LiveCoach, settings: CoachSettings) -> bool:
    """Have the head model write initial notes once (they are saved and reused)."""
    if coach.head_client is None or coach.context is None or coach.notes_from_claude:
        return False
    try:
        notes = await asyncio.wait_for(
            write_initial_notes(coach.head_client, coach.context, coach.pace.notes),
            settings.request_timeout_s * 2,
        )
    except Exception as exc:  # noqa: BLE001 - template notes still work
        _log.warning("initial-notes-failed", error=repr(exc))
        return False
    coach.pace.notes.update(notes)
    coach._cues_dirty = True
    if coach.state_path is not None:
        coach.pace.save(coach.state_path)
        data = json.loads(coach.state_path.read_text(encoding="utf-8"))
        data["notes_source"] = "claude"
        coach.state_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    coach.notes_from_claude = True
    return True


# -- reference packs ---------------------------------------------------------
def make_reference(
    csv_path: Path,
    *,
    track: str,
    layout: str,
    car: str,
    length_m: float,
    out_dir: Path,
    source: str,
) -> Path:
    samples = list(read_samples_csv(csv_path))
    lap, _ = best_lap(samples, length_m=length_m)
    ref = Reference(track=track, layout=layout, car=car, source=source, lap=lap)
    return save_reference(ref, reference_path(out_dir, track, layout, car))


# -- replay ------------------------------------------------------------------
async def replay(coach: LiveCoach, csv_path: Path, *, realtime: float = 0.0) -> None:
    """Feed a recorded session through *coach*.

    ``realtime=0`` runs as fast as possible and waits for each lap's plan before
    continuing, so calls land on the following lap as they would on track.
    """
    previous: Sample | None = None
    for sample in read_samples_csv(csv_path):
        if realtime > 0 and previous is not None:
            await asyncio.sleep(max(0.0, (sample.t - previous.t) / realtime))
        await coach.feed(sample)
        if realtime <= 0 and coach.pending():
            await coach.drain()
        previous = sample
    await coach.drain()


# -- live --------------------------------------------------------------------
class _CsvWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        self._w.writerow(CSV_COLUMNS)
        self._n = 0
        self.path = path

    def write(self, sample: Sample) -> None:
        self._w.writerow(sample.row())
        self._n += 1
        if self._n % 120 == 0:
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


async def run_live(
    settings: CoachSettings, *, use_llm: bool, poll_hz: int = 60
) -> None:  # pragma: no cover - needs AC
    """Coach every AC session until interrupted (reconnects between sessions)."""
    import os

    from src.config import find_ac_install, get_settings
    from src.knowledge import load_track
    from src.knowledge.track_edges import load_edges
    from src.telemetry import SharedMemoryTelemetrySource
    from src.telemetry.shm_reader import SharedMemoryUnavailableError
    from src.telemetry.shm_structs import ACStatus

    speaker = make_speaker(enabled=settings.voice, rate=settings.voice_rate)
    if use_llm and not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        _log.warning("no-anthropic-key", note="set ANTHROPIC_API_KEY; using basic coaching")
        speaker.say("No Claude key found. Basic coaching only.")
        use_llm = False
    ac_path = find_ac_install(get_settings().app.ac_install_path)
    try:
        while True:
            source = SharedMemoryTelemetrySource()
            waiting_logged = False
            while True:
                try:
                    static = source.connect()
                    break
                except SharedMemoryUnavailableError:
                    if not waiting_logged:
                        _log.info("waiting-for-ac")
                        waiting_logged = True
                    await asyncio.sleep(2.0)
            # Shared memory exists while AC runs; wait for a live session.
            while source.read_frame().graphics.status != ACStatus.LIVE:
                await asyncio.sleep(1.0)
            static = source.connect()
            layout = static.track_configuration or ""
            ref_file = reference_path(
                settings.reference_dir, static.track, layout, static.car_model
            )
            if not ref_file.is_file():
                _log.error("no-reference", expected=str(ref_file))
                speaker.say("No reference lap for this car and track. Coaching is off.")
                await _wait_for_session_end(source)
                continue
            reference = load_reference(ref_file)
            track_dir = Path(ac_path) / "content" / "tracks" / static.track if ac_path else None
            track = (
                load_track(track_dir, layout=layout)
                if track_dir
                else TrackInfo(track_id=static.track, name=static.track)
            )
            stamp = time.strftime("%Y%m%d-%H%M%S")
            base = (
                Path(settings.sessions_dir)
                / f"coach_{static.track}_{layout}_{static.car_model}_{stamp}"
            )
            writer = _CsvWriter(base.with_suffix(".csv"))
            edges = load_edges(track_dir, layout=layout) if track_dir else None
            coach = build_coach(
                settings,
                track=track,
                reference=reference,
                speaker=speaker,
                use_llm=use_llm,
                log_path=base.with_suffix(".jsonl"),
                edges=edges,
            )
            notes_task = asyncio.get_running_loop().create_task(prepare_notes(coach, settings))
            _log.info("coaching", track=track.name, car=static.car_model, csv=str(writer.path))
            speaker.say("Coach ready.")
            last_packet = -1
            try:
                while True:
                    try:
                        frame = source.read_frame()
                    except Exception:  # noqa: BLE001 - AC closed
                        break
                    status = frame.graphics.status
                    if status == ACStatus.OFF:
                        break
                    if status == ACStatus.LIVE and frame.physics.packet_id != last_packet:
                        last_packet = frame.physics.packet_id
                        sample = Sample.from_frame(frame)
                        writer.write(sample)
                        await coach.feed(sample)
                    await asyncio.sleep(1.0 / poll_hz)
            finally:
                writer.close()
                source.close()
                notes_task.cancel()
                await coach.drain()
                coach.finish()
                _log.info("session-ended", csv=str(writer.path), usage=coach.usage_summary())
    finally:
        speaker.close()


async def _wait_for_session_end(source: Any) -> None:  # pragma: no cover - needs AC
    from src.telemetry.shm_structs import ACStatus

    while True:
        try:
            if source.read_frame().graphics.status == ACStatus.OFF:
                break
        except Exception:  # noqa: BLE001
            break
        await asyncio.sleep(2.0)
    source.close()


# -- debrief -----------------------------------------------------------------
_DEBRIEF_RULES = """\
You are the same driving coach, now writing a short written debrief after a \
practice session in Assetto Corsa. Be specific and use only the data provided. \
Structure: 1) Headline (best lap vs reference, trend over the session). \
2) What improved, with numbers. 3) The three biggest remaining gains, each with \
the corner, the cause in the data, and a concrete drill for next session. \
4) Consistency: which corners vary most lap to lap. Keep it under 400 words, in \
Markdown.
"""


def session_laps(
    csv_path: Path, reference: Reference, track: TrackInfo
) -> list[tuple[int, float, list[CornerComparison]]]:
    ref_metrics = measure_lap(reference.lap, track.corners)
    out: list[tuple[int, float, list[CornerComparison]]] = []
    for lap_no, lap_samples, lap_ms in split_laps(list(read_samples_csv(csv_path))):
        if lap_ms <= 0 or any(s.in_pit for s in lap_samples):
            continue
        try:
            trace = LapTrace(lap_samples, length_m=reference.lap.length_m, lap_time_s=lap_ms / 1000)
        except ValueError:
            continue
        out.append(
            (lap_no + 1, lap_ms / 1000, compare(measure_lap(trace, track.corners), ref_metrics))
        )
    return out


def debrief_message(
    laps: list[tuple[int, float, list[CornerComparison]]], reference: Reference, events: list[str]
) -> str:
    lines = [f"Reference lap: {reference.lap_time_s:.3f}s ({reference.source})."]
    lines.append("Laps: " + ", ".join(f"#{n} {t:.3f}s" for n, t, _ in laps))
    if laps:
        best = min(laps, key=lambda lap: lap[1])
        lines.append(f"\nBest lap #{best[0]} ({best[1]:.3f}s) vs the reference:")
        lines.append(comparison_table(best[2]))
        names = [r.name for r in best[2]]
        lines.append("\nTime lost per corner on every lap (s):")
        for name in names:
            series = [
                next((r.time_lost_s for r in rows if r.name == name), float("nan"))
                for _, _, rows in laps
            ]
            lines.append(f"{name}: " + ", ".join(f"{v:+.2f}" for v in series))
    if events:
        lines.append("\nWhat you told the driver during the session:")
        lines.extend(events)
    return "\n".join(lines)


async def debrief(
    settings: CoachSettings, csv_path: Path, *, reference: Reference, track: TrackInfo
) -> str:  # pragma: no cover - network
    laps = session_laps(csv_path, reference, track)
    events: list[str] = []
    log_file = csv_path.with_suffix(".jsonl")
    if log_file.is_file():
        for line in log_file.read_text(encoding="utf-8").splitlines():
            ev = json.loads(line)
            if ev.get("event") == "plan":
                calls = "; ".join(f"{c['corner']}: {c['say']}" for c in ev.get("calls", []))
                events.append(f"- after lap {ev.get('lap')}: {ev.get('comment')} | {calls}")
    client = AnthropicClient(
        model=settings.model,
        effort=_effort(settings.debrief_effort),
        speed=settings.speed,
        timeout_s=max(settings.request_timeout_s, 180.0),
    )
    context = [Path(p) for p in settings.context_files]
    system = (
        _DEBRIEF_RULES
        + "\n\n"
        + "\n\n".join(p.read_text(encoding="utf-8") for p in context if p.is_file())
    )
    return await client.complete(system, debrief_message(laps, reference, events))
