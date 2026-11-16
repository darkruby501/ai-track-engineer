"""Application entry point and command-line interface.

Phase 0 provides the CLI skeleton and an environment ``doctor`` command. The
real-time telemetry orchestration is wired in from Phase 1 onwards.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Annotated

import typer
from fastapi import FastAPI

from src import __version__
from src.ai import OllamaClient, RaceEngineerAdvisor
from src.analysis import EngineMonitor
from src.config import find_ac_install, get_settings
from src.dashboard import DashboardState, TelemetryHub, create_app, run_session
from src.knowledge import TrackInfo, load_track, map_png_path
from src.observability import configure_logging, get_logger
from src.storage import SqliteStore
from src.storage.session_recorder import SessionRecorder
from src.telemetry import (
    MockTelemetrySource,
    SharedMemoryTelemetrySource,
    TelemetrySource,
)
from src.telemetry.models import ACStaticInfo
from src.telemetry.shm_reader import SharedMemoryUnavailableError

app = typer.Typer(
    name="ai-track-engineer",
    help="AI-powered race engineer for Assetto Corsa.",
    add_completion=False,
    no_args_is_help=True,
)


coach_app = typer.Typer(help="Claude-powered driving coach.", no_args_is_help=True)
app.add_typer(coach_app, name="coach")


def _coach_track(track: str, layout: str) -> TrackInfo:
    settings = get_settings()
    ac_path = find_ac_install(settings.app.ac_install_path)
    track_dir = Path(ac_path) / "content" / "tracks" / track if ac_path else Path(track)
    return load_track(track_dir, track_id=track, layout=layout)


@coach_app.command("run")
def coach_run(
    offline: bool = typer.Option(
        False, "--offline", help="No Claude calls: simple built-in coaching only."
    ),
    no_voice: bool = typer.Option(False, "--no-voice", help="Log calls instead of speaking."),
) -> None:  # pragma: no cover - needs Assetto Corsa
    """Coach live AC sessions (run this on the sim PC, from the desktop)."""
    from src.coach.runner import run_live

    settings = get_settings()
    configure_logging(settings.app.log_level)
    coach_settings = settings.coach.model_copy(update={"voice": not no_voice})
    poll_hz = settings.app.telemetry_poll_hz
    asyncio.run(run_live(coach_settings, use_llm=not offline, poll_hz=poll_hz))


@coach_app.command("make-reference")
def coach_make_reference(
    csv_path: Annotated[Path, typer.Argument(help="Recording (AI reference or coach CSV).")],
    track: str = typer.Option("nd_buttonwillow", help="AC track folder name."),
    layout: str = typer.Option("layout_classic_13", help="AC layout folder name."),
    car: str = typer.Option("ks_mazda_miata", help="AC car folder name."),
    length_m: float = typer.Option(..., "--length-m", help="Track spline length in metres."),
    source: str = typer.Option("AC AI", help="Where the lap came from (shown to the coach)."),
) -> None:
    """Save the fastest flying lap of a recording as the reference lap."""
    from src.coach.runner import make_reference

    out = make_reference(
        csv_path,
        track=track,
        layout=layout,
        car=car,
        length_m=length_m,
        out_dir=Path(get_settings().coach.reference_dir),
        source=source,
    )
    typer.echo(f"reference saved: {out}")


@coach_app.command("replay")
def coach_replay(
    csv_path: Annotated[Path, typer.Argument(help="Recorded session CSV.")],
    track: str = typer.Option("nd_buttonwillow"),
    layout: str = typer.Option("layout_classic_13"),
    car: str = typer.Option("ks_mazda_miata"),
    track_dir: Annotated[
        Path | None, typer.Option(help="Track folder for edges (default: AC install).")
    ] = None,
    offline: bool = typer.Option(False, "--offline", help="No Claude calls."),
    voice: bool = typer.Option(False, "--voice", help="Speak (Mac/Windows)."),
    realtime: float = typer.Option(0.0, help="Playback speed; 0 = as fast as possible."),
    max_laps: int = typer.Option(0, help="Stop after this many laps (0 = all)."),
) -> None:  # pragma: no cover - integration
    """Run the coach over a recorded session, printing what it would say."""
    from src.coach.reference import load_reference, reference_path
    from src.coach.runner import build_coach, prepare_notes, replay, speaker_from_settings
    from src.coach.samples import CSV_COLUMNS, read_samples_csv
    from src.knowledge.track_edges import load_edges

    settings = get_settings()
    configure_logging(settings.app.log_level)
    reference = load_reference(reference_path(settings.coach.reference_dir, track, layout, car))
    speaker = speaker_from_settings(settings.coach, enabled=voice)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = Path(settings.coach.sessions_dir) / f"replay_{stamp}.jsonl"
    if track_dir is None:
        ac_path = find_ac_install(settings.app.ac_install_path)
        track_dir = Path(ac_path) / "content" / "tracks" / track if ac_path else Path(track)
    coach = build_coach(
        settings.coach,
        track=_coach_track(track, layout),
        reference=reference,
        speaker=speaker,
        use_llm=not offline,
        log_path=log_path,
        edges=load_edges(track_dir, layout=layout),
    )
    source = csv_path
    if max_laps > 0:
        import csv as _csv
        import tempfile

        rows = list(read_samples_csv(csv_path))
        keep = [s for s in rows if s.lap <= rows[0].lap + max_laps]
        source = Path(tempfile.mkstemp(suffix=".csv")[1])
        with source.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.writer(fh)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(s.row() for s in keep)

    async def go() -> None:
        if await prepare_notes(coach, settings.coach):
            typer.echo("Claude wrote the initial notes:")
            for name, note in coach.pace.notes.items():
                typer.echo(f"  {name}: '{note.say}' | apex '{note.apex_say}'")
        coach.state_path = None  # keep replays from rewriting the live notes
        await replay(coach, source, realtime=realtime)

    asyncio.run(go())
    coach.finish()
    typer.echo("\nSpoken:")
    for line in speaker.spoken:
        typer.echo(f"  > {line}")
    typer.echo(f"\n{coach.usage_summary()}")
    typer.echo(f"log: {log_path}")


@coach_app.command("target-line")
def coach_target_line(
    track: str = typer.Option("nd_buttonwillow"),
    layout: str = typer.Option("layout_classic_13"),
    car: str = typer.Option("ks_mazda_miata"),
    track_dir: Annotated[
        Path | None, typer.Option(help="Track folder (default: AC install).")
    ] = None,
    base_margin_m: float = typer.Option(0.3, help="Wheel margin inside the edges, metres."),
    lat_g: float = typer.Option(0.9, help="Cornering grip used for target speeds."),
    brake_g: float = typer.Option(0.95, help="Braking used for target braking points."),
) -> None:
    """Build the conservative target line, marks and an AC ideal-line file."""
    import json

    import numpy as np

    from src.coach.metrics import corner_spans
    from src.coach.reference import load_reference, reference_path
    from src.coach.target_line import (
        DEFAULT_RISK_ZONES,
        CarModel,
        build_conservative_target,
        read_ai_spline,
        to_ideal_line,
        write_ai_spline,
    )

    settings = get_settings()
    reference = load_reference(reference_path(settings.coach.reference_dir, track, layout, car))
    if track_dir is None:
        ac_path = find_ac_install(settings.app.ac_install_path)
        track_dir = Path(ac_path) / "content" / "tracks" / track if ac_path else Path(track)
    base = track_dir / layout if (track_dir / layout).is_dir() else track_dir
    spline = read_ai_spline(base / "ai" / "fast_lane.ai")
    right = np.loadtxt(base / "data" / "side_r.csv", delimiter=",", usecols=(0, 1, 2), ndmin=2)
    info = load_track(track_dir, track_id=track, layout=layout)
    spans = corner_spans(info.corners, reference.lap.length_m)
    target = build_conservative_target(
        spline,
        np.c_[right[:, 0], -right[:, 2]],
        ref_s=reference.lap.s,
        ref_speed_kmh=reference.lap.speed,
        car=CarModel(lat_g=lat_g, brake_g=brake_g),
        base_margin_m=base_margin_m,
        risk_zones=DEFAULT_RISK_ZONES.get((track, layout), []),
    )
    marks = target.brake_points(spans)
    out_dir = Path(settings.coach.notes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{track}__{layout}__{car}"
    payload = {
        "lap_time_model_s": round(target.lap_time_s, 2),
        "marks": {k: round(v, 1) for k, v in marks.items()},
        "corner_min_kmh": {
            name: round(
                float(target.v[(target.s_spline >= a) & (target.s_spline <= b)].min()) * 3.6, 1
            )
            for name, a, b in spans
        },
        "settings": {"base_margin_m": base_margin_m, "lat_g": lat_g, "brake_g": brake_g},
    }
    (out_dir / f"{stem}__target.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    write_ai_spline(to_ideal_line(spline, target), out_dir / f"{stem}__ideal_line.ai")
    typer.echo(json.dumps(payload, indent=1))
    typer.echo(f"wrote {out_dir / (stem + '__target.json')} and {stem}__ideal_line.ai")


@coach_app.command("debrief")
def coach_debrief(
    csv_path: Annotated[Path, typer.Argument(help="Session CSV written by `coach run`.")],
    track: str = typer.Option("nd_buttonwillow"),
    layout: str = typer.Option("layout_classic_13"),
    car: str = typer.Option("ks_mazda_miata"),
) -> None:  # pragma: no cover - network
    """Write a Claude debrief of a recorded session next to its CSV."""
    from src.coach.reference import load_reference, reference_path
    from src.coach.runner import debrief

    settings = get_settings()
    configure_logging(settings.app.log_level)
    reference = load_reference(reference_path(settings.coach.reference_dir, track, layout, car))
    text = asyncio.run(
        debrief(settings.coach, csv_path, reference=reference, track=_coach_track(track, layout))
    )
    out = csv_path.with_name(csv_path.stem + "_debrief.md")
    out.write_text(text, encoding="utf-8")
    typer.echo(text)
    typer.echo(f"\nsaved: {out}")


@app.command()
def version() -> None:
    """Print the installed version and exit."""
    typer.echo(f"ai-track-engineer {__version__}")


@app.command()
def report() -> None:
    """Print a summary of the most recent recorded session."""
    settings = get_settings()
    configure_logging(settings.app.log_level)
    asyncio.run(_print_latest_report(settings.app.database_path))


async def _print_latest_report(database_path: str) -> None:  # pragma: no cover - integration
    """Load the latest session from SQLite and print its report."""
    from src.analysis import build_session_report

    async with SqliteStore(database_path) as store:
        session = await store.latest_session()
        if session is None:
            typer.echo("No recorded sessions found.")
            return
        laps = await store.laps_for_session(session.id)
    summary = build_session_report(laps)
    typer.echo(f"Session on {session.track} ({session.car})")
    typer.echo(f"  Laps: {summary.valid_laps} valid / {summary.total_laps} total")
    if summary.best_lap_seconds is not None:
        typer.echo(f"  Best lap: {summary.best_lap_seconds:.3f}s")
    if summary.average_lap_ms is not None:
        typer.echo(f"  Average: {summary.average_lap_ms / 1000:.3f}s")
    if summary.consistency_stdev_ms is not None:
        typer.echo(f"  Consistency (stdev): {summary.consistency_stdev_ms / 1000:.3f}s")
    if summary.theoretical_best_ms is not None:
        typer.echo(f"  Theoretical best: {summary.theoretical_best_ms / 1000:.3f}s")


@app.command()
def doctor() -> None:
    """Check the local environment (settings, Assetto Corsa, services)."""
    settings = get_settings()
    configure_logging(settings.app.log_level)
    log = get_logger("doctor")

    ac_path = find_ac_install(settings.app.ac_install_path)
    log.info(
        "environment-check",
        version=__version__,
        ac_install="found" if ac_path else "not-found",
        ac_path=str(ac_path) if ac_path else None,
        influx_url=settings.influxdb.url,
        ollama_url=settings.ollama.url,
        ollama_model=settings.ollama.model,
    )
    if ac_path is None:
        typer.echo(
            "Assetto Corsa installation not found. Set AC_INSTALL_PATH in .env "
            "to point at your installation."
        )


@app.command()
def run(
    mock: bool = typer.Option(
        False, "--mock", help="Use simulated telemetry instead of a live session."
    ),
    voice: bool = typer.Option(
        False, "--voice", help="Enable the bidirectional voice assistant."
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Dashboard bind address."),
    port: int = typer.Option(8000, "--port", help="Dashboard port."),
) -> None:
    """Start the engineer: capture telemetry and serve the live dashboard.

    Connects a telemetry source (live or ``--mock``), fans frames out to the web
    dashboard, and segments laps. Open http://HOST:PORT to view live telemetry.
    Press Ctrl+C to stop.
    """
    settings = get_settings()
    configure_logging(settings.app.log_level)
    log = get_logger("run")

    if voice:
        log.warning("voice-not-implemented", note="voice arrives in Phase 6")

    source: TelemetrySource = (
        MockTelemetrySource() if mock else SharedMemoryTelemetrySource()
    )
    hub = TelemetryHub()
    state = DashboardState()

    async def producer() -> None:
        await _capture_and_analyze(source, hub, state, hz=settings.app.telemetry_poll_hz)

    app_instance = create_app(hub, state, producer=producer)
    log.info(
        "startup",
        mode="mock" if mock else "live",
        dashboard=f"http://{host}:{port}",
        poll_hz=settings.app.telemetry_poll_hz,
    )
    _serve(app_instance, host=host, port=port, log_level=settings.app.log_level)


def _serve(  # pragma: no cover - integration entry point
    app_instance: FastAPI, *, host: str, port: int, log_level: str
) -> None:
    """Run the ASGI app with uvicorn (integration entry point)."""
    import uvicorn

    uvicorn.run(app_instance, host=host, port=port, log_level=log_level.lower())


def _track_dir_for(static: ACStaticInfo) -> Path | None:  # pragma: no cover - integration
    """Resolve the Assetto Corsa content directory for the active track."""
    settings = get_settings()
    ac_path = find_ac_install(settings.app.ac_install_path)
    if ac_path is not None and static.track:
        track_dir = Path(ac_path) / "content" / "tracks" / static.track
        if track_dir.is_dir():
            return track_dir
    return None


def _load_track_info(static: ACStaticInfo) -> TrackInfo:  # pragma: no cover - integration
    """Load track knowledge for the active session, with a safe fallback."""
    track_dir = _track_dir_for(static)
    if track_dir is not None:
        return load_track(track_dir, layout=static.track_configuration or "")
    label = static.track or "unknown"
    return TrackInfo(track_id=label, name=label)


async def _connect_when_ready(  # pragma: no cover - integration entry point
    source: TelemetrySource, *, poll_seconds: float = 2.0
) -> ACStaticInfo:
    """Connect to *source*, waiting for Assetto Corsa if it is not ready yet.

    The live shared-memory source raises :class:`SharedMemoryUnavailableError`
    until AC has created its regions, so we poll instead of failing. This lets
    the operator start the dashboard before or after launching the game, and it
    never pre-creates the shared-memory regions (which would crash AC).
    """
    log = get_logger("analysis")
    waiting = False
    while True:
        try:
            return source.connect()
        except SharedMemoryUnavailableError:
            if not waiting:
                log.info(
                    "waiting-for-ac",
                    note="start Assetto Corsa and enter a session to begin",
                )
                waiting = True
            await asyncio.sleep(poll_seconds)


async def _capture_and_analyze(  # pragma: no cover - integration entry point
    source: TelemetrySource,
    hub: TelemetryHub,
    state: DashboardState,
    *,
    hz: int,
) -> None:
    """Connect, load track + map, and stream the session's events to the hub."""
    static = await _connect_when_ready(source)
    track = _load_track_info(static)
    track_dir = _track_dir_for(static)
    if track_dir is not None:
        state.set_map_png(map_png_path(track_dir, layout=static.track_configuration or ""))
    engine_monitor = EngineMonitor(static.max_rpm) if static.max_rpm > 0 else None
    settings = get_settings()
    advisor = RaceEngineerAdvisor(OllamaClient.from_settings(settings.ollama))
    log = get_logger("analysis")

    db_path = Path(settings.app.database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with SqliteStore(str(db_path)) as store:
        session = await store.create_session(
            track=track.track_id,
            car=static.car_model,
            started_at=time.time(),
            track_config=static.track_configuration or "",
        )
        log.info(
            "session",
            track=track.name,
            car=static.car_model,
            corners=track.corner_count,
            ai_model=settings.ollama.model,
            session_id=session.id,
        )
        best_ever_ms = await store.best_lap_ms_for(
            track=track.track_id, car=static.car_model
        )
        sessions_dir = Path("data/sessions")
        recorder = SessionRecorder(
            sessions_dir, track.track_id, static.car_model, session.id
        )
        with recorder:
            log.info("recording-telemetry", path=str(recorder.path))
            await run_session(
                source,
                hub,
                state,
                track,
                static,
                hz=hz,
                advisor=advisor,
                engine_monitor=engine_monitor,
                store=store,
                session_id=session.id,
                recorder=recorder,
                best_ever_ms=best_ever_ms,
            )
            log.info("recording-complete", rows=recorder.rows_written)


if __name__ == "__main__":  # pragma: no cover
    app()
