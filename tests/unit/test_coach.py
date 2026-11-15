"""Tests for the Claude driving coach (analysis, planning, live calls, replay)."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from src.ai.anthropic_client import AnthropicClient, ClaudeRefusalError
from src.coach.live import LiveCoach, heuristic_plan, say_lap_time
from src.coach.metrics import compare, comparison_table, measure_lap, reference_table
from src.coach.planner import (
    CoachPlan,
    CoachPlanner,
    CornerCall,
    build_system_prompt,
    plan_schema,
)
from src.coach.reference import (
    Reference,
    best_lap,
    load_reference,
    reference_path,
    save_reference,
)
from src.coach.runner import (
    build_coach,
    debrief_message,
    make_reference,
    replay,
    session_laps,
)
from src.coach.samples import CSV_COLUMNS, LapTrace, Sample, read_samples_csv, split_laps
from src.coach.speaker import LogSpeaker, make_speaker
from src.config.settings import CoachSettings
from src.knowledge.models import Corner, TrackInfo

LENGTH = 1000.0
CORNERS = (
    Corner(index=0, name="Alpha", entry=0.20, exit=0.30),
    Corner(index=1, name="Bravo", entry=0.60, exit=0.70),
)
TRACK = TrackInfo(track_id="test", name="Test Ring", corners=CORNERS)


def _speed_profile(
    s: float, brake_a: float, brake_b: float, min_a: float
) -> tuple[float, float, float]:
    """(speed km/h, gas, brake) at distance *s* for a two-corner synthetic lap."""
    for brake_at, apex, vmin in ((brake_a, 250.0, min_a), (brake_b, 650.0, 90.0)):
        if brake_at <= s < apex:
            frac = (s - brake_at) / (apex - brake_at)
            return 150 - (150 - vmin) * frac, 0.0, 1.0
        if apex <= s < apex + 120:
            frac = (s - apex) / 120
            return vmin + (150 - vmin) * frac, 1.0, 0.0
    return 150.0, 1.0, 0.0


def make_lap(
    lap: int,
    *,
    t0: float,
    brake_a: float = 150.0,
    brake_b: float = 560.0,
    min_a: float = 80.0,
    last_ms: int = 0,
) -> tuple[list[Sample], float]:
    """Samples for one lap at ~60 Hz; returns (samples, lap duration)."""
    samples: list[Sample] = []
    s, t = 0.0, t0
    while s < LENGTH:
        v, gas, brake = _speed_profile(s, brake_a, brake_b, min_a)
        samples.append(
            Sample(
                t=t,
                lap=lap,
                pos=s / LENGTH,
                speed_kmh=v,
                gear=4,
                gas=gas,
                brake=brake,
                current_ms=int((t - t0) * 1000),
                last_ms=last_ms,
            )
        )
        dt = 1 / 60
        s += v / 3.6 * dt
        t += dt
    return samples, t - t0


def make_session(n_laps: int, **kwargs: Any) -> list[Sample]:
    out: list[Sample] = []
    t, last_ms = 0.0, 0
    for lap in range(n_laps):
        samples, duration = make_lap(lap, t0=t, last_ms=last_ms, **kwargs)
        out.extend(samples)
        t += duration
        last_ms = int(duration * 1000)
    # One sample of the following lap so the last lap counts as complete.
    out.append(
        Sample(t=t, lap=n_laps, pos=0.001, speed_kmh=150, gear=4, gas=1, brake=0, last_ms=last_ms)
    )
    return out


def reference() -> Reference:
    samples, duration = make_lap(1, t0=0.0)
    return Reference(
        "test", "", "car", "synthetic", LapTrace(samples, length_m=LENGTH, lap_time_s=duration)
    )


# -- metrics ------------------------------------------------------------------
def test_measure_lap_finds_brake_min_and_throttle() -> None:
    metrics = measure_lap(reference().lap, CORNERS)
    alpha = metrics[0]
    assert alpha.name == "Alpha"
    assert alpha.brake_m == pytest.approx(150, abs=3)
    assert alpha.min_kmh == pytest.approx(80, abs=2)
    assert alpha.min_m == pytest.approx(250, abs=4)
    assert alpha.throttle_m == pytest.approx(250, abs=6)
    assert alpha.entry_kmh == pytest.approx(150)
    assert alpha.min_gear == 3


def test_compare_reports_early_braking_and_time_lost() -> None:
    ref = measure_lap(reference().lap, CORNERS)
    samples, duration = make_lap(1, t0=0.0, brake_a=110.0, min_a=70.0)
    driver = measure_lap(LapTrace(samples, length_m=LENGTH, lap_time_s=duration), CORNERS)
    rows = compare(driver, ref)
    alpha = rows[0]
    assert alpha.brake_diff_m == pytest.approx(-40, abs=4)
    assert alpha.min_speed_diff == pytest.approx(-10, abs=2)
    assert alpha.time_lost_s > 0.2
    assert rows[1].time_lost_s == pytest.approx(0.0, abs=0.1)  # slower exit carries over
    table = comparison_table(rows)
    assert "Alpha" in table and "time lost" in table
    assert "Bravo" in reference_table(ref)


def test_lap_trace_drops_position_strays_around_the_line() -> None:
    samples, duration = make_lap(1, t0=0.0)
    stray_start = Sample(t=-0.01, lap=1, pos=0.999, speed_kmh=150, gear=4, gas=1, brake=0)
    stray_end = Sample(t=duration + 1, lap=1, pos=0.0005, speed_kmh=150, gear=4, gas=1, brake=0)
    trace = LapTrace([stray_start, *samples, stray_end], length_m=LENGTH, lap_time_s=duration)
    assert trace.s[0] < 5 and trace.s[-1] > 990
    assert trace.time_at(500) == pytest.approx(
        LapTrace(samples, length_m=LENGTH, lap_time_s=duration).time_at(500) + 0.01, abs=0.02
    )


def test_lap_trace_needs_samples() -> None:
    with pytest.raises(ValueError):
        LapTrace([], length_m=LENGTH, lap_time_s=1.0)


def test_split_laps_and_best_lap() -> None:
    session = make_session(3)
    laps = split_laps(session)
    assert [lap for lap, _, _ in laps] == [0, 1, 2]
    assert all(ms > 0 for _, _, ms in laps)
    trace, lap_no = best_lap(session, length_m=LENGTH)
    assert lap_no in (1, 2)
    assert trace.lap_time_s > 0
    with pytest.raises(ValueError):
        best_lap(session[:10], length_m=LENGTH)


# -- spoken text and heuristics -------------------------------------------------
def test_say_lap_time_rounds_into_the_next_second() -> None:
    assert say_lap_time(139.969, 139.382) == "2 20 point 0, plus 0.6"
    assert say_lap_time(139.382, 139.382) == "2 19 point 4, plus 0.0"
    assert say_lap_time(138.0, 139.382) == "2 18 point 0, minus 1.4"


def test_heuristic_plan_targets_the_worst_corner() -> None:
    ref = measure_lap(reference().lap, CORNERS)
    samples, duration = make_lap(1, t0=0.0, brake_a=110.0)
    rows = compare(
        measure_lap(LapTrace(samples, length_m=LENGTH, lap_time_s=duration), CORNERS), ref
    )
    plan = heuristic_plan(rows)
    assert plan.calls[0].corner == "Alpha"
    assert "brake" in plan.calls[0].say
    assert heuristic_plan([]).calls == []
    same = compare(ref, ref)
    assert heuristic_plan(same).calls == []


# -- planner --------------------------------------------------------------------
class FakeLLM:
    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((system, user, schema))
        return self.reply


def test_plan_schema_restricts_corners() -> None:
    schema = plan_schema(["Alpha", "Bravo"], 2)
    item = schema["properties"]["calls"]["items"]
    assert item["properties"]["corner"]["enum"] == ["Alpha", "Bravo"]
    assert schema["properties"]["calls"]["maxItems"] == 2


def test_coach_plan_from_json_drops_unknown_corners() -> None:
    plan = CoachPlan.from_json(
        {
            "lap_comment": " Good. ",
            "calls": [
                {"corner": "Alpha", "say": "Brake later", "why": "x"},
                {"corner": "Nowhere", "say": "?", "why": ""},
                {"corner": "Bravo", "say": "  ", "why": ""},
            ],
            "notes": "n",
        },
        valid_corners={"Alpha", "Bravo"},
    )
    assert plan.lap_comment == "Good."
    assert [c.corner for c in plan.calls] == ["Alpha"]


async def test_planner_sends_table_and_keeps_history(tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("Stay left before Alpha.", encoding="utf-8")
    ref = reference()
    ref_metrics = measure_lap(ref.lap, CORNERS)
    system = build_system_prompt(
        track_name="Test Ring",
        car="car",
        reference_time_s=ref.lap_time_s,
        reference_metrics=ref_metrics,
        context_files=[notes, tmp_path / "missing.md"],
        max_calls=2,
    )
    assert "Stay left before Alpha." in system and "Alpha |" in system
    llm = FakeLLM(
        {
            "lap_comment": "Brake later into Alpha.",
            "calls": [{"corner": "Alpha", "say": "Alpha: brake later", "why": "early"}],
            "notes": "",
        }
    )
    planner = CoachPlanner(
        llm, system_prompt=system, corner_names=["Alpha", "Bravo"], reference_time_s=ref.lap_time_s
    )
    rows = compare(ref_metrics, ref_metrics)
    plan = await planner.plan(
        lap_no=3, lap_time_s=ref.lap_time_s + 1, rows=rows, flags=["test flag"]
    )
    assert plan.calls[0].say == "Alpha: brake later"
    _, user, _ = llm.calls[0]
    assert "Lap 3" in user and "test flag" in user and "Alpha |" in user
    await planner.plan(lap_no=4, lap_time_s=ref.lap_time_s, rows=rows)
    assert "lap 3:" in llm.calls[1][1]
    assert len(planner.history) == 2


# -- live coach -------------------------------------------------------------------
async def test_live_coach_speaks_lap_time_comment_and_calls(tmp_path: Path) -> None:
    speaker = LogSpeaker()
    log = tmp_path / "log.jsonl"

    async def plan_fn(lap_no: int, lap_s: float, rows: Any, flags: list[str]) -> CoachPlan:
        return CoachPlan(
            lap_comment="Work on Bravo.",
            calls=[CornerCall(corner="Bravo", say="Bravo: brake later")],
        )

    coach = LiveCoach(
        corners=CORNERS,
        reference=reference(),
        speaker=speaker,
        plan_fn=plan_fn,
        call_lead_s=2.0,
        log_path=log,
    )
    for sample in make_session(3):
        lap_changed = coach._samples and sample.lap != coach._samples[-1].lap
        await coach.feed(sample)
        if lap_changed:
            await coach.drain()
    await coach.drain()
    spoken = speaker.spoken
    assert sum("point" in s for s in spoken) == 3  # three lap times
    assert spoken.count("Bravo: brake later") == 2  # once on each following lap
    assert "Work on Bravo." in spoken
    events = [json.loads(line)["event"] for line in log.read_text().splitlines()]
    assert {"lap", "plan", "call", "comment"} <= set(events)


async def test_live_coach_falls_back_when_planner_fails() -> None:
    speaker = LogSpeaker()

    async def broken(*_: Any) -> CoachPlan:
        raise RuntimeError("network down")

    coach = LiveCoach(corners=CORNERS, reference=reference(), speaker=speaker, plan_fn=broken)
    for sample in make_session(2, brake_a=100.0):
        await coach.feed(sample)
    await coach.drain()
    assert coach.plans and coach.plans[-1].calls[0].corner == "Alpha"


async def test_live_coach_skips_pit_laps() -> None:
    speaker = LogSpeaker()
    coach = LiveCoach(corners=CORNERS, reference=reference(), speaker=speaker)
    samples, duration = make_lap(0, t0=0.0)
    pit = [replace(s, in_pit=True) for s in samples]
    for s in pit:
        await coach.feed(s)
    await coach.feed(
        Sample(
            t=duration,
            lap=1,
            pos=0.0,
            speed_kmh=100,
            gear=3,
            gas=1,
            brake=0,
            last_ms=int(duration * 1000),
        )
    )
    await coach.drain()
    assert coach.plans == []
    assert len(speaker.spoken) == 1  # just the lap time


# -- Claude client ------------------------------------------------------------------
class _FakeMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _fake_client(text: str, stop: str = "end_turn") -> tuple[Any, _FakeMessages]:
    response = SimpleNamespace(
        stop_reason=stop,
        stop_details=None,
        model="claude-opus-5-5",
        usage=SimpleNamespace(
            input_tokens=10,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=5,
            output_tokens=3,
        ),
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
    )
    messages = _FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


async def test_anthropic_client_json_request_shape() -> None:
    fake, messages = _fake_client('{"ok": true}')
    client = AnthropicClient(model="claude-opus-5-5", effort="low", speed="fast", client=fake)
    data = await client.complete_json("sys", "user", {"type": "object"})
    assert data == {"ok": True}
    kw = messages.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["speed"] == "fast"
    assert "thinking" not in kw


async def test_anthropic_client_text_and_refusal() -> None:
    fake, messages = _fake_client("hello")
    client = AnthropicClient(model="m", client=fake)
    assert await client.complete("s", "u") == "hello"
    assert "format" not in messages.kwargs["output_config"]
    assert "speed" not in messages.kwargs
    refusing, _ = _fake_client("", stop="refusal")
    with pytest.raises(ClaudeRefusalError):
        await AnthropicClient(model="m", client=refusing).complete("s", "u")


# -- references, replay, debrief ----------------------------------------------------
def _write_csv(path: Path, samples: list[Sample]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(s.row() for s in samples)
    return path


def test_reference_roundtrip_and_make_reference(tmp_path: Path) -> None:
    ref = reference()
    path = save_reference(ref, reference_path(tmp_path, "t", "l", "c"))
    loaded = load_reference(path)
    assert loaded.lap_time_s == pytest.approx(ref.lap_time_s)
    assert loaded.lap.speed_at(500) == pytest.approx(ref.lap.speed_at(500), abs=0.1)
    csv_path = _write_csv(tmp_path / "rec.csv", make_session(3))
    assert len(list(read_samples_csv(csv_path))) > 100
    out = make_reference(
        csv_path,
        track="t",
        layout="l",
        car="c",
        length_m=LENGTH,
        out_dir=tmp_path / "refs",
        source="test",
    )
    assert load_reference(out).source == "test"


async def test_replay_and_debrief_message(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "session.csv", make_session(3, brake_a=120.0))
    speaker = LogSpeaker()
    llm = FakeLLM(
        {
            "lap_comment": "Alpha next.",
            "calls": [{"corner": "Alpha", "say": "Alpha: later", "why": ""}],
            "notes": "",
        }
    )
    coach = build_coach(
        CoachSettings(context_files=[]),
        track=TRACK,
        reference=reference(),
        speaker=speaker,
        use_llm=True,
        log_path=None,
        llm=llm,
    )
    await replay(coach, csv_path)
    assert "Alpha: later" in speaker.spoken
    assert len(llm.calls) == 3
    laps = session_laps(csv_path, reference(), TRACK)
    assert len(laps) == 3
    text = debrief_message(laps, reference(), ["- after lap 1: Alpha next."])
    assert "Best lap" in text and "Alpha:" in text and "after lap 1" in text


def test_build_coach_offline_and_speaker_factory() -> None:
    coach = build_coach(
        CoachSettings(context_files=[]),
        track=TRACK,
        reference=reference(),
        speaker=LogSpeaker(),
        use_llm=False,
        log_path=None,
    )
    assert coach.plan_fn is None
    assert isinstance(make_speaker(enabled=False), LogSpeaker)


def test_coach_settings_defaults() -> None:
    settings = CoachSettings()
    assert settings.model == "claude-opus-5-5"
    assert settings.lap_effort == "low"
    assert settings.max_calls_per_lap == 2


# -- usage and cost accounting ---------------------------------------------------------
def test_estimate_usage_prices_opus_tokens() -> None:
    from src.ai.anthropic_client import estimate_usage

    usage = SimpleNamespace(
        input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    est = estimate_usage("claude-opus-5-5", usage)
    assert est.cost_usd == pytest.approx(4.0 + 0.20 + 5.0 + 20.0)
    assert estimate_usage("claude-opus-5-5", usage, fast=True).cost_usd == pytest.approx(58.4)
    assert estimate_usage("some-unknown-model", usage).cost_usd == 0.0
    assert estimate_usage("claude-opus-5-5", None).calls == 1


async def test_client_tracks_usage_and_planner_attaches_it() -> None:
    fake, _ = _fake_client('{"lap_comment": "ok", "calls": [], "notes": ""}')
    client = AnthropicClient(model="claude-opus-5-5", client=fake)
    planner = CoachPlanner(
        client, system_prompt="s", corner_names=["Alpha"], reference_time_s=100.0
    )
    plan = await planner.plan(lap_no=1, lap_time_s=101.0, rows=[])
    await planner.plan(lap_no=2, lap_time_s=100.5, rows=[])
    assert plan.usage is not None and plan.usage["calls"] == 1
    assert plan.usage["cache_write_tokens"] == 5
    assert client.total.calls == 2
    assert client.total.cost_usd == pytest.approx(2 * (10 * 4 + 3 * 20 + 5 * 5) / 1e6)


async def test_live_coach_sums_usage(tmp_path: Path) -> None:
    async def plan_fn(*_: Any) -> CoachPlan:
        return CoachPlan(
            lap_comment="",
            usage={
                "calls": 1,
                "input_tokens": 100,
                "cache_read_tokens": 4000,
                "cache_write_tokens": 0,
                "output_tokens": 300,
                "cost_usd": 0.0078,
            },
        )

    log = tmp_path / "log.jsonl"
    coach = LiveCoach(
        corners=CORNERS, reference=reference(), speaker=LogSpeaker(), plan_fn=plan_fn, log_path=log
    )
    assert coach.usage_summary() == "Claude calls: 0"
    for sample in make_session(3):
        await coach.feed(sample)
    await coach.drain()
    coach.finish()
    assert coach.usage["calls"] == 3
    assert "per lap" in coach.usage_summary() and "$0.0078 per lap" in coach.usage_summary()
    assert '"event": "session-usage"' in log.read_text()
