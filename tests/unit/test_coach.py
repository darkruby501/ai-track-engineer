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
from src.coach.coaches import CoachContext, CornerAdvisor, HeadCoach, pass_message
from src.coach.live import LiveCoach, say_lap_time
from src.coach.metrics import compare, comparison_table, measure_lap, reference_table
from src.coach.pacenotes import CornerNote, PaceNotes, PassRecord
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
    prepare_notes,
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
        # Steering: Alpha is a right-hander, Bravo a left-hander (turn-in after braking).
        steer = 0.2 if 200 <= s < 310 else (-0.2 if 600 <= s < 710 else 0.0)
        samples.append(
            Sample(
                t=t,
                lap=lap,
                pos=s / LENGTH,
                speed_kmh=v,
                gear=4,
                gas=gas,
                brake=brake,
                steer=steer,
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
    assert "thinking" not in kw
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
async def test_haiku_requests_skip_fallbacks_and_thinking() -> None:
    fake, messages = _fake_client('{"ok": 1}')
    client = AnthropicClient(model="claude-haiku-5-5", client=fake, thinking_off=True)
    await client.complete_json("s", "u", {"type": "object"})
    kw = messages.kwargs
    assert "fallbacks" not in kw and "betas" not in kw
    assert kw["thinking"] == {"type": "disabled"}
    opus = AnthropicClient(model="claude-opus-5-5", client=fake, thinking_off=True)
    assert opus.thinking_off is False


async def test_client_tracks_usage() -> None:
    fake, _ = _fake_client('{"x": 1}')
    client = AnthropicClient(model="claude-opus-5-5", client=fake)
    await client.complete_json("s", "u", {"type": "object"})
    await client.complete_json("s", "u", {"type": "object"})
    assert client.last is not None and client.last.calls == 1
    assert client.total.calls == 2
    assert client.total.cost_usd == pytest.approx(2 * (10 * 4 + 3 * 20 + 5 * 5) / 1e6)


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


# -- fakes for the Claude roles ------------------------------------------------------
class FakeJSON:
    """Stands in for a Claude client: returns canned JSON per role."""

    def __init__(self, reply: dict[str, Any] | Exception) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.model = "fake"

    async def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((system, user, schema))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


ADVICE = {
    "now": "Alpha: better.",
    "say": "Alpha: brake at the mark",
    "apex_say": "Kerb",
    "call_turn": True,
    "call_throttle": False,
}
PLAN = {
    "lap_comment": "Good lap. Work on Alpha.",
    "notes": "n",
    "focus": [
        {"corner": "Alpha", "objective": "Brake at the mark"},
        {"corner": "Nowhere", "objective": "ignored"},
    ],
}


def _context(tmp_path: Path | None = None) -> CoachContext:
    ref = reference()
    files = []
    if tmp_path is not None:
        brief = tmp_path / "brief.md"
        brief.write_text("Alpha: use the kerb.", encoding="utf-8")
        files = [brief, tmp_path / "missing.md"]
    return CoachContext(
        "Test Ring",
        "car",
        ref.lap_time_s,
        measure_lap(ref.lap, CORNERS),
        {"Alpha": 125.0},
        {"Alpha": 76.0},
        files,
    )


def _coach(*, advisor: Any = None, head: Any = None, log: Path | None = None) -> LiveCoach:
    ref = reference()
    pace = PaceNotes(ref_metrics=measure_lap(ref.lap, CORNERS), ref_lap=ref.lap)
    ctx = _context()
    return LiveCoach(
        corners=CORNERS,
        reference=ref,
        speaker=LogSpeaker(),
        pace=pace,
        advisor=CornerAdvisor(advisor, ctx) if advisor else None,
        head=HeadCoach(head, ctx, ["Alpha", "Bravo"]) if head else None,
        log_path=log,
    )


async def _drive(coach: LiveCoach, samples: list[Sample]) -> None:
    for sample in samples:
        await coach.feed(sample)
        if coach.pending():
            await coach.drain()
    await coach.drain()


def test_context_text_includes_files_and_targets(tmp_path: Path) -> None:
    text = _context(tmp_path).text()
    assert "Alpha: use the kerb." in text and "braking mark" in text and "Alpha |" in text


async def test_live_coach_cues_verdicts_and_focus(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    fast, head = FakeJSON(ADVICE), FakeJSON(PLAN)
    coach = _coach(advisor=fast, head=head, log=log)
    await _drive(coach, make_session(3, brake_a=120.0))
    spoken = coach.speaker.spoken
    assert "Brake" in spoken
    assert any(s.startswith("Alpha") for s in spoken)  # approach note
    assert "Alpha: better." in spoken  # verdict from the fast model
    assert sum("point" in s for s in spoken) >= 2  # lap times
    assert "Good lap. Work on Alpha." in spoken
    assert coach.pace.notes["Alpha"].say == "Alpha: brake at the mark"
    assert coach.pace.notes["Alpha"].objective == "Brake at the mark"
    assert coach.pace.notes["Bravo"].objective == ""
    assert "Kerb" in spoken and "Turn in" in spoken  # cues switched on by the advisor
    assert "Corner: Alpha" in fast.calls[0][1] and "mark" in fast.calls[0][1]
    assert "Lap 1" in head.calls[0][1]
    events = {json.loads(line)["event"] for line in log.read_text().splitlines()}
    assert {"cue", "pass", "advice", "lap", "plan", "said"} <= events


async def test_live_coach_without_models_uses_quick_verdicts() -> None:
    coach = _coach()
    await _drive(coach, make_session(3, brake_a=100.0))
    assert any("braked early" in s for s in coach.speaker.spoken)


async def test_advisor_failure_falls_back_and_keeps_notes() -> None:
    coach = _coach(advisor=FakeJSON(RuntimeError("down")), head=FakeJSON(RuntimeError("down")))
    before = coach.pace.notes["Alpha"].say
    await _drive(coach, make_session(2, brake_a=100.0))
    assert coach.pace.notes["Alpha"].say == before
    assert any("braked early" in s for s in coach.speaker.spoken)
    assert coach.plans == []


async def test_pit_laps_get_no_head_plan() -> None:
    head = FakeJSON(PLAN)
    coach = _coach(head=head)
    samples, duration = make_lap(0, t0=0.0)
    for s in samples:
        await coach.feed(replace(s, in_pit=True))
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
    assert head.calls == []


async def test_initial_notes_writer_and_prepare(tmp_path: Path) -> None:
    from src.coach.coaches import write_initial_notes

    reply = {
        "notes": [
            {
                "corner": "Alpha",
                "say": "Alpha: left side",
                "apex_say": "Kerb",
                "call_turn": False,
                "call_throttle": True,
            },
            {
                "corner": "Nowhere",
                "say": "x",
                "apex_say": "",
                "call_turn": False,
                "call_throttle": False,
            },
        ]
    }
    coach = _coach()
    notes = await write_initial_notes(FakeJSON(reply), _context(), coach.pace.notes)
    assert notes["Alpha"].say == "Alpha: left side" and notes["Alpha"].call_throttle
    assert "Nowhere" not in notes

    settings = CoachSettings(context_files=[], notes_dir=str(tmp_path))
    built = build_coach(
        settings,
        track=TRACK,
        reference=reference(),
        speaker=LogSpeaker(),
        use_llm=True,
        log_path=None,
        head_llm=FakeJSON(reply),
        fast_llm=FakeJSON(ADVICE),
    )
    assert await prepare_notes(built, settings) is True
    assert built.pace.notes["Alpha"].apex_say == "Kerb"
    again = build_coach(
        settings,
        track=TRACK,
        reference=reference(),
        speaker=LogSpeaker(),
        use_llm=True,
        log_path=None,
        head_llm=FakeJSON(reply),
        fast_llm=FakeJSON(ADVICE),
    )
    assert again.notes_from_claude and await prepare_notes(again, settings) is False
    assert again.pace.notes["Alpha"].say == "Alpha: left side"


def test_build_coach_uses_target_marks(tmp_path: Path) -> None:
    stem = "test__-__car"
    (tmp_path / f"{stem}__target.json").write_text(
        json.dumps({"marks": {"Alpha": 140.0}, "corner_min_kmh": {"Alpha": 75.0}}), encoding="utf-8"
    )
    ref = reference()
    ref = Reference("test", "-", "car", "synthetic", ref.lap)
    settings = CoachSettings(context_files=[], notes_dir=str(tmp_path))
    coach = build_coach(
        settings, track=TRACK, reference=ref, speaker=LogSpeaker(), use_llm=False, log_path=None
    )
    assert coach.pace.marks["Alpha"] == 140.0
    assert coach.target_min_kmh == {"Alpha": 75.0}
    assert coach.advisor is None and coach.usage_summary() == "Claude calls: 0"
    assert isinstance(make_speaker(enabled=False), LogSpeaker)


async def test_replay_and_debrief_message(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "session.csv", make_session(3, brake_a=120.0))
    settings = CoachSettings(context_files=[], notes_dir=str(tmp_path))
    fast = FakeJSON(ADVICE)
    coach = build_coach(
        settings,
        track=TRACK,
        reference=reference(),
        speaker=LogSpeaker(),
        use_llm=True,
        log_path=None,
        head_llm=FakeJSON(PLAN),
        fast_llm=fast,
    )
    await replay(coach, csv_path)
    assert "Alpha: better." in coach.speaker.spoken and len(fast.calls) >= 4
    laps = session_laps(csv_path, reference(), TRACK)
    assert len(laps) == 3
    text = debrief_message(laps, reference(), ["- after lap 1: Alpha next."])
    assert "Best lap" in text and "Alpha:" in text and "after lap 1" in text
    coach.finish()


def test_pass_message_mentions_history_and_objective() -> None:
    ref_metrics = measure_lap(reference().lap, CORNERS)
    row = compare(ref_metrics, ref_metrics)[0]
    note = CornerNote("Alpha", say="Alpha: left", objective="Brake later")
    msg = pass_message(
        row,
        lap=4,
        mark=125.0,
        target_min=76.0,
        history=[PassRecord(3, row, 125.0, "Same.")],
        note=note,
    )
    assert "Lap 4" in msg and "mark @125" in msg and "Brake later" in msg and "Same." in msg


def test_coach_settings_defaults() -> None:
    settings = CoachSettings()
    assert settings.model == "claude-opus-5-5"
    assert settings.fast_model == "claude-haiku-5-5"
    assert settings.lap_effort == "low"
    assert settings.pace_notes is True


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


def test_speakable_rejects_distances_and_numbers() -> None:
    from src.coach.coaches import speakable

    assert speakable("Phil Hill: better, braked on the mark.")
    assert speakable("Sunrise: a tenth quicker, 0.1 up.")
    for bad in ["braked 29 metres late", "mark is 3124", "0.03 faster", "brake 30 m earlier"]:
        assert not speakable(bad), bad


async def test_advisor_filters_unspeakable_text() -> None:
    reply = {
        "now": "Alpha: braked 29 metres late",
        "say": "Brake at 3124",
        "apex_say": "Kerb",
        "call_turn": False,
        "call_throttle": False,
    }
    advisor = CornerAdvisor(FakeJSON(reply), _context())
    note = CornerNote("Alpha", say="Alpha: stay left")
    advice = await advisor.advise("msg", note)
    assert advice.now == "" and advice.rejected
    assert advice.note.say == "Alpha: stay left" and advice.note.apex_say == "Kerb"


# -- ElevenLabs speaker ----------------------------------------------------------------
class _FakePlayer:
    def __init__(self) -> None:
        self.played: list[Path] = []
        self.stops = 0

    def play(self, path: Path) -> None:
        self.played.append(path)

    def stop(self) -> None:
        self.stops += 1


def _eleven(
    tmp_path: Path, *, fail: bool = False
) -> tuple[Any, list[dict[str, Any]], _FakePlayer, LogSpeaker]:
    from src.coach.speaker import ElevenLabsSpeaker

    posts: list[dict[str, Any]] = []

    def post(url: str, headers: dict[str, str], body: dict[str, Any]) -> bytes:
        if fail:
            raise RuntimeError("network")
        posts.append({"url": url, "body": body, "headers": headers})
        return b"\x00\x00" * 2205  # 0.1 s of silence

    def get(url: str, headers: dict[str, str]) -> dict[str, Any]:
        return {
            "voices": [
                {"name": "Rachel", "voice_id": "r1"},
                {"name": "Brian - deep", "voice_id": "b1"},
            ]
        }

    player, fallback = _FakePlayer(), LogSpeaker()
    speaker = ElevenLabsSpeaker(
        "key",
        cache_dir=tmp_path,
        fallback=fallback,
        player=player,
        http_post=post,
        http_get=get,
        start_threads=False,
    )
    return speaker, posts, player, fallback


def test_elevenlabs_caches_audio_and_picks_a_voice(tmp_path: Path) -> None:
    from src.coach.speaker import wav_seconds

    speaker, posts, player, _ = _eleven(tmp_path)
    speaker.prefetch(["Brake", "Brake", ""])
    assert speaker.prefetch_next() and not speaker.prefetch_next()
    assert len(posts) == 1 and "/text-to-speech/b1?" in posts[0]["url"]
    assert posts[0]["body"]["model_id"] == "eleven_flash_v2_5"
    assert posts[0]["headers"] == {"xi-api-key": "key"}
    speaker.say("Brake")
    assert speaker.play_next() and len(posts) == 1  # served from the cache
    assert wav_seconds(player.played[0].read_bytes()) == pytest.approx(0.1)
    assert speaker.cached("Brake") and not speaker.cached("Throttle")


def test_elevenlabs_urgent_interrupts_and_queues(tmp_path: Path) -> None:
    speaker, _, player, _ = _eleven(tmp_path)
    speaker.say("first")
    speaker.say("second")
    assert speaker.play_next()
    assert not speaker.play_next()  # still "playing" the first for 0.1 s
    speaker.say("Brake", urgent=True)  # drops "second", stops playback
    assert player.stops == 1
    assert speaker.play_next()
    assert player.played[-1].name == speaker._cache_path("Brake").name
    assert speaker.spoken == ["first", "second", "Brake"]


def test_elevenlabs_falls_back_to_system_voice(tmp_path: Path) -> None:
    speaker, _, player, fallback = _eleven(tmp_path, fail=True)
    speaker.say("Brake", urgent=True)
    assert speaker.play_next()
    assert fallback.spoken == ["Brake"] and speaker.failures == 1 and player.played == []
    speaker.prefetch(["Throttle"])
    assert speaker.prefetch_next() and speaker.failures == 2
    speaker.close()
