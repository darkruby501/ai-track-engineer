"""Tests for the conservative target line and the pace-notes model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.coach.metrics import compare, measure_lap
from src.coach.pacenotes import (
    CornerNote,
    PaceNotes,
    PassRecord,
    estimate_speech_s,
    quick_verdict,
    template_note,
    time_words,
)
from src.coach.target_line import (
    BRAKE,
    SIDE_L,
    SIDE_R,
    AiSpline,
    CarModel,
    build_conservative_target,
    margin_offsets,
    monotone_envelope,
    orient_normals,
    pedal_flags,
    pedal_flags_from_reference,
    profile_from_reference,
    read_ai_spline,
    speed_profile,
    to_ideal_line,
    turning_points,
    write_ai_spline,
)

from tests.unit.test_coach import CORNERS, LENGTH, make_lap, reference


def _oval(n: int = 400, radius: float = 60.0, straight: float = 300.0) -> AiSpline:
    """A clockwise-agnostic stadium: two straights joined by half circles."""
    pts = []
    per = 2 * straight + 2 * np.pi * radius
    for i in range(n):
        d = per * i / n
        if d < straight:
            x, z = d, 0.0
        elif d < straight + np.pi * radius:
            a = (d - straight) / radius
            x, z = straight + radius * np.sin(a), radius - radius * np.cos(a)
        elif d < 2 * straight + np.pi * radius:
            x, z = straight - (d - straight - np.pi * radius), 2 * radius
        else:
            a = (d - 2 * straight - np.pi * radius) / radius
            x, z = -radius * np.sin(a), radius + radius * np.cos(a)
        pts.append((x, 0.0, z, d))
    points = np.array(pts)
    extra = np.zeros((n, 18))
    extra[:, SIDE_L] = 5.0
    extra[:, SIDE_R] = 5.0
    return AiSpline((7, n, 0, 0), points, np.arange(n, dtype=np.int32), extra, b"grid")


def _right_edge(spline: AiSpline, width: float = 5.0) -> np.ndarray:
    xz = spline.xz
    fwd = np.roll(xz, -1, axis=0) - np.roll(xz, 1, axis=0)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    right = np.c_[-fwd[:, 1], fwd[:, 0]] * -1  # either sign; orient_normals must cope
    return xz + right * width


def test_ai_spline_roundtrip(tmp_path: Path) -> None:
    spline = _oval()
    path = tmp_path / "line.ai"
    write_ai_spline(spline, path)
    back = read_ai_spline(path)
    assert back.header == spline.header and back.tail == b"grid"
    assert np.allclose(back.points, spline.points, atol=1e-3)
    assert np.allclose(back.extra, spline.extra, atol=1e-5)


def test_margin_offsets_only_move_where_needed() -> None:
    spline = _oval()
    spline.extra[100:120, SIDE_L] = 1.0  # line hugs the left edge here
    margins = np.full(spline.n, 0.3)
    off = margin_offsets(spline, half_width_m=0.84, wheel_margin_m=margins)
    assert np.all(off[100:120] >= 0.84 + 0.3 - 1.0 - 1e-9)  # pushed right enough
    assert np.allclose(off[:80], 0.0) and np.allclose(off[200:], 0.0)


def test_orient_normals_points_right_whatever_the_edge_sign() -> None:
    spline = _oval()
    edge = _right_edge(spline)
    normals = orient_normals(spline, edge)
    toward = edge - spline.xz
    assert np.median(np.einsum("ij,ij->i", normals, toward)) > 0


def test_speed_profiles_respect_limits() -> None:
    spline = _oval()
    car = CarModel(lat_g=0.9, brake_g=0.95)
    s, v, kappa = speed_profile(spline.xz, car)
    corner = np.abs(kappa) > 1 / 70
    assert v[corner].max() == pytest.approx(np.sqrt(0.9 * 9.81 * 60), rel=0.08)
    s_m = spline.points[:, 3]
    ref = _ref_with_corners(s_m) / 3.6
    vp = profile_from_reference(s_m, ref, spline.xz, car)
    in_a = (s_m >= 330) & (s_m <= 360)
    assert vp[in_a].max() <= 60.0 / 3.6 * 0.95 + 1e-6
    assert np.all(vp <= ref + 1e-6)  # never faster than the reference
    seg = np.hypot(*(np.roll(spline.xz, -1, axis=0) - spline.xz).T)
    decel = (vp**2 - np.roll(vp, -1) ** 2) / (2 * seg)
    assert decel.max() <= 0.95 * 9.81 + 1e-3


def _ref_with_corners(s_m: np.ndarray, wobble: float = 0.0) -> np.ndarray:
    """Reference speed (km/h): 120 on straights, 60 through each half circle."""
    per = 2 * 300 + 2 * np.pi * 60
    corner = ((s_m > 300) & (s_m < 300 + np.pi * 60)) | (s_m > per - np.pi * 60)
    base = np.where(corner, 60.0, 120.0)
    k = 15
    smooth = np.convolve(np.r_[base[-k:], base, base[:k]], np.ones(k) / k, "same")[k:-k]
    return smooth + wobble * np.sin(s_m / 4.0)


def test_monotone_envelope_flattens_wobbles() -> None:
    s_m = _oval().points[:, 3]
    ref = _ref_with_corners(s_m, wobble=1.0) / 3.6  # +-1 km/h noise
    cap = monotone_envelope(ref, 0.95)
    pts = turning_points(ref, 5.0 / 3.6)
    assert len(pts) == 4  # peak, trough, peak, trough: the noise is ignored
    a, b = pts[0], pts[1]
    assert np.all(np.diff(cap[a : b + 1]) <= 1e-9)  # only falls into the corner
    assert np.all(cap <= ref + 1e-9)


def test_pedal_flags_one_brake_zone_then_coast() -> None:
    s = np.arange(0.0, 600.0, 1.5)
    v = np.full(s.size, 40.0)
    brake = (s >= 100) & (s < 160)
    v[brake] = 40.0 - (s[brake] - 100) * 0.3  # hard braking 40 -> 22 m/s
    v[(s >= 160) & (s < 200)] = 22.0 - (s[(s >= 160) & (s < 200)] - 160) * 0.02  # roll in
    v[s >= 200] = np.minimum(21.2 + (s[s >= 200] - 200) * 0.1, 40.0)
    lift = (s >= 400) & (s < 420)
    v[lift] -= np.linspace(0, 1.0, np.count_nonzero(lift))  # a 3.6 km/h lift
    gas, br = pedal_flags(s, v)
    red = br > 0.5
    starts = np.flatnonzero(red & ~np.roll(red, 1))
    assert len(starts) == 1 and s[starts[0]] == pytest.approx(100.0, abs=3)
    coast = (gas < 0.5) & ~red
    assert coast[(s > 165) & (s < 195)].all()  # white into the apex
    assert not red[(s > 390) & (s < 430)].any()  # a small lift is never red
    assert gas[(s > 230) & (s < 390)].all()


def test_pedal_flags_follow_the_reference_pedals() -> None:
    s = np.arange(0.0, 600.0, 1.5)
    v = np.full(s.size, 40.0)
    zone = (s >= 100) & (s < 200)
    v[zone] = 40.0 - (s[zone] - 100) * 0.18  # our braking starts at 100 m
    v[s >= 200] = np.minimum(22.0 + (s[s >= 200] - 200) * 0.1, 40.0)
    ai_gas = np.where((s < 115) | (s > 156), 1.0, 0.0)
    ai_gas[(s > 300) & (s < 340)] = 0.0  # a real lift
    ai_brake = np.where((s >= 115) & (s < 150), 0.8, 0.0)  # the AI brakes later
    gas, brake = pedal_flags_from_reference(s, v, ai_gas, ai_brake)
    red = brake > 0.5
    assert red[(s >= 101) & (s < 149)].all()  # from our mark to the AI's release
    assert gas[(s > 160) & (s < 295)].all()  # throttle from turn-in, speed still falling
    assert not (gas[(s > 305) & (s < 335)] > 0.5).any()  # white lift
    assert not red[s > 160].any()


def test_build_target_and_ideal_line() -> None:
    spline = _oval()
    edge = _right_edge(spline)
    s_m = spline.points[:, 3]
    target = build_conservative_target(
        spline,
        edge,
        ref_s=s_m,
        ref_speed_kmh=_ref_with_corners(s_m),
        car=CarModel(),
        risk_zones=[(0.0, 50.0, 1.0)],
    )
    assert target.lap_time_s > 0
    spans = [("A", 300.0, 300 + np.pi * 60), ("B", 900.0, 900 + np.pi * 60)]
    marks = target.brake_points(spans)
    assert set(marks) == {"A", "B"} and marks["A"] < 300.0
    ideal = to_ideal_line(spline, target)
    assert ideal.n == spline.n and ideal.extra[:, 0].max() > 0
    red = ideal.extra[:, BRAKE] > 0.5
    assert np.count_nonzero(red & ~np.roll(red, 1)) == 2  # one brake zone per corner
    assert ideal.extra[:, BRAKE][red].min() == 1.0
    assert marks["A"] == pytest.approx(s_m[np.flatnonzero(red & ~np.roll(red, 1))[0]])


# -- pace notes -----------------------------------------------------------------
def _pace() -> PaceNotes:
    ref = reference()
    return PaceNotes(ref_metrics=measure_lap(ref.lap, CORNERS), ref_lap=ref.lap)


def test_template_notes_and_cues_are_ordered() -> None:
    pace = _pace()
    assert set(pace.notes) == {"Alpha", "Bravo"}
    assert pace.notes["Alpha"].say.startswith("Alpha")
    pace.notes["Alpha"] = CornerNote(
        "Alpha", say="Alpha, stay left", apex_say="Kerb", call_turn=True, call_throttle=True
    )
    cues = pace.cues()
    kinds = [c.kind for c in cues if c.corner == "Alpha"]
    # Alpha is the first corner: its approach sentence wraps to the end of the
    # previous lap; within the lap the cue words come in driving order. The
    # apex note would run into "Throttle", so "Throttle" gives way.
    assert kinds == ["brake", "turn", "apex", "approach"]
    assert next(c for c in cues if c.kind == "approach" and c.corner == "Alpha").at_m > 900
    assert [c.at_m for c in cues] == sorted(c.at_m for c in cues)
    brake = next(c for c in cues if c.corner == "Alpha" and c.kind == "brake")
    assert brake.text == "Brake" and brake.urgent
    assert brake.at_m < pace.marks["Alpha"]
    only = pace.cues(only={"Bravo"})
    assert {c.corner for c in only} == {"Bravo"}


def _assert_no_overlaps(pace: PaceNotes, cues: list) -> None:
    times = [pace._time(c.at_m) for c in cues]
    for i, c in enumerate(cues):
        nxt = times[(i + 1) % len(cues)] + (pace.lap_s if i == len(cues) - 1 else 0.0)
        assert times[i] + pace.speech_s(c.text) + pace.gap_s <= nxt + 1e-6, c


def test_cues_never_cut_each_other_off() -> None:
    pace = _pace()
    long_note = "Brake before the left kerb, hold right, then squeeze the throttle"
    pace.notes["Alpha"] = CornerNote(
        "Alpha",
        say=long_note,
        apex_say="Late apex, smooth hands, patient",
        call_turn=True,
        call_throttle=True,
    )
    pace.notes["Bravo"] = CornerNote(
        "Bravo", say="Bravo, third", apex_say="Kerb", call_turn=True, call_throttle=True
    )
    cues = pace.cues()
    _assert_no_overlaps(pace, cues)
    kinds = {(c.corner, c.kind) for c in cues}
    # "Brake" and "Turn in" are never given up for a note
    assert {("Alpha", "brake"), ("Alpha", "turn"), ("Bravo", "brake"), ("Bravo", "turn")} <= kinds
    approach = next(c for c in cues if c.corner == "Alpha" and c.kind == "approach")
    brake = next(c for c in cues if c.corner == "Alpha" and c.kind == "brake")
    # the long approach note moved earlier so it ends before "Brake"
    start, end = pace._time(approach.at_m), pace._time(brake.at_m)
    assert (end - start) % pace.lap_s >= pace.speech_s(long_note) + pace.gap_s - 1e-6
    # measured durations replace the estimate
    pace.speech_s = lambda text: 0.5
    assert len(pace.cues()) >= len(cues)


def test_estimate_speech_s_grows_with_words() -> None:
    short, long = estimate_speech_s("Brake"), estimate_speech_s("Brake at the kink, 2nd gear")
    assert 0.6 < short < 1.0 < long < 3.0


def test_marks_move_within_bounds_and_state_roundtrips(tmp_path: Path) -> None:
    pace = _pace()
    ref_brake = pace.ref("Alpha").brake_m  # type: ignore[union-attr]
    start = pace.marks["Alpha"]
    assert pace.move_mark("Alpha", 1000) == pytest.approx(min(start + 25, ref_brake + 5))
    assert pace.move_mark("Nowhere", 10) is None
    path = tmp_path / "state.json"
    pace.notes["Bravo"].apex_say = "Off the kerb"
    pace.save(path)
    fresh = _pace()
    fresh.apply_state(PaceNotes.load_state(path))
    assert fresh.marks["Alpha"] == pytest.approx(pace.marks["Alpha"], abs=0.1)
    assert fresh.notes["Bravo"].apex_say == "Off the kerb"
    assert PaceNotes.load_state(tmp_path / "missing.json") == {}


def test_quick_verdict() -> None:
    ref = reference()
    ref_metrics = measure_lap(ref.lap, CORNERS)
    early, d1 = make_lap(1, t0=0.0, brake_a=110.0)
    from src.coach.samples import LapTrace

    slow = compare(
        measure_lap(LapTrace(early, length_m=LENGTH, lap_time_s=d1), CORNERS), ref_metrics
    )[0]
    best = compare(ref_metrics, ref_metrics)[0]
    mark = 150.0
    assert quick_verdict(slow, None, mark) == "In Alpha, you braked early."
    previous = PassRecord(lap=1, row=slow, mark_m=mark)
    verdict = quick_verdict(best, previous, mark)
    assert verdict.startswith("Alpha was") and "quicker than last time" in verdict
    assert verdict.endswith("and you braked right on the mark.")
    assert [time_words(x) for x in (0.08, -0.3, 0.5, 1.24, 1.6, 4.4, 143.7)] == [
        "a tenth",
        "three tenths",
        "half a second",
        "a second",
        "a second and a half",
        "about four seconds",
        "about 144 seconds",
    ]
    assert quick_verdict(best, PassRecord(1, best, None), None) == ""
    from src.coach.pacenotes import spoken_name

    assert spoken_name("Cotton Corners (right)") == "Cotton right"
    assert spoken_name("Phil Hill") == "Phil Hill"
    assert template_note(ref_metrics[0]).call_brake
