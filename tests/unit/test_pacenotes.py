"""Tests for the conservative target line and the pace-notes model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.coach.metrics import compare, measure_lap
from src.coach.pacenotes import CornerNote, PaceNotes, PassRecord, quick_verdict, template_note
from src.coach.target_line import (
    SIDE_L,
    SIDE_R,
    AiSpline,
    CarModel,
    build_conservative_target,
    margin_offsets,
    orient_normals,
    profile_from_reference,
    read_ai_spline,
    speed_profile,
    to_ideal_line,
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
    spans = [("A", 300.0, 300 + np.pi * 60)]
    ref = np.full(spline.n, 30.0)
    vp = profile_from_reference(spline.points[:, 3], ref, spline.xz, spans, car)
    in_a = (spline.points[:, 3] >= 300) & (spline.points[:, 3] <= 300 + np.pi * 60)
    assert vp[in_a].max() <= 30.0 * 0.95 + 1e-6
    seg = np.hypot(*(np.roll(spline.xz, -1, axis=0) - spline.xz).T)
    decel = (vp**2 - np.roll(vp, -1) ** 2) / (2 * seg)
    assert decel.max() <= 0.95 * 9.81 + 1e-3


def test_build_target_and_ideal_line() -> None:
    spline = _oval()
    edge = _right_edge(spline)
    target = build_conservative_target(
        spline,
        edge,
        ref_s=spline.points[:, 3],
        ref_speed_kmh=np.full(spline.n, 100.0),
        corner_spans=[("A", 300.0, 300 + np.pi * 60), ("B", 900.0, 900 + np.pi * 60)],
        car=CarModel(),
        risk_zones=[(0.0, 50.0, 1.0)],
    )
    assert target.lap_time_s > 0
    marks = target.brake_points([("A", 300.0, 300 + np.pi * 60)])
    assert "A" in marks and marks["A"] < 300.0
    ideal = to_ideal_line(spline, target)
    assert ideal.n == spline.n and ideal.extra[:, 0].max() > 0


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
    # previous lap; within the lap the cue words come in driving order.
    assert kinds == ["brake", "turn", "apex", "throttle", "approach"]
    assert next(c for c in cues if c.kind == "approach" and c.corner == "Alpha").at_m > 900
    assert [c.at_m for c in cues] == sorted(c.at_m for c in cues)
    brake = next(c for c in cues if c.corner == "Alpha" and c.kind == "brake")
    assert brake.text == "Brake" and brake.urgent
    assert brake.at_m < pace.marks["Alpha"]
    only = pace.cues(only={"Bravo"})
    assert {c.corner for c in only} == {"Bravo"}


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
    assert quick_verdict(slow, None, mark) == "Alpha: braked early."
    previous = PassRecord(lap=1, row=slow, mark_m=mark)
    verdict = quick_verdict(best, previous, mark)
    assert verdict.startswith("Alpha: better") and "on the mark" in verdict
    assert quick_verdict(best, PassRecord(1, best, None), None) == ""
    from src.coach.pacenotes import spoken_name

    assert spoken_name("Cotton Corners (right)") == "Cotton right"
    assert spoken_name("Phil Hill") == "Phil Hill"
    assert template_note(ref_metrics[0]).call_brake
