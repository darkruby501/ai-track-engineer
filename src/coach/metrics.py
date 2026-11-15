"""Per-corner lap metrics and lap-vs-reference comparison.

For every corner we measure where the driver started braking, the speed they
carried into the corner, their minimum speed and where it happened, when they
got back on the throttle, and the time spent in the section ending at the
corner's exit. Comparing those numbers with the reference lap tells the coach
*what* was different, not only *where* time was lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.coach.samples import LapTrace
from src.knowledge.models import Corner
from src.knowledge.track_edges import TrackEdges

BRAKE_ON = 0.1
LIFT_BELOW = 0.2
THROTTLE_ON = 0.5
FULL_THROTTLE = 0.95
TURN_IN_FRACTION = 0.35  # of the corner's peak steering, in the corner's direction


@dataclass(frozen=True, slots=True)
class CornerMetrics:
    name: str
    start_m: float
    end_m: float
    entry_kmh: float
    peak_m: float
    brake_m: float | None
    lift_m: float | None
    peak_brake: float
    min_kmh: float
    min_m: float
    min_gear: int  # 1 = first gear
    throttle_m: float | None
    full_throttle_m: float | None
    section_s: float
    direction: str = ""  # "L" or "R" (from steering at the minimum-speed point)
    turn_in_m: float | None = None
    # Line placement in metres (needs track edges): gap to the inside edge at
    # turn-in and at the apex, and gap to the outside edge at the corner exit.
    turn_in_inside_m: float | None = None
    apex_inside_m: float | None = None
    exit_outside_m: float | None = None


@dataclass(frozen=True, slots=True)
class CornerComparison:
    driver: CornerMetrics
    ref: CornerMetrics

    @property
    def name(self) -> str:
        return self.driver.name

    @property
    def time_lost_s(self) -> float:
        return self.driver.section_s - self.ref.section_s

    @property
    def brake_diff_m(self) -> float | None:
        """Positive = driver braked *later* than the reference."""
        if self.driver.brake_m is None or self.ref.brake_m is None:
            return None
        return self.driver.brake_m - self.ref.brake_m

    @property
    def min_speed_diff(self) -> float:
        return self.driver.min_kmh - self.ref.min_kmh

    @property
    def entry_speed_diff(self) -> float:
        return self.driver.entry_kmh - self.ref.entry_kmh

    @property
    def throttle_diff_m(self) -> float | None:
        """Positive = driver got back on the throttle *later* than the reference."""
        if self.driver.throttle_m is None or self.ref.throttle_m is None:
            return None
        return self.driver.throttle_m - self.ref.throttle_m

    @property
    def turn_in_diff_m(self) -> float | None:
        """Positive = driver turned in *later* than the reference."""
        if self.driver.turn_in_m is None or self.ref.turn_in_m is None:
            return None
        return self.driver.turn_in_m - self.ref.turn_in_m


def corner_spans(corners: tuple[Corner, ...], length_m: float) -> list[tuple[str, float, float]]:
    """``(name, start_m, end_m)`` for each corner, in lap order (no wrapping corners)."""
    spans = [(c.name, c.entry * length_m, c.exit * length_m) for c in corners]
    return sorted(spans, key=lambda span: span[1])


def _first(mask: np.ndarray, start: int, stop: int) -> int | None:
    idx = np.flatnonzero(mask[start:stop])
    return int(start + idx[0]) if idx.size else None


def _lateral(edges: TrackEdges | None, lap: LapTrace, i: int, side: str) -> float | None:
    """Gap in metres from the car to the ``side`` ("L"/"R") edge at sample *i*."""
    if edges is None or side not in ("L", "R") or lap.x.size <= i:
        return None
    left, right = edges.lateral(float(lap.x[i]), float(lap.z[i]))
    return round(left if side == "L" else right, 2)


def _other(side: str) -> str:
    return {"L": "R", "R": "L"}.get(side, "")


def measure_lap(
    lap: LapTrace,
    corners: tuple[Corner, ...],
    *,
    edges: TrackEdges | None = None,
    only: set[str] | None = None,
) -> list[CornerMetrics]:
    """Measure every corner of *lap* (or just the corners named in *only*)."""
    spans = corner_spans(corners, lap.length_m)
    out: list[CornerMetrics] = []
    prev_min_i = 0
    prev_end_m = 0.0
    n = len(lap.s)
    for name, start_m, end_m in spans:
        lo = int(np.searchsorted(lap.s, start_m))
        hi = int(np.searchsorted(lap.s, end_m))
        if hi <= lo:
            continue
        i_min = lo + int(np.argmin(lap.speed[lo:hi]))
        search_from = min(prev_min_i, i_min)
        i_peak = search_from + int(np.argmax(lap.speed[search_from : i_min + 1]))
        i_brake = _first(lap.brake > BRAKE_ON, i_peak, i_min + 1)
        i_lift = _first(lap.gas < LIFT_BELOW, i_peak, i_min + 1)
        i_thr = _first(lap.gas > THROTTLE_ON, i_min, n)
        i_full = _first(lap.gas > FULL_THROTTLE, i_min, n)
        has_steer = lap.steer.size >= hi > lo
        mean_steer = float(lap.steer[lo:hi].mean()) if has_steer else 0.0
        steer_at_min = float(lap.steer[i_min]) if lap.steer.size > i_min else 0.0
        basis = steer_at_min if abs(steer_at_min) >= 0.02 else mean_steer
        direction = "" if abs(basis) < 0.02 else ("L" if basis < 0 else "R")
        # Apex: the minimum-speed point when the corner is braked for; for flat or
        # lift-only corners, the point of peak steering in the corner's direction.
        i_apex = i_min
        if i_brake is None and direction and has_steer:
            sgn = -1.0 if direction == "L" else 1.0
            i_apex = lo + int(np.argmax(lap.steer[lo:hi] * sgn))
        i_turn: int | None = None
        if direction:
            sign = -1.0 if direction == "L" else 1.0
            win_from = i_brake if i_brake is not None else i_peak
            signed = lap.steer[win_from : i_apex + 1] * sign
            if signed.size and signed.max() > 0:
                i_turn = _first(signed >= TURN_IN_FRACTION * signed.max(), 0, signed.size)
                i_turn = None if i_turn is None else win_from + i_turn
        i_exit = max(min(hi, n - 1), 0)
        prev_min_i_next = i_min
        if only is not None and name not in only:
            prev_min_i = prev_min_i_next
            prev_end_m = end_m
            continue
        out.append(
            CornerMetrics(
                name=name,
                start_m=start_m,
                end_m=end_m,
                entry_kmh=float(lap.speed[i_peak]),
                peak_m=float(lap.s[i_peak]),
                brake_m=float(lap.s[i_brake]) if i_brake is not None else None,
                lift_m=float(lap.s[i_lift]) if i_lift is not None else None,
                peak_brake=float(lap.brake[i_peak : i_min + 1].max()),
                min_kmh=float(lap.speed[i_min]),
                min_m=float(lap.s[i_min]),
                min_gear=int(lap.gear[i_min]) - 1,
                throttle_m=float(lap.s[i_thr]) if i_thr is not None else None,
                full_throttle_m=float(lap.s[i_full]) if i_full is not None else None,
                section_s=lap.time_at(end_m) - lap.time_at(prev_end_m),
                direction=direction,
                turn_in_m=float(lap.s[i_turn]) if i_turn is not None else None,
                turn_in_inside_m=(
                    _lateral(edges, lap, i_turn, direction) if i_turn is not None else None
                ),
                apex_inside_m=_lateral(edges, lap, i_apex, direction),
                exit_outside_m=_lateral(edges, lap, i_exit, _other(direction)),
            )
        )
        prev_min_i = prev_min_i_next
        prev_end_m = end_m
    return out


def compare(driver: list[CornerMetrics], ref: list[CornerMetrics]) -> list[CornerComparison]:
    """Pair driver and reference metrics by corner name."""
    by_name = {m.name: m for m in ref}
    return [CornerComparison(d, by_name[d.name]) for d in driver if d.name in by_name]


def _m(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}"


def _signed(v: float | None, fmt: str = "+.0f") -> str:
    return "-" if v is None else format(v, fmt)


def comparison_table(rows: list[CornerComparison]) -> str:
    """Compact text table of a lap vs the reference, for the LLM prompt."""
    header = (
        "corner | time lost s | brake @m you/ref (diff, + = later) | turn-in @m you/ref "
        "(diff) | entry km/h you/ref | min km/h you/ref (diff) | min gear you/ref | "
        "throttle @m you/ref (diff, + = later) | apex m from inside you/ref | "
        "exit m from outside you/ref"
    )
    lines = [header]
    for r in rows:
        d, f = r.driver, r.ref
        lines.append(
            f"{r.name} | {r.time_lost_s:+.2f} | {_m(d.brake_m)}/{_m(f.brake_m)} "
            f"({_signed(r.brake_diff_m)}) | {_m(d.turn_in_m)}/{_m(f.turn_in_m)} "
            f"({_signed(r.turn_in_diff_m)}) | {d.entry_kmh:.0f}/{f.entry_kmh:.0f} | "
            f"{d.min_kmh:.0f}/{f.min_kmh:.0f} ({r.min_speed_diff:+.0f}) | "
            f"{d.min_gear}/{f.min_gear} | {_m(d.throttle_m)}/{_m(f.throttle_m)} "
            f"({_signed(r.throttle_diff_m)}) | {_g(d.apex_inside_m)}/{_g(f.apex_inside_m)} | "
            f"{_g(d.exit_outside_m)}/{_g(f.exit_outside_m)}"
        )
    return "\n".join(lines)


def _g(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def reference_table(ref: list[CornerMetrics]) -> str:
    """The reference lap's corner-by-corner numbers, for the system prompt."""
    lines = [
        "corner | dir | span m | brake @m | turn-in @m | entry km/h | min km/h @m | gear | "
        "throttle @m | apex m from inside | exit m from outside"
    ]
    for m in ref:
        lines.append(
            f"{m.name} | {m.direction or '-'} | {m.start_m:.0f}-{m.end_m:.0f} | "
            f"{_m(m.brake_m)} | {_m(m.turn_in_m)} | {m.entry_kmh:.0f} | "
            f"{m.min_kmh:.0f} @{m.min_m:.0f} | {m.min_gear} | {_m(m.throttle_m)} | "
            f"{_g(m.apex_inside_m)} | {_g(m.exit_outside_m)}"
        )
    return "\n".join(lines)
