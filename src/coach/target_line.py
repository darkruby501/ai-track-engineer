"""A conservative target line: smoothest path with safety margins, plus speeds.

Starting from the track's AI spline (``ai/fast_lane.ai``, which also stores each
point's distance to the left and right track edges), we

1. pick a lateral offset at every point that minimises the line's curvature
   (a bounded linear least-squares problem), keeping the car's wheels a chosen
   margin inside both edges - more margin where going off is costly;
2. compute a speed profile on that line with a simple point-mass model
   (cornering grip, braking and power-limited acceleration, combined with a
   friction ellipse), using grip *below* what the car can do; and
3. read the target braking points off the speed profile.

The result can be written back in AC's ``.ai`` format as ``data/ideal_line.ai``
so AC's "ideal line" assist (Ctrl+I) draws it on track.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.optimize import lsq_linear

G = 9.81
SPEED, GAS, BRAKE, RADIUS, SIDE_L, SIDE_R, LENGTH_X = 0, 1, 2, 4, 5, 6, 12
BRAKE_DROP_KMH = 9.0  # a slow-down smaller than this is a lift, not a brake zone
FX, FZ = 13, 15


@dataclass
class AiSpline:
    """AC ``.ai`` spline (version 7) - only the parts we read or rewrite."""

    header: tuple[int, int, int, int]
    points: NDArray[np.float64]  # (n, 4): x, y, z, length
    ids: NDArray[np.int32]
    extra: NDArray[np.float64]  # (n, 18)
    tail: bytes  # everything after the extra block (kept verbatim)

    @property
    def xz(self) -> NDArray[np.float64]:
        return self.points[:, [0, 2]]

    @property
    def n(self) -> int:
        return len(self.points)


def read_ai_spline(path: str | Path) -> AiSpline:
    data = Path(path).read_bytes()
    header = struct.unpack_from("<4i", data, 0)
    n = header[1]
    off = 16
    rec = np.frombuffer(data, dtype=np.dtype([("p", "<f4", 4), ("id", "<i4")]), count=n, offset=off)
    off += n * 20
    (count,) = struct.unpack_from("<i", data, off)
    off += 4
    extra = np.frombuffer(data, dtype="<f4", count=count * 18, offset=off).reshape(count, 18)
    off += count * 72
    return AiSpline(
        header=(int(header[0]), int(header[1]), int(header[2]), int(header[3])),
        points=rec["p"].astype(np.float64),
        ids=rec["id"].astype(np.int32),
        extra=extra.astype(np.float64),
        tail=data[off:],
    )


def write_ai_spline(spline: AiSpline, path: str | Path) -> None:
    rec = np.zeros(spline.n, dtype=np.dtype([("p", "<f4", 4), ("id", "<i4")]))
    rec["p"] = spline.points.astype(np.float32)
    rec["id"] = spline.ids
    blob = (
        struct.pack("<4i", *spline.header)
        + rec.tobytes()
        + struct.pack("<i", len(spline.extra))
        + spline.extra.astype("<f4").tobytes()
        + spline.tail
    )
    Path(path).write_bytes(blob)


def _right_normals(spline: AiSpline) -> NDArray[np.float64]:
    """Unit normals to the spline (sign is fixed later by :func:`orient_normals`)."""
    xz = spline.xz
    fwd = np.roll(xz, -1, axis=0) - np.roll(xz, 1, axis=0)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    return np.asarray(np.c_[-fwd[:, 1], fwd[:, 0]], dtype=np.float64)


def orient_normals(spline: AiSpline, edges_right_xz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Right-pointing normals, with the sign checked against the right edge."""
    from scipy.spatial import cKDTree

    normals = _right_normals(spline)
    _, idx = cKDTree(edges_right_xz).query(spline.xz)
    toward_right = edges_right_xz[idx] - spline.xz
    agree = np.einsum("ij,ij->i", normals, toward_right)
    return normals if np.median(agree) > 0 else -normals


def optimise_offsets(
    spline: AiSpline,
    normals: NDArray[np.float64],
    *,
    half_width_m: float,
    wheel_margin_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Lateral offsets (metres, + = right) giving the least-curvature closed line."""
    n = spline.n
    keep = half_width_m + wheel_margin_m
    lo = -(spline.extra[:, SIDE_L] - keep)
    hi = spline.extra[:, SIDE_R] - keep
    bad = lo > hi  # track too narrow for the margin: centre it
    mid = (lo + hi) / 2
    lo = np.where(bad, mid - 1e-3, lo)
    hi = np.where(bad, mid + 1e-3, hi)
    # Second-difference operator on a closed loop.
    rows = np.repeat(np.arange(n), 3)
    cols = np.c_[(np.arange(n) - 1) % n, np.arange(n), (np.arange(n) + 1) % n].ravel()
    vals = np.tile([1.0, -2.0, 1.0], n)
    d2 = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    xz = spline.xz
    a = sparse.vstack([d2 @ sparse.diags(normals[:, 0]), d2 @ sparse.diags(normals[:, 1])])
    b = -np.r_[d2 @ xz[:, 0], d2 @ xz[:, 1]]
    result = lsq_linear(a, b, bounds=(lo, hi), method="trf", lsmr_tol="auto", max_iter=500)
    return np.asarray(result.x, dtype=np.float64)


def curvature(xz: NDArray[np.float64], smooth: int = 5) -> NDArray[np.float64]:
    """Signed curvature of a closed polyline (1/m)."""

    def sm(a: NDArray[np.float64]) -> NDArray[np.float64]:
        k = np.ones(smooth) / smooth
        pad = np.r_[a[-smooth:], a, a[:smooth]]
        return np.convolve(pad, k, "same")[smooth:-smooth]

    x, z = sm(xz[:, 0]), sm(xz[:, 1])
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(z)))]
    dx, dz = np.gradient(x, s), np.gradient(z, s)
    ddx, ddz = np.gradient(dx, s), np.gradient(dz, s)
    den = np.power(dx * dx + dz * dz, 1.5)
    den = np.where(den < 1e-9, 1e-9, den)
    return sm((dx * ddz - dz * ddx) / den)


@dataclass
class CarModel:
    lat_g: float = 0.9  # usable cornering grip
    brake_g: float = 0.95  # usable braking
    top_speed_kmh: float = 195.0
    # Full-throttle acceleration (g) by speed (km/h), measured from the AI lap.
    accel_speeds: tuple[float, ...] = (50.0, 90.0, 110.0, 130.0, 150.0, 175.0, 195.0)
    accel_g: tuple[float, ...] = (0.30, 0.27, 0.17, 0.15, 0.09, 0.04, 0.0)

    def accel_at(self, v_ms: float) -> float:
        return float(np.interp(v_ms * 3.6, self.accel_speeds, self.accel_g)) * G


def speed_profile(
    xz: NDArray[np.float64], car: CarModel, *, laps: int = 3
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(s, v m/s, curvature)`` along a closed line, by a forward/backward pass."""
    seg = np.hypot(*(np.roll(xz, -1, axis=0) - xz).T)
    s = np.r_[0.0, np.cumsum(seg)[:-1]]
    kappa = curvature(xz)
    a_lat = car.lat_g * G
    v_corner = np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-6))
    v_cap = np.minimum(v_corner, car.top_speed_kmh / 3.6)
    n = len(xz)
    v = v_cap.copy()
    for _ in range(laps):  # forward: power- and grip-limited acceleration
        for i in range(n):
            j = (i + 1) % n
            lat = v[i] ** 2 * abs(kappa[i])
            room = max(0.0, 1.0 - (lat / a_lat) ** 2) ** 0.5
            acc = min(car.accel_at(v[i]), car.brake_g * G * room)
            v[j] = min(v[j], (v[i] ** 2 + 2 * acc * seg[i]) ** 0.5)
    for _ in range(laps):  # backward: braking within the friction ellipse
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            lat = v[j] ** 2 * abs(kappa[j])
            room = max(0.0, 1.0 - (lat / a_lat) ** 2) ** 0.5
            dec = car.brake_g * G * room
            v[i] = min(v[i], (v[j] ** 2 + 2 * dec * seg[i]) ** 0.5)
    return s, v, kappa


@dataclass
class TargetLine:
    offsets: NDArray[np.float64]
    xz: NDArray[np.float64]
    s_spline: NDArray[np.float64]  # AC spline distance (metres) at each point
    v: NDArray[np.float64]
    kappa: NDArray[np.float64]
    lap_time_s: float
    margins: NDArray[np.float64] = field(repr=False, default_factory=lambda: np.zeros(0))

    def brake_points(self, spans: list[tuple[str, float, float]]) -> dict[str, float]:
        """Target braking point (spline metres) for corners that need braking.

        Each red (brake) zone of the ideal line belongs to the first corner that
        ends after it; its start is that corner's mark, so "Brake" is called
        exactly where the line turns red.
        """
        _, brake = pedal_flags(self.s_spline, self.v)
        on = brake > 0.5
        starts = np.flatnonzero(on & ~np.roll(on, 1))
        ends = np.flatnonzero(on & ~np.roll(on, -1))
        out: dict[str, float] = {}
        for a, b in zip(starts, ends if ends[0] >= starts[0] else np.roll(ends, -1), strict=True):
            end_m = float(self.s_spline[b])
            owner = next((name for name, _, stop in spans if stop >= end_m), None)
            if owner is not None and owner not in out:
                out[owner] = float(self.s_spline[a])
        return out


def build_target_line(
    spline: AiSpline,
    edges_right_xz: NDArray[np.float64],
    *,
    car: CarModel,
    half_width_m: float = 0.84,
    base_margin_m: float = 0.3,
    risk_zones: list[tuple[float, float, float]] | None = None,
) -> TargetLine:
    """Optimise a line and its speeds. ``risk_zones``: (from_m, to_m, wheel margin)."""
    s_spline = spline.points[:, 3]
    margins = np.full(spline.n, base_margin_m)
    for a, b, m in risk_zones or []:
        margins = np.where((s_spline >= a) & (s_spline <= b), np.maximum(margins, m), margins)
    normals = orient_normals(spline, edges_right_xz)
    offsets = optimise_offsets(spline, normals, half_width_m=half_width_m, wheel_margin_m=margins)
    xz = spline.xz + normals * offsets[:, None]
    _, v, kappa = speed_profile(xz, car)
    seg = np.hypot(*(np.roll(xz, -1, axis=0) - xz).T)
    lap_time = float(np.sum(seg / np.maximum(v, 1.0)))
    return TargetLine(offsets, xz, s_spline, v, kappa, lap_time, margins)


def pedal_flags(
    s: NDArray[np.float64],
    v: NDArray[np.float64],
    *,
    brake_drop_kmh: float = BRAKE_DROP_KMH,
    trail_g: float = 0.35,
    min_run_m: float = 12.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(gas, brake)`` per point, as AC colours the ideal line (iRacing style).

    Each stretch where the speed never rises is one slow-down. If it loses at
    least *brake_drop_kmh* it is braking (red) from its first falling point to
    its last point slowing harder than *trail_g*, then coasting (white) into
    the apex; a smaller slow-down is a lift (white). Everything else is
    throttle (green). Runs shorter than *min_run_m* take the colour before
    them, so the line never flickers.
    """
    n = len(v)
    dv = np.roll(v, -1) - v
    falling = dv < -0.02
    seg = np.r_[np.diff(s), 0.0]
    decel_g = (v**2 - np.roll(v, -1) ** 2) / (2 * np.maximum(seg, 1e-3) * G)
    label = np.zeros(n, dtype=np.int8)  # 0 throttle, 1 coast, 2 brake
    i = 0
    while i < n:
        if dv[i] > 0.005:
            i += 1
            continue
        j = i
        while j + 1 < n and dv[j + 1] <= 0.005:
            j += 1
        drops = np.flatnonzero(falling[i : j + 1])
        if drops.size:
            a, b = i + int(drops[0]), i + int(drops[-1])
            lost_kmh = (v[a] - v[min(b + 1, n - 1)]) * 3.6
            label[a : b + 1] = 1
            hard = np.flatnonzero(decel_g[a : b + 1] >= trail_g)
            if lost_kmh >= brake_drop_kmh and hard.size:
                label[a : a + int(hard[-1]) + 1] = 2
        i = j + 1
    start = 0
    for k in range(1, n + 1):
        if k == n or label[k] != label[start]:
            if start > 0 and float(np.sum(seg[start:k])) < min_run_m:
                label[start:k] = label[start - 1]
            start = k
    return (label == 0).astype(np.float64), (label == 2).astype(np.float64)


def to_ideal_line(spline: AiSpline, target: TargetLine) -> AiSpline:
    """A copy of *spline* moved onto the target line, with its speeds."""
    pts = spline.points.copy()
    pts[:, 0], pts[:, 2] = target.xz[:, 0], target.xz[:, 1]
    seg = np.hypot(*(np.roll(target.xz, -1, axis=0) - target.xz).T)
    pts[:, 3] = np.r_[0.0, np.cumsum(seg)[:-1]]
    extra = spline.extra.copy()
    extra[:, SPEED] = target.v
    extra[:, GAS], extra[:, BRAKE] = pedal_flags(pts[:, 3], target.v)
    extra[:, RADIUS] = 1.0 / np.maximum(np.abs(target.kappa), 1e-6)
    extra[:, SIDE_L] = spline.extra[:, SIDE_L] + target.offsets
    extra[:, SIDE_R] = spline.extra[:, SIDE_R] - target.offsets
    extra[:, LENGTH_X] = seg
    return AiSpline(spline.header, pts, spline.ids.copy(), extra, spline.tail)


def margin_offsets(
    spline: AiSpline,
    *,
    half_width_m: float,
    wheel_margin_m: NDArray[np.float64],
    smooth_points: int = 15,
) -> NDArray[np.float64]:
    """Smallest lateral moves (+ = right) that keep the wheels inside the margins.

    The AI line is kept wherever it already has the margin; elsewhere it is
    nudged inward just enough, and the nudges are smoothed so the line has no
    kinks.
    """
    keep = half_width_m + wheel_margin_m
    lo = -(spline.extra[:, SIDE_L] - keep)
    hi = spline.extra[:, SIDE_R] - keep
    mid = (lo + hi) / 2
    lo, hi = np.where(lo > hi, mid, lo), np.where(lo > hi, mid, hi)
    raw = np.clip(np.zeros(spline.n), lo, hi)
    k = np.ones(smooth_points) / smooth_points
    pad = np.r_[raw[-smooth_points:], raw, raw[:smooth_points]]
    smoothed = np.convolve(pad, k, "same")[smooth_points:-smooth_points]
    # Smoothing can undo part of a nudge; never give back margin.
    return np.where(
        raw > 0, np.maximum(smoothed, raw), np.where(raw < 0, np.minimum(smoothed, raw), smoothed)
    )


def turning_points(x: NDArray[np.float64], min_swing: float) -> list[int]:
    """Alternating peaks and troughs of a closed trace, ignoring swings < *min_swing*.

    Returned in order, starting with the global maximum.
    """
    n = len(x)
    start = int(np.argmax(x))
    order = (np.arange(n) + start) % n
    pts, cand, falling = [start], start, True
    for i in order[1:]:
        if falling:
            if x[i] < x[cand]:
                cand = int(i)
            elif x[i] - x[cand] > min_swing:
                pts.append(cand)
                cand, falling = int(i), False
        else:
            if x[i] > x[cand]:
                cand = int(i)
            elif x[cand] - x[i] > min_swing:
                pts.append(cand)
                cand, falling = int(i), True
    if not falling:  # the lap ends climbing back to the global maximum
        return pts
    pts.append(cand)
    return pts


def monotone_envelope(
    ref: NDArray[np.float64], slow_factor: float, *, min_swing_kmh: float = 5.0
) -> NDArray[np.float64]:
    """Speed cap that only falls into each slow point and only rises out of it.

    From each peak to the next trough the cap is *slow_factor* x the
    reference, never rising (its wobbles flattened); from each trough to the
    next peak it is the reference less the trough's margin (fading out by the
    peak), never falling.
    Lifts smaller than *min_swing_kmh* are ignored.
    """
    n = len(ref)
    pts = turning_points(ref, min_swing_kmh / 3.6)
    cap = ref.copy()
    for k, a in enumerate(pts):
        b = pts[(k + 1) % len(pts)]
        idx = (np.arange(a, a + (b - a) % n + 1)) % n
        if ref[a] >= ref[b]:  # peak -> trough: slow down, never speed up
            cap[idx] = np.minimum.accumulate(ref[idx] * slow_factor)
        else:  # trough -> peak: speed up, never slow down; the trough's margin
            # fades out so the next straight ends at full speed (early brake points)
            fade = np.linspace(1.0, 0.0, len(idx) - 1)
            out = ref[idx[1:]] - (1.0 - slow_factor) * ref[a] * fade
            cap[idx[1:]] = np.minimum.accumulate(out[::-1])[::-1]
    return cap


def profile_from_reference(
    s_spline: NDArray[np.float64],
    ref_speed_ms: NDArray[np.float64],
    xz: NDArray[np.float64],
    car: CarModel,
    *,
    corner_speed_factor: float = 0.95,
) -> NDArray[np.float64]:
    """Target speeds: the reference's slow-downs scaled down, braking at ``car.brake_g``.

    The cap is :func:`monotone_envelope` of the reference (``corner_speed_factor``
    x the reference while it slows, the reference while it speeds up). A forward
    pass applies acceleration limits and a backward pass the (gentler) braking.
    """
    n = len(s_spline)
    cap = np.minimum(
        monotone_envelope(np.asarray(ref_speed_ms, dtype=np.float64), corner_speed_factor),
        car.top_speed_kmh / 3.6,
    )
    seg = np.hypot(*(np.roll(xz, -1, axis=0) - xz).T)
    v = cap.copy()
    for _ in range(3):
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], (v[i] ** 2 + 2 * car.accel_at(v[i]) * seg[i]) ** 0.5)
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], (v[j] ** 2 + 2 * car.brake_g * G * seg[i]) ** 0.5)
    return v


DEFAULT_RISK_ZONES = {
    # (from_m, to_m, wheel margin m) along the AC spline, Buttonwillow 13CW
    ("nd_buttonwillow", "layout_classic_13"): [
        (1631.0, 1770.0, 1.0),  # Bus Stop and its exit (sources: costly off here)
        (2649.0, 2807.0, 1.0),  # Phil Hill crest (overshoot risk)
        (3330.0, 3800.0, 1.0),  # Sweeper exit and the Esses (rollover risk if off)
    ]
}


def build_conservative_target(
    spline: AiSpline,
    edges_right_xz: NDArray[np.float64],
    *,
    ref_s: NDArray[np.float64],
    ref_speed_kmh: NDArray[np.float64],
    car: CarModel,
    half_width_m: float = 0.84,
    base_margin_m: float = 0.3,
    risk_zones: list[tuple[float, float, float]] | None = None,
    corner_speed_factor: float = 0.95,
) -> TargetLine:
    """The AI line kept inside safety margins, with conservative target speeds."""
    s = spline.points[:, 3]
    margins = np.full(spline.n, base_margin_m)
    for a, b, m in risk_zones or []:
        margins = np.where((s >= a) & (s <= b), np.maximum(margins, m), margins)
    off = margin_offsets(spline, half_width_m=half_width_m, wheel_margin_m=margins)
    xz = spline.xz + orient_normals(spline, edges_right_xz) * off[:, None]
    v_ref = np.interp(s, ref_s, ref_speed_kmh) / 3.6
    v = profile_from_reference(s, v_ref, xz, car, corner_speed_factor=corner_speed_factor)
    seg = np.hypot(*(np.roll(xz, -1, axis=0) - xz).T)
    lap = float(np.sum(seg / np.maximum(v, 1.0)))
    return TargetLine(off, xz, s, v, curvature(xz), lap, margins)
