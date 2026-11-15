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
        """Target braking point (spline metres) for corners that need braking."""
        out: dict[str, float] = {}
        prev_min = 0
        for name, start_m, end_m in spans:
            idx = np.flatnonzero((self.s_spline >= start_m) & (self.s_spline <= end_m))
            if idx.size == 0:
                continue
            i_min = int(idx[np.argmin(self.v[idx])])
            lo = min(prev_min, i_min)
            i_peak = lo + int(np.argmax(self.v[lo : i_min + 1]))
            if self.v[i_peak] - self.v[i_min] > 3.0 / 3.6 * 3:  # > ~9 km/h drop = braking
                out[name] = float(self.s_spline[i_peak])
            prev_min = i_min
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


def to_ideal_line(spline: AiSpline, target: TargetLine) -> AiSpline:
    """A copy of *spline* moved onto the target line, with its speeds."""
    pts = spline.points.copy()
    pts[:, 0], pts[:, 2] = target.xz[:, 0], target.xz[:, 1]
    seg = np.hypot(*(np.roll(target.xz, -1, axis=0) - target.xz).T)
    pts[:, 3] = np.r_[0.0, np.cumsum(seg)[:-1]]
    extra = spline.extra.copy()
    extra[:, SPEED] = target.v
    dv = np.roll(target.v, -1) - target.v
    extra[:, GAS] = np.where(dv >= -0.01, 1.0, 0.0)
    extra[:, BRAKE] = np.where(dv < -0.05, 1.0, 0.0)
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


def profile_from_reference(
    s_spline: NDArray[np.float64],
    ref_speed_ms: NDArray[np.float64],
    xz: NDArray[np.float64],
    corner_spans: list[tuple[str, float, float]],
    car: CarModel,
    *,
    corner_speed_factor: float = 0.95,
) -> NDArray[np.float64]:
    """Target speeds: the reference's corner speeds scaled down, braking at ``car.brake_g``.

    Inside corner spans the cap is ``corner_speed_factor`` x the reference speed;
    on straights only power limits apply. A forward pass applies acceleration
    limits and a backward pass applies the (gentler) braking.
    """
    n = len(s_spline)
    in_corner = np.zeros(n, dtype=bool)
    for _, a, b in corner_spans:
        in_corner |= (s_spline >= a) & (s_spline <= b)
    cap = np.where(in_corner, ref_speed_ms * corner_speed_factor, car.top_speed_kmh / 3.6)
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
    corner_spans: list[tuple[str, float, float]],
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
    v = profile_from_reference(
        s, v_ref, xz, corner_spans, car, corner_speed_factor=corner_speed_factor
    )
    seg = np.hypot(*(np.roll(xz, -1, axis=0) - xz).T)
    lap = float(np.sum(seg / np.maximum(v, 1.0)))
    return TargetLine(off, xz, s, v, curvature(xz), lap, margins)
