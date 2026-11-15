"""Telemetry samples and whole-lap traces used by the coach.

The coach works on a small, flat :class:`Sample` rather than full telemetry
frames so the same code runs on live shared memory and on recorded CSV files
(which is how it is tested and how sessions are replayed).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.telemetry.models import TelemetryFrame

# Columns written by the coach / AI-reference recorders (superset is fine).
CSV_COLUMNS = [
    "timestamp",
    "lap",
    "lap_pos",
    "speed_kmh",
    "gear",
    "gas",
    "brake",
    "steer_angle",
    "g_lat",
    "g_lon",
    "car_x",
    "car_z",
    "current_time_ms",
    "last_time_ms",
    "in_pit",
]


@dataclass(frozen=True, slots=True)
class Sample:
    """One telemetry sample.

    ``gear`` uses AC's raw encoding (0 = reverse, 1 = neutral, 2 = first ...).
    ``lap`` is AC's completed-laps counter, so it increments as a lap ends.
    """

    t: float
    lap: int
    pos: float
    speed_kmh: float
    gear: int
    gas: float
    brake: float
    steer: float = 0.0
    g_lat: float = 0.0
    g_lon: float = 0.0
    x: float = 0.0
    z: float = 0.0
    current_ms: int = 0
    last_ms: int = 0
    in_pit: bool = False

    @classmethod
    def from_frame(cls, frame: TelemetryFrame) -> Sample:
        p, g = frame.physics, frame.graphics
        return cls(
            t=frame.timestamp,
            lap=g.completed_laps,
            pos=g.normalized_car_position,
            speed_kmh=p.speed_kmh,
            gear=p.gear,
            gas=p.gas,
            brake=p.brake,
            steer=p.steer_angle,
            g_lat=p.g_force_lateral,
            g_lon=p.g_force_longitudinal,
            x=g.car_coordinates[0],
            z=g.car_coordinates[2],
            current_ms=g.current_time_ms,
            last_ms=g.last_time_ms,
            in_pit=bool(g.is_in_pit or g.is_in_pit_lane),
        )

    def row(self) -> list[object]:
        return [
            round(self.t, 4),
            self.lap,
            round(self.pos, 6),
            round(self.speed_kmh, 2),
            self.gear,
            round(self.gas, 3),
            round(self.brake, 3),
            round(self.steer, 3),
            round(self.g_lat, 3),
            round(self.g_lon, 3),
            round(self.x, 2),
            round(self.z, 2),
            self.current_ms,
            self.last_ms,
            int(self.in_pit),
        ]


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else default


def read_samples_csv(path: str | Path) -> Iterator[Sample]:
    """Yield samples from a recorder CSV (coach or AI-reference format)."""
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            yield Sample(
                t=_f(row, "timestamp"),
                lap=int(_f(row, "lap")),
                pos=_f(row, "lap_pos"),
                speed_kmh=_f(row, "speed_kmh"),
                gear=int(_f(row, "gear")),
                gas=_f(row, "gas"),
                brake=_f(row, "brake"),
                steer=_f(row, "steer_angle"),
                g_lat=_f(row, "g_lat"),
                g_lon=_f(row, "g_lon"),
                x=_f(row, "car_x"),
                z=_f(row, "car_z"),
                current_ms=int(_f(row, "current_time_ms")),
                last_ms=int(_f(row, "last_time_ms")),
                in_pit=bool(int(_f(row, "in_pit"))),
            )


class LapTrace:
    """A completed lap as distance-indexed arrays (metres from the S/F line).

    Samples are sorted by distance; samples from the previous lap that leak in
    around the line (position near 1.0 at the start) are dropped.
    """

    def __init__(self, samples: Iterable[Sample], *, length_m: float, lap_time_s: float) -> None:
        rows = list(samples)
        if len(rows) < 20:
            raise ValueError("a lap needs at least 20 samples")
        t0 = rows[0].t
        # Around the line the spline position can still read ~1.0 just after the
        # lap starts, or already ~0.0 just before it ends; drop those strays.
        n = len(rows)
        keep = [
            s
            for i, s in enumerate(rows)
            if not (s.pos > 0.9 and i < n // 4) and not (s.pos < 0.1 and i > 3 * n // 4)
        ]
        keep.sort(key=lambda s: s.pos)
        self.length_m = length_m
        self.lap_time_s = lap_time_s
        self.s: NDArray[np.float64] = np.array([s.pos * length_m for s in keep])
        self.t: NDArray[np.float64] = np.array([s.t - t0 for s in keep])
        self.speed: NDArray[np.float64] = np.array([s.speed_kmh for s in keep])
        self.gas: NDArray[np.float64] = np.array([s.gas for s in keep])
        self.brake: NDArray[np.float64] = np.array([s.brake for s in keep])
        self.gear: NDArray[np.int64] = np.array([s.gear for s in keep], dtype=np.int64)
        self.x: NDArray[np.float64] = np.array([s.x for s in keep])
        self.z: NDArray[np.float64] = np.array([s.z for s in keep])
        # Elapsed time must rise with distance for interpolation; enforce it.
        self.t = np.maximum.accumulate(self.t)

    def time_at(self, s_m: float) -> float:
        return float(np.interp(s_m, self.s, self.t))

    def speed_at(self, s_m: float) -> float:
        return float(np.interp(s_m, self.s, self.speed))

    def to_dict(self) -> dict[str, object]:
        return {
            "length_m": self.length_m,
            "lap_time_s": self.lap_time_s,
            "s": [round(float(v), 2) for v in self.s],
            "t": [round(float(v), 4) for v in self.t],
            "speed": [round(float(v), 2) for v in self.speed],
            "gas": [round(float(v), 3) for v in self.gas],
            "brake": [round(float(v), 3) for v in self.brake],
            "gear": [int(v) for v in self.gear],
            "x": [round(float(v), 2) for v in self.x],
            "z": [round(float(v), 2) for v in self.z],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> LapTrace:
        obj = cls.__new__(cls)
        obj.length_m = float(data["length_m"])  # type: ignore[arg-type]
        obj.lap_time_s = float(data["lap_time_s"])  # type: ignore[arg-type]
        obj.s = np.array(data["s"], dtype=float)
        obj.t = np.array(data["t"], dtype=float)
        obj.speed = np.array(data["speed"], dtype=float)
        obj.gas = np.array(data["gas"], dtype=float)
        obj.brake = np.array(data["brake"], dtype=float)
        obj.gear = np.array(data["gear"], dtype=np.int64)
        obj.x = np.array(data.get("x", []), dtype=float)
        obj.z = np.array(data.get("z", []), dtype=float)
        return obj


def split_laps(samples: Iterable[Sample]) -> list[tuple[int, list[Sample], int]]:
    """Group samples into laps: ``(completed_laps value, samples, lap_time_ms)``.

    A lap's time is AC's ``last_time_ms`` reported on the first sample after it
    ends; the trailing, unfinished lap is dropped.
    """
    laps: list[tuple[int, list[Sample], int]] = []
    current: list[Sample] = []
    for sample in samples:
        if current and sample.lap != current[-1].lap:
            laps.append((current[-1].lap, current, sample.last_ms))
            current = []
        current.append(sample)
    return laps
