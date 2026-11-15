"""Reference laps: build them from recordings, store and load them.

A reference is one fast lap (e.g. AC's AI driving the same car and setup)
saved as a JSON "pack" keyed by track, layout and car.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.coach.samples import LapTrace, Sample, split_laps


@dataclass
class Reference:
    track: str
    layout: str
    car: str
    source: str
    lap: LapTrace

    @property
    def lap_time_s(self) -> float:
        return self.lap.lap_time_s


def reference_path(directory: str | Path, track: str, layout: str, car: str) -> Path:
    return Path(directory) / f"{track}__{layout}__{car}.json"


def best_lap(
    samples: list[Sample], *, length_m: float, skip_first: bool = True
) -> tuple[LapTrace, int]:
    """Pick the fastest complete lap (not in the pits) from *samples*.

    Returns the trace and AC's completed-laps value for it. The first lap of a
    recording is skipped by default (it starts mid-lap or from a standstill).
    """
    candidates = []
    for index, (lap_no, lap_samples, lap_ms) in enumerate(split_laps(samples)):
        if skip_first and index == 0:
            continue
        if lap_ms <= 0 or any(s.in_pit for s in lap_samples):
            continue
        candidates.append((lap_ms, lap_no, lap_samples))
    if not candidates:
        raise ValueError("no complete flying lap found in the recording")
    lap_ms, lap_no, lap_samples = min(candidates, key=lambda c: c[0])
    return LapTrace(lap_samples, length_m=length_m, lap_time_s=lap_ms / 1000.0), lap_no


def save_reference(ref: Reference, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "track": ref.track,
        "layout": ref.layout,
        "car": ref.car,
        "source": ref.source,
        "lap": ref.lap.to_dict(),
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def load_reference(path: str | Path) -> Reference:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Reference(
        track=data["track"],
        layout=data["layout"],
        car=data["car"],
        source=data["source"],
        lap=LapTrace.from_dict(data["lap"]),
    )
