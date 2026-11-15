"""Claude-powered driving coach: pace notes, per-corner advice, lap analysis."""

from __future__ import annotations

from src.coach.coaches import CoachContext, CornerAdvisor, HeadCoach
from src.coach.live import LiveCoach, say_lap_time
from src.coach.metrics import CornerComparison, CornerMetrics, compare, measure_lap
from src.coach.pacenotes import CornerNote, PaceNotes
from src.coach.reference import Reference, best_lap, load_reference, save_reference
from src.coach.samples import LapTrace, Sample, read_samples_csv, split_laps

__all__ = [
    "CoachContext",
    "CornerAdvisor",
    "CornerComparison",
    "CornerMetrics",
    "CornerNote",
    "HeadCoach",
    "LapTrace",
    "LiveCoach",
    "PaceNotes",
    "Reference",
    "Sample",
    "best_lap",
    "compare",
    "load_reference",
    "measure_lap",
    "read_samples_csv",
    "save_reference",
    "say_lap_time",
    "split_laps",
]
