"""Claude-powered driving coach: lap analysis vs a reference and spoken calls."""

from __future__ import annotations

from src.coach.live import LiveCoach, heuristic_plan, say_lap_time
from src.coach.metrics import CornerComparison, CornerMetrics, compare, measure_lap
from src.coach.planner import CoachPlan, CoachPlanner, CornerCall
from src.coach.reference import Reference, best_lap, load_reference, save_reference
from src.coach.samples import LapTrace, Sample, read_samples_csv, split_laps

__all__ = [
    "CoachPlan",
    "CoachPlanner",
    "CornerCall",
    "CornerComparison",
    "CornerMetrics",
    "LapTrace",
    "LiveCoach",
    "Reference",
    "Sample",
    "best_lap",
    "compare",
    "heuristic_plan",
    "load_reference",
    "measure_lap",
    "read_samples_csv",
    "save_reference",
    "say_lap_time",
    "split_laps",
]
