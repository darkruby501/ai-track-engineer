"""Track edges (``data/side_l.csv`` / ``side_r.csv``) and lateral position.

AC's edge files list points along the left and right track edges as
``x, y, -z, normalised position``; note the stored third column is the
*negated* world z. With both edges loaded we can say where a car is across the
track: metres from the left edge and from the right edge (in the driving
direction).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


class TrackEdges:
    def __init__(self, left_xz: NDArray[np.float64], right_xz: NDArray[np.float64]) -> None:
        if len(left_xz) < 2 or len(right_xz) < 2:
            raise ValueError("track edges need at least two points each")
        self.left = left_xz
        self.right = right_xz
        self._lt = cKDTree(left_xz)
        self._rt = cKDTree(right_xz)

    def lateral(self, x: float, z: float) -> tuple[float, float]:
        """``(metres from left edge, metres from right edge)`` for a point.

        Values can go slightly negative when the car is beyond an edge.
        """
        p = np.array([x, z])
        _, il = self._lt.query(p)
        _, ir = self._rt.query(p)
        a, b = self.left[int(il)], self.right[int(ir)]
        span = b - a
        width = float(np.hypot(*span))
        if width < 1e-6:
            return 0.0, 0.0
        frac = float(np.dot(p - a, span) / width**2)
        return frac * width, (1.0 - frac) * width

    def lateral_many(
        self, x: NDArray[np.float64], z: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        pts = np.c_[x, z]
        _, il = self._lt.query(pts)
        _, ir = self._rt.query(pts)
        a, b = self.left[il], self.right[ir]
        span = b - a
        width = np.hypot(span[:, 0], span[:, 1])
        width = np.where(width < 1e-6, 1e-6, width)
        frac = np.einsum("ij,ij->i", pts - a, span) / width**2
        return frac * width, (1.0 - frac) * width


def _read_edge(path: Path) -> NDArray[np.float64]:
    data = np.loadtxt(path, delimiter=",", usecols=(0, 1, 2), ndmin=2)
    return np.asarray(np.c_[data[:, 0], -data[:, 2]], dtype=np.float64)


def load_edges(track_dir: str | Path, *, layout: str = "") -> TrackEdges | None:
    """Load a layout's edges, or ``None`` if the track does not ship them."""
    base = Path(track_dir)
    if layout and (base / layout).is_dir():
        base = base / layout
    left, right = base / "data" / "side_l.csv", base / "data" / "side_r.csv"
    if not (left.is_file() and right.is_file()):
        return None
    return TrackEdges(_read_edge(left), _read_edge(right))
