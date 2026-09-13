from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import defs


@dataclass(frozen=True, eq=False)
class RawCapture:
    source: str
    position: defs.PositionInput
    time_s: NDArray[np.float64]
    field_nt: NDArray[np.float64]

    def __post_init__(self) -> None:
        time_s = np.array(self.time_s, dtype=np.float64, copy=True)
        field_nt = np.array(self.field_nt, dtype=np.float64, copy=True)
        if time_s.ndim != 1:
            raise ValueError("recording timestamps must be one-dimensional")
        if field_nt.shape != (time_s.size, 3):
            raise ValueError("recording magnetic data must have shape (N, 3)")
        if time_s.size < 2:
            raise ValueError("recording must contain at least two samples")
        if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(field_nt)):
            raise ValueError("recording data must be finite")
        if np.any(np.diff(time_s) <= 0.0):
            raise ValueError("recording timestamps must increase strictly")
        time_s.setflags(write=False)
        field_nt.setflags(write=False)
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "field_nt", field_nt)

    @property
    def n_samples(self) -> int:
        return int(self.time_s.size)

    @property
    def fs_hz(self) -> float:
        return 1.0 / float(np.median(np.diff(self.time_s)))

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])


@dataclass(frozen=True)
class RawCollection:
    name: str
    captures: tuple[RawCapture, ...]

    def __post_init__(self) -> None:
        captures = tuple(self.captures)
        if not self.name:
            raise ValueError("raw collection name cannot be empty")
        if not captures:
            raise ValueError("raw collection cannot be empty")
        object.__setattr__(self, "captures", captures)
