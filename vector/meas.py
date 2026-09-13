from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from . import defs


@dataclass(frozen=True, eq=False)
class Measurement:
    position: defs.PositionInput
    spectrum: defs.SpectrumInput

    @property
    def phasor(self) -> NDArray[np.complex128]:
        yaw = self.position.sensor_yaw_rad
        if yaw == 0.0:
            return self.spectrum.phasor_nt
        sensor = self.spectrum.phasor_nt
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return np.array(
            [
                cosine * sensor[0] - sine * sensor[1],
                sine * sensor[0] + cosine * sensor[1],
                sensor[2],
            ],
            dtype=complex,
        )


@dataclass(frozen=True, eq=False)
class Collection:
    name: str
    measurements: tuple[Measurement, ...]
    _positions_m: NDArray[np.float64] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _phasors_nt: NDArray[np.complex128] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _stderr_nt: NDArray[np.float64] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        rows = tuple(self.measurements)
        if not self.name:
            raise ValueError("collection name cannot be empty")
        if not rows:
            raise ValueError("collection cannot be empty")
        frames = {row.position.frame for row in rows}
        if len(frames) != 1:
            raise ValueError("all collection positions must use one frame")
        object.__setattr__(self, "measurements", rows)

    @property
    def positions_m(self) -> NDArray[np.float64]:
        values = self._positions_m
        if values is None:
            values = np.array(
                [
                    [row.position.horizontal_m, row.position.vertical_m]
                    for row in self.measurements
                ]
            )
            values.setflags(write=False)
            object.__setattr__(self, "_positions_m", values)
        return values

    @property
    def phasors_nt(self) -> NDArray[np.complex128]:
        values = self._phasors_nt
        if values is None:
            values = np.vstack([row.phasor for row in self.measurements])
            values.setflags(write=False)
            object.__setattr__(self, "_phasors_nt", values)
        return values

    @property
    def stderr_nt(self) -> NDArray[np.float64]:
        values = self._stderr_nt
        if values is None:
            values = np.array([row.spectrum.stderr_nt for row in self.measurements])
            values.setflags(write=False)
            object.__setattr__(self, "_stderr_nt", values)
        return values
