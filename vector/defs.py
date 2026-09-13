from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


class PositionInput(Protocol):
    @property
    def horizontal_m(self) -> float: ...

    @property
    def vertical_m(self) -> float: ...

    @property
    def frame(self) -> str: ...

    @property
    def sensor_yaw_rad(self) -> float: ...


class RecordingInput(Protocol):
    @property
    def time_s(self) -> NDArray[np.float64]: ...

    @property
    def field_nt(self) -> NDArray[np.float64]: ...

    @property
    def n_samples(self) -> int: ...

    @property
    def fs_hz(self) -> float: ...

    @property
    def duration_s(self) -> float: ...


class CaptureInput(RecordingInput, Protocol):
    @property
    def source(self) -> str: ...

    @property
    def position(self) -> PositionInput: ...


class CaptureCollectionInput(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def captures(self) -> Sequence[CaptureInput]: ...


class SpectrumInput(Protocol):
    @property
    def phasor_nt(self) -> NDArray[np.complex128]: ...

    @property
    def stderr_nt(self) -> float: ...

    @property
    def line_hz(self) -> float: ...

    @property
    def quadrature_ratio(self) -> float: ...

    @property
    def block_count(self) -> int: ...

    @property
    def drift_db(self) -> float: ...

    @property
    def circularity(self) -> float: ...


class MeasurementInput(Protocol):
    @property
    def position(self) -> PositionInput: ...

    @property
    def spectrum(self) -> SpectrumInput: ...

    @property
    def phasor(self) -> NDArray[np.complex128]: ...


class MeasurementCollectionInput(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def measurements(self) -> Sequence[MeasurementInput]: ...

    @property
    def positions_m(self) -> NDArray[np.float64]: ...

    @property
    def phasors_nt(self) -> NDArray[np.complex128]: ...

    @property
    def stderr_nt(self) -> NDArray[np.float64]: ...
