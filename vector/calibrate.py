from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol, Sequence, cast

import numpy as np
from scipy.optimize import (
    brentq,
    minimize_scalar,
)

from . import core, defs, parse


class _ScalarMinimum(Protocol):
    @property
    def success(self) -> bool: ...

    @property
    def message(self) -> str: ...

    @property
    def x(self) -> float: ...

    @property
    def fun(self) -> float: ...


MIN_COLLECTIONS = 2
MIN_POSITIONS = parse.DEFAULT_MIN_POSITIONS
MIN_APERTURE_RATIO = parse.MIN_APERTURE_RATIO
MAX_DRIFT_DB = parse.MAX_STATIONARITY_DB
MIN_FLOOR = 1e-4
MAX_FLOOR = 0.3
PROFILE_95 = 3.84
LOG_FLOOR_TOLERANCE = 1e-4


@dataclass(frozen=True)
class CollectionCalibration:
    name: str
    frame: str
    positions: int
    aperture_ratio: float
    max_drift_db: float
    median_field_nt: float
    median_stderr_nt: float
    handedness: int


@dataclass(frozen=True)
class _CollectionGates:
    positions: int
    aperture_ratio: float
    max_drift_db: float


@dataclass(frozen=True)
class FloorCalibration:
    fractional_floor: float
    low_95: float
    low_95_open: bool
    high_95: float
    high_95_open: bool
    search_min: float
    search_max: float
    objective_evaluations: int
    profile_threshold: float
    collections: tuple[CollectionCalibration, ...]
    method: str = (
        "pooled healthy-coherent Gaussian maximum likelihood; one shared "
        "fractional floor, independent position/background fits"
    )


def _position_set(collection: defs.MeasurementCollectionInput) -> np.ndarray:
    positions = np.unique(collection.positions_m, axis=0)
    order = np.lexsort((positions[:, 1], positions[:, 0]))
    return positions[order]


def _validate_collections(
    collections: Sequence[defs.MeasurementCollectionInput],
) -> tuple[str, np.ndarray, tuple[_CollectionGates, ...]]:
    if len(collections) < MIN_COLLECTIONS:
        raise ValueError(
            f"floor calibration needs at least {MIN_COLLECTIONS} independent "
            "healthy collections"
        )
    names = [collection.name for collection in collections]
    if len(set(names)) != len(names):
        raise ValueError("calibration collection names must be unique")

    reference_positions = _position_set(collections[0])
    reference_frames = {row.position.frame for row in collections[0].measurements}
    if len(reference_frames) != 1:
        raise ValueError("one calibration collection contains multiple frames")
    frame = next(iter(reference_frames))

    gates: list[_CollectionGates] = []
    for collection in collections:
        frames = {row.position.frame for row in collection.measurements}
        if frames != {frame}:
            raise ValueError(
                "all calibration collections must use the same coordinate frame"
            )
        positions = _position_set(collection)
        if positions.shape != reference_positions.shape or not np.allclose(
            positions, reference_positions, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                "all calibration collections must contain the same scan positions"
            )
        if positions.shape[0] < MIN_POSITIONS:
            raise ValueError(
                f"calibration needs at least {MIN_POSITIONS} unique positions"
            )
        aperture = parse.aperture_ratio(positions)
        if aperture < MIN_APERTURE_RATIO:
            raise ValueError(
                f"calibration scan is nearly collinear (aperture {aperture:.3f})"
            )
        max_drift = max(row.spectrum.drift_db for row in collection.measurements)
        if max_drift > MAX_DRIFT_DB:
            raise ValueError(
                f"{collection.name}: capture drift {max_drift:.2f} dB exceeds "
                f"the calibration limit {MAX_DRIFT_DB:g} dB"
            )
        gates.append(
            _CollectionGates(positions.shape[0], aperture, max_drift)
        )
    return frame, reference_positions, tuple(gates)


def _gaussian_neg2_log_likelihood(
    fitted: core.VectorFit,
    data: core.VectorData,
) -> float:
    residual = data.observed_nt - fitted.capture_gauges[:, None] * fitted.prediction_nt
    chi_square = float(np.sum(np.abs(residual) ** 2 / data.sigma_nt[:, None] ** 2))
    log_determinant = 6.0 * float(np.sum(np.log(data.sigma_nt**2)))
    return chi_square + log_determinant


def calibrate_fractional_floor(
    collections: Sequence[defs.MeasurementCollectionInput],
    *,
    fit_config: core.FitConfig | None = None,
    search_min: float = MIN_FLOOR,
    search_max: float = MAX_FLOOR,
    profile_threshold: float = PROFILE_95,
) -> FloorCalibration:
    if not 0.0 < search_min < search_max < 1.0:
        raise ValueError("floor search must satisfy 0 < min < max < 1")
    if not math.isfinite(profile_threshold) or profile_threshold <= 0.0:
        raise ValueError("profile threshold must be positive and finite")
    frame, _, gates = _validate_collections(collections)
    settings = fit_config or core.FitConfig()

    handednesses: list[int] = []
    warm: list[core.VectorFit] = []
    for collection in collections:
        seed_data = core.make_vector_data(collection, settings)
        selected = core.select_handedness(seed_data, pins=core.NO_LEAK_PINS)
        if selected is None:
            raise RuntimeError(
                f"{collection.name}: healthy calibration fit did not converge"
            )
        handednesses.append(selected.handedness)
        warm.append(selected)

    evaluations = 0

    def objective(log_floor: float) -> float:
        nonlocal evaluations
        floor = math.exp(float(log_floor))
        total = 0.0
        for index, collection in enumerate(collections):
            trial = replace(settings, fractional_floor=floor)
            data = core.make_vector_data(collection, trial)
            fitted = core.fit_model(
                data,
                handednesses[index],
                pins=core.NO_LEAK_PINS,
                starts=[warm[index].params],
            )
            if not fitted.success:
                raise RuntimeError(
                    f"{collection.name}: floor-profile fit did not converge"
                )
            warm[index] = fitted
            total += _gaussian_neg2_log_likelihood(fitted, data)
        evaluations += 1
        return total

    lower_log, upper_log = math.log(search_min), math.log(search_max)
    optimum = cast(
        _ScalarMinimum,
        minimize_scalar(
            objective,
            method="bounded",
            bounds=(lower_log, upper_log),
            options={"xatol": LOG_FLOOR_TOLERANCE},
        ),
    )
    if not bool(optimum.success):
        raise RuntimeError(f"floor calibration failed: {optimum.message}")
    center_log = float(optimum.x)
    center_objective = float(optimum.fun)
    target = center_objective + profile_threshold

    def profile_edge(bound: float) -> tuple[float, bool]:
        score_at_bound = objective(bound) - target
        if score_at_bound <= 0.0:
            return math.exp(bound), True

        def excess(value: float) -> float:
            return objective(value) - target

        root = cast(
            float,
            brentq(
                excess,
                min(bound, center_log),
                max(bound, center_log),
                xtol=LOG_FLOOR_TOLERANCE,
            ),
        )
        return math.exp(root), False

    low, low_open = profile_edge(lower_log)
    high, high_open = profile_edge(upper_log)
    diagnostics = tuple(
        CollectionCalibration(
            name=collection.name,
            frame=frame,
            positions=gates[index].positions,
            aperture_ratio=gates[index].aperture_ratio,
            max_drift_db=gates[index].max_drift_db,
            median_field_nt=float(
                np.median(np.linalg.norm(collection.phasors_nt, axis=1))
            ),
            median_stderr_nt=float(np.median(collection.stderr_nt)),
            handedness=handednesses[index],
        )
        for index, collection in enumerate(collections)
    )
    return FloorCalibration(
        fractional_floor=math.exp(center_log),
        low_95=low,
        low_95_open=low_open,
        high_95=high,
        high_95_open=high_open,
        search_min=search_min,
        search_max=search_max,
        objective_evaluations=evaluations,
        profile_threshold=profile_threshold,
        collections=diagnostics,
    )
