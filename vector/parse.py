from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from . import core

CURRENT_PER_COEFF = 5.0
MIN_LOAD_CURRENT_A = 0.1
LEAK_SUSPECT_MA = 12.0
LEAK_DETECT_MA = 18.0
DEFAULT_MIN_POSITIONS = 9
MIN_RESIDUAL_DOF_TO_CONFIRM = 4
MAX_STATIONARITY_DB = 3.5
MAX_REDUCED_CHI_SQUARE = 4.0
MIN_APERTURE_RATIO = 0.05
DEPTH_PINNED_TOLERANCE_M = 1e-4


class QualityThresholds(TypedDict):
    min_unique_positions: int
    min_aperture_ratio: float
    min_residual_dof: int
    max_reduced_chi_square: float
    max_drift_db: float


class QualityReport(TypedDict):
    unique_positions: int
    aperture_ratio: float
    residual_dof: int
    reduced_chi_square: float
    max_drift_db: float
    thresholds: QualityThresholds
    calibration_risk: str
    failures: list[str]


class FitReport(TypedDict):
    success: bool
    weighted_rss: float
    n_free: int
    param_names: list[str]
    params: list[float]
    message: str
    coherent_background_nt: float


class MeasurementQualityReport(TypedDict):
    median_quadrature_ratio: float
    median_lockin_stderr_nt: float
    median_circularity: float
    fractional_model_floor: float


class CollectionFitReport(TypedDict):
    name: str
    wire_horizontal_mm: float
    wire_vertical_mm: float
    coherent_vector_real_nt: list[float]
    coherent_vector_imag_nt: list[float]
    coherent_magnitude_nt: float


class Report(TypedDict):
    method: str
    health: str
    quality: QualityReport
    selected_model: str
    minimum_load_current_a: float
    leak_suspect_threshold_ma: float
    leak_detect_threshold_ma: float
    verdict_note: str
    leak_inphase_ma: float
    leak_quadrature_ma: float
    collection: str
    total_positions: int
    dipole_coeff_nt_m2: float
    estimated_pair_current_a: float
    estimated_load_current_a: float
    pair_current_note: str
    fit: FitReport
    plane_basis_sensor_coordinates: list[list[float]]
    plane_out_of_plane_energy_fraction: float
    measurement_quality: MeasurementQualityReport
    assumptions: list[str]
    collection_fit: CollectionFitReport
    leak_coeff_nt_m: float
    estimated_leak_current_ma: float
    load_current_note: str
    estimated_leak_phase_deg: float
    estimated_leak_phase_cosine_k: float
    phase_note: str


@dataclass(frozen=True)
class ParseConfig:
    pair_spacing_mm: float = core.PAIR_SPACING_MM
    min_positions: int = DEFAULT_MIN_POSITIONS
    min_residual_dof: int = MIN_RESIDUAL_DOF_TO_CONFIRM
    max_drift_db: float = MAX_STATIONARITY_DB
    max_reduced_chi_square: float = MAX_REDUCED_CHI_SQUARE
    min_aperture_ratio: float = MIN_APERTURE_RATIO

    def __post_init__(self) -> None:
        scalars = (
            self.pair_spacing_mm,
            self.max_drift_db,
            self.max_reduced_chi_square,
            self.min_aperture_ratio,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("interpretation thresholds must be finite")
        if self.pair_spacing_mm <= 0.0:
            raise ValueError("pair spacing must be positive")
        if self.min_positions <= 0 or self.min_residual_dof < 0:
            raise ValueError("position and residual-DOF thresholds are invalid")
        if self.max_drift_db <= 0.0 or self.max_reduced_chi_square <= 0.0:
            raise ValueError("quality upper bounds must be positive")
        if not 0.0 <= self.min_aperture_ratio <= 1.0:
            raise ValueError("aperture ratio must lie in [0,1]")


@dataclass(frozen=True)
class QualityObservations:
    positions_m: NDArray[np.float64]
    drift_db: NDArray[np.float64]
    circularity: NDArray[np.float64]
    quadrature_ratio: NDArray[np.float64]
    stderr_nt: NDArray[np.float64]

    def __post_init__(self) -> None:
        positions = np.array(self.positions_m, dtype=float, copy=True)
        arrays = [
            np.array(value, dtype=float, copy=True)
            for value in (
                self.drift_db,
                self.circularity,
                self.quadrature_ratio,
                self.stderr_nt,
            )
        ]
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("quality positions must have shape (N,2)")
        if any(value.shape != (positions.shape[0],) for value in arrays):
            raise ValueError("quality metadata length does not match positions")
        for value in [positions, *arrays]:
            value.setflags(write=False)
        object.__setattr__(self, "positions_m", positions)
        for name, value in zip(
            ("drift_db", "circularity", "quadrature_ratio", "stderr_nt"), arrays
        ):
            object.__setattr__(self, name, value)


def pair_current_a(dipole_coeff_nt_m2: float, pair_spacing_mm: float) -> float:
    return CURRENT_PER_COEFF * dipole_coeff_nt_m2 / pair_spacing_mm


def leak_current_ma(leak_coeff_nt_m: float) -> float:
    return CURRENT_PER_COEFF * leak_coeff_nt_m


def classify_health(load_current_a: float, fitted_leak_current_ma: float) -> str:
    if load_current_a < MIN_LOAD_CURRENT_A:
        return "no_cable"
    if fitted_leak_current_ma < LEAK_SUSPECT_MA:
        return "healthy"
    if fitted_leak_current_ma < LEAK_DETECT_MA:
        return "leakage_suspected"
    return "leakage_detected"


def aperture_ratio(positions_m: NDArray[np.float64]) -> float:
    if positions_m.shape[0] < 2:
        return 0.0
    singular = np.linalg.svd(positions_m - positions_m.mean(axis=0), compute_uv=False)
    return float(singular[1] / singular[0]) if singular[0] > 0.0 else 0.0


def _assess_quality(
    selected: core.VectorFit,
    observations: QualityObservations,
    config: ParseConfig,
    vertical_lower_bound_m: float,
) -> QualityReport:
    unique = len({tuple(position) for position in observations.positions_m})
    aperture = aperture_ratio(observations.positions_m)
    dof = selected.residual_dof
    reduced_chi_square = selected.weighted_rss / dof if dof > 0 else float("inf")
    max_drift = float(np.max(observations.drift_db))
    failures: list[str] = []
    if unique < config.min_positions:
        failures.append(
            f"only {unique} unique 2-D positions, need {config.min_positions}"
        )
    if aperture < config.min_aperture_ratio:
        failures.append(
            f"capture grid is nearly collinear (aperture ratio {aperture:.3f} "
            f"< {config.min_aperture_ratio}); spread captures in both axes"
        )
    if dof < config.min_residual_dof:
        failures.append(
            f"only {dof} residual degrees of freedom, need "
            f"{config.min_residual_dof}"
        )
    if not reduced_chi_square <= config.max_reduced_chi_square:
        scale = (
            "immeasurably"
            if not math.isfinite(reduced_chi_square)
            else f"{reduced_chi_square:.0f}x"
        )
        failures.append(
            f"fit is {scale} worse than measurement noise (reduced chi-square "
            f"> {config.max_reduced_chi_square:g}); field does not look like "
            "a single cable pair"
        )
    if max_drift > config.max_drift_db:
        failures.append(
            f"a capture drifted {max_drift:.1f} dB, over the "
            f"{config.max_drift_db:g} dB limit"
        )
    depth_m = float(selected.wire_position_m()[1])
    if depth_m - vertical_lower_bound_m <= DEPTH_PINNED_TOLERANCE_M:
        failures.append(
            f"fitted cable depth sits on the {vertical_lower_bound_m:.3f} m "
            "lower bound, so the leak estimate is not trustworthy; set "
            "--z1-height-mm for this frame's origin"
        )
    return {
        "unique_positions": unique,
        "aperture_ratio": aperture,
        "residual_dof": dof,
        "reduced_chi_square": reduced_chi_square,
        "max_drift_db": max_drift,
        "thresholds": {
            "min_unique_positions": config.min_positions,
            "min_aperture_ratio": config.min_aperture_ratio,
            "min_residual_dof": config.min_residual_dof,
            "max_reduced_chi_square": config.max_reduced_chi_square,
            "max_drift_db": config.max_drift_db,
        },
        "calibration_risk": (
            "quality gates were tuned on three bench collections and one known "
            "failure; recalibrate before changing collection geometry"
        ),
        "failures": failures,
    }


def _fit_record(fitted: core.VectorFit) -> FitReport:
    return {
        "success": fitted.success,
        "weighted_rss": fitted.weighted_rss,
        "n_free": fitted.n_free,
        "param_names": [parameter.name for parameter in core.PARAMETERS],
        "params": [float(value) for value in fitted.params],
        "message": fitted.message,
        "coherent_background_nt": float(np.linalg.norm(fitted.coherent_vector())),
    }


def interpret(
    result: core.FitResult,
    observations: QualityObservations,
    *,
    config: ParseConfig | None = None,
) -> Report:
    settings = config or ParseConfig()
    if observations.positions_m.shape[0] != result.n_positions:
        raise ValueError("quality metadata does not match fitted observations")
    selected = result.fit
    quality = _assess_quality(
        selected, observations, settings, result.vertical_lower_bound_m
    )
    a = selected.dipole_coeff_nt_m2
    c = selected.leak_coeff_nt_m
    pair_current = pair_current_a(a, settings.pair_spacing_mm)
    leak_ma = leak_current_ma(c)
    estimated_load_current = pair_current + leak_ma / 2000.0
    health = classify_health(estimated_load_current, leak_ma)
    psi = selected.leak_phase_rad

    wire_position = selected.wire_position_m()
    coherent_array = np.asarray(selected.coherent_vector())
    collection_fit: CollectionFitReport = {
        "name": result.collection_name,
        "wire_horizontal_mm": 1000.0 * float(wire_position[0]),
        "wire_vertical_mm": 1000.0 * float(wire_position[1]),
        "coherent_vector_real_nt": [float(value) for value in np.real(coherent_array)],
        "coherent_vector_imag_nt": [float(value) for value in np.imag(coherent_array)],
        "coherent_magnitude_nt": float(np.linalg.norm(coherent_array)),
    }

    return {
        "method": (
            "segmented complex three-axis vector lock-in + "
            + (
                "one phase gauge shared by every capture"
                if selected.shared_reference
                else "per-capture phase gauges"
            )
            + " + fixed current thresholds"
        ),
        "health": health,
        "quality": quality,
        "selected_model": selected.model_name,
        "minimum_load_current_a": MIN_LOAD_CURRENT_A,
        "leak_suspect_threshold_ma": LEAK_SUSPECT_MA,
        "leak_detect_threshold_ma": LEAK_DETECT_MA,
        "verdict_note": (
            f"no cable below {MIN_LOAD_CURRENT_A:g} A fitted load; otherwise "
            f"healthy below {LEAK_SUSPECT_MA:g} mA fitted leak, suspected "
            f"from {LEAK_SUSPECT_MA:g} mA, and detected from {LEAK_DETECT_MA:g} mA"
        ),
        "leak_inphase_ma": leak_current_ma(selected.value("c_inphase")),
        "leak_quadrature_ma": leak_current_ma(selected.value("c_quadrature")),
        "collection": result.collection_name,
        "total_positions": result.n_positions,
        "dipole_coeff_nt_m2": a,
        "estimated_pair_current_a": pair_current,
        "estimated_load_current_a": estimated_load_current,
        "pair_current_note": (
            "this is I_load-I_leak/2 in the symmetric far-field decomposition; "
            "spacing uncertainty is not included"
        ),
        "fit": _fit_record(selected),
        "plane_basis_sensor_coordinates": [
            [float(value) for value in row] for row in result.oriented_plane_basis
        ],
        "plane_out_of_plane_energy_fraction": result.out_of_plane_fraction,
        "measurement_quality": {
            "median_quadrature_ratio": float(np.median(observations.quadrature_ratio)),
            "median_lockin_stderr_nt": float(np.median(observations.stderr_nt)),
            "median_circularity": float(np.median(observations.circularity)),
            "fractional_model_floor": result.fractional_floor,
        },
        "assumptions": [
            "each capture's sensor pose matches the scan plan, and orientation "
            "is fixed within a capture",
            "relative axis I/Q phase is preserved and axis delays are negligible",
            "coherent nuisance field is uniform and phase-stable",
        ],
        "collection_fit": collection_fit,
        "leak_coeff_nt_m": c,
        "estimated_leak_current_ma": leak_ma,
        "load_current_note": "adds I_leak/2 to symmetric pair current",
        "estimated_leak_phase_deg": math.degrees(psi),
        "estimated_leak_phase_cosine_k": math.cos(psi),
        "phase_note": (
            "phase is fitted as Cartesian in-phase/quadrature components; "
            "the magnitude drives the hard-threshold verdict"
        ),
    }
