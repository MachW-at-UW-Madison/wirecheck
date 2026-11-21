from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, ClassVar, Literal, Mapping, Protocol, Sequence, cast

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.optimize import (
    least_squares as _least_squares,
)

from . import _vector_kernel, defs

CORE_MM = 0.0
PAIR_SPACING_MM = 5.5
DEFAULT_FRACTIONAL_FLOOR = 0.094
MIN_SIGMA_NT = 0.25
MIN_RADIUS_M = 1e-6
MAX_STARTS = 320
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Position = tuple[float, float]
ResidualFn = Callable[[FloatArray], FloatArray]
JacobianFn = Callable[[FloatArray], FloatArray]
JacobianArg = JacobianFn | Literal["2-point"]
CompiledPair = tuple[ResidualFn, JacobianFn]


ParameterName = Literal[
    "gamma",
    "theta",
    "log_a",
    "c_inphase",
    "c_quadrature",
    "x0",
    "y0",
    "u_real_x",
    "u_real_y",
    "u_real_z",
    "u_imag_x",
    "u_imag_y",
    "u_imag_z",
]
DEFAULT_LEAK_COEFF_SEED_NT_M = 7.8


class _LeastSquaresResult(Protocol):
    x: NDArray[np.float64]
    fun: NDArray[np.float64]
    success: bool
    message: str


least_squares: Callable[..., _LeastSquaresResult] = cast(
    Callable[..., _LeastSquaresResult], _least_squares
)


def _readonly(array: ArrayLike, *, dtype: DTypeLike | None = None) -> NDArray[Any]:
    value = np.array(array, dtype=dtype, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class FitConfig:
    core_m: float = CORE_MM / 1000.0
    pair_spacing_m: float = PAIR_SPACING_MM / 1000.0
    vertical_lower_bound_m: float = -0.054
    fractional_floor: float = DEFAULT_FRACTIONAL_FLOOR
    min_sigma_nt: float = MIN_SIGMA_NT
    leak_seed_coeff_nt_m: float = DEFAULT_LEAK_COEFF_SEED_NT_M
    vertical_position_seeds_m: tuple[float, ...] = (-0.039, -0.033)

    def __post_init__(self) -> None:
        scalars = {
            "core radius": self.core_m,
            "pair spacing": self.pair_spacing_m,
            "vertical lower bound": self.vertical_lower_bound_m,
            "fractional floor": self.fractional_floor,
            "minimum sigma": self.min_sigma_nt,
            "leak coefficient seed": self.leak_seed_coeff_nt_m,
        }
        for name, value in scalars.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.core_m < 0.0:
            raise ValueError("conductor radius must not be negative")
        if self.pair_spacing_m <= 0.0:
            raise ValueError("pair spacing must be positive")
        if self.min_sigma_nt <= 0.0:
            raise ValueError("minimum sigma must be positive")
        if self.leak_seed_coeff_nt_m <= 0.0:
            raise ValueError("leak coefficient seed must be positive")
        if not self.vertical_position_seeds_m or not np.all(
            np.isfinite(self.vertical_position_seeds_m)
        ):
            raise ValueError("vertical position seeds must be finite and nonempty")
        if not 0.0 < self.fractional_floor < 1.0:
            raise ValueError("fractional floor must lie strictly between 0 and 1")


class ParameterKind(Enum):
    ANGLE = auto()
    LOG_COEFFICIENT = auto()
    LINEAR_COEFFICIENT = auto()
    HORIZONTAL_POSITION = auto()
    VERTICAL_POSITION = auto()
    COHERENT_REAL = auto()
    COHERENT_IMAG = auto()


@dataclass(frozen=True)
class ParameterDef:
    NAME_TO_KIND: ClassVar[dict[ParameterName, ParameterKind]] = {
        "gamma": ParameterKind.ANGLE,
        "theta": ParameterKind.ANGLE,
        "log_a": ParameterKind.LOG_COEFFICIENT,
        "c_inphase": ParameterKind.LINEAR_COEFFICIENT,
        "c_quadrature": ParameterKind.LINEAR_COEFFICIENT,
        "x0": ParameterKind.HORIZONTAL_POSITION,
        "y0": ParameterKind.VERTICAL_POSITION,
        "u_real_x": ParameterKind.COHERENT_REAL,
        "u_real_y": ParameterKind.COHERENT_REAL,
        "u_real_z": ParameterKind.COHERENT_REAL,
        "u_imag_x": ParameterKind.COHERENT_IMAG,
        "u_imag_y": ParameterKind.COHERENT_IMAG,
        "u_imag_z": ParameterKind.COHERENT_IMAG,
    }

    name: ParameterName
    kind: ParameterKind = field(init=False)
    axis: int | None = None

    def __post_init__(self) -> None:
        kind = self.NAME_TO_KIND[self.name]
        object.__setattr__(self, "kind", kind)
        coherent = kind in {ParameterKind.COHERENT_REAL, ParameterKind.COHERENT_IMAG}
        if coherent != (self.axis is not None):
            raise ValueError("only coherent parameters require an axis")
        if self.axis is not None and self.axis not in (0, 1, 2):
            raise ValueError("coherent parameter axis must be 0, 1, or 2")


MODEL_NAME = "leaking_coherent"
PARAMETERS: tuple[ParameterDef, ...] = (
    ParameterDef("gamma"),
    ParameterDef("theta"),
    ParameterDef("log_a"),
    ParameterDef("c_inphase"),
    ParameterDef("c_quadrature"),
    ParameterDef("x0"),
    ParameterDef("y0"),
    ParameterDef("u_real_x", axis=0),
    ParameterDef("u_real_y", axis=1),
    ParameterDef("u_real_z", axis=2),
    ParameterDef("u_imag_x", axis=0),
    ParameterDef("u_imag_y", axis=1),
    ParameterDef("u_imag_z", axis=2),
)
PARAMETER_INDEX: dict[str, int] = {
    parameter.name: index for index, parameter in enumerate(PARAMETERS)
}
N_PARAMETERS = len(PARAMETERS)
NO_LEAK_PINS: Mapping[str, float] = {"c_inphase": 0.0, "c_quadrature": 0.0}


@dataclass(frozen=True, eq=False)
class VectorData:
    collection_name: str
    horizontal_m: FloatArray
    vertical_m: FloatArray
    observed_nt: ComplexArray
    sigma_nt: FloatArray
    plane_basis: FloatArray
    plane_eigenvalues: FloatArray
    config: FitConfig
    shared_reference: bool = False

    def __post_init__(self) -> None:
        horizontal: NDArray[np.float64] = _readonly(self.horizontal_m, dtype=float)
        vertical: NDArray[np.float64] = _readonly(self.vertical_m, dtype=float)
        observed: NDArray[np.complex128] = _readonly(self.observed_nt, dtype=complex)
        sigma: NDArray[np.float64] = _readonly(self.sigma_nt, dtype=float)
        basis: NDArray[np.float64] = _readonly(self.plane_basis, dtype=float)
        eigenvalues: NDArray[np.float64] = _readonly(
            self.plane_eigenvalues, dtype=float
        )
        n = horizontal.size
        if not self.collection_name:
            raise ValueError("collection name cannot be empty")
        if n == 0:
            raise ValueError("collection cannot be empty")
        if vertical.shape != (n,) or observed.shape != (n, 3):
            raise ValueError("inconsistent fit-data shapes")
        if sigma.shape != (n,):
            raise ValueError("inconsistent sigma shape")
        if not (
            np.all(np.isfinite(horizontal))
            and np.all(np.isfinite(vertical))
            and np.all(np.isfinite(observed.real))
            and np.all(np.isfinite(observed.imag))
            and np.all(np.isfinite(sigma))
        ):
            raise ValueError("collection contains non-finite values")
        if np.any(sigma <= 0.0):
            raise ValueError("collection uncertainties must be positive")
        object.__setattr__(self, "horizontal_m", horizontal)
        object.__setattr__(self, "vertical_m", vertical)
        object.__setattr__(self, "observed_nt", observed)
        object.__setattr__(self, "sigma_nt", sigma)
        object.__setattr__(self, "plane_basis", basis)
        object.__setattr__(self, "plane_eigenvalues", eigenvalues)

    @property
    def n_positions(self) -> int:
        return int(self.observed_nt.shape[0])

    @property
    def n_components(self) -> int:
        return 6 * self.n_positions

    @property
    def out_of_plane_fraction(self) -> float:
        total = float(self.plane_eigenvalues.sum())
        return (
            max(0.0, float(self.plane_eigenvalues.min())) / total
            if total > 0.0
            else 0.0
        )


@dataclass(frozen=True, eq=False)
class VectorFit:
    handedness: int
    params: FloatArray
    prediction_nt: ComplexArray
    capture_gauges: ComplexArray
    weighted_rss: float
    success: bool
    message: str
    n_pinned: int = 0
    shared_reference: bool = False

    def __post_init__(self) -> None:
        if self.handedness not in (-1, 1):
            raise ValueError("handedness must be +1 or -1")
        object.__setattr__(self, "params", _readonly(self.params, dtype=float))
        object.__setattr__(
            self, "prediction_nt", _readonly(self.prediction_nt, dtype=complex)
        )
        object.__setattr__(
            self, "capture_gauges", _readonly(self.capture_gauges, dtype=complex)
        )

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    @property
    def n_free(self) -> int:
        gauges = 1 if self.shared_reference else self.capture_gauges.size
        return N_PARAMETERS - self.n_pinned + gauges

    @property
    def n_observations(self) -> int:
        return 2 * self.prediction_nt.size

    @property
    def residual_dof(self) -> int:
        return self.n_observations - self.n_free

    def value(self, name: str) -> float:
        return float(self.params[PARAMETER_INDEX[name]])

    @property
    def dipole_coeff_nt_m2(self) -> float:
        return math.exp(self.value("log_a"))

    @property
    def leak_coeff_nt_m(self) -> float:
        return float(abs(_leak_coefficient(self.params)))

    @property
    def leak_phase_rad(self) -> float:
        return float(np.angle(_leak_coefficient(self.params)))

    def wire_position_m(self) -> np.ndarray:
        return np.array(
            [self.params[PARAMETER_INDEX["x0"]], self.params[PARAMETER_INDEX["y0"]]]
        )

    def coherent_vector(self) -> np.ndarray:
        return _coherent_from_params(self.params)


@dataclass(frozen=True)
class FitResult:
    fit: VectorFit
    _data: VectorData = field(repr=False, compare=False)

    @property
    def collection_name(self) -> str:
        return self._data.collection_name

    @property
    def n_positions(self) -> int:
        return self._data.n_positions

    @property
    def horizontal_m(self) -> FloatArray:
        return self._data.horizontal_m

    @property
    def vertical_m(self) -> FloatArray:
        return self._data.vertical_m

    @property
    def observed_nt(self) -> ComplexArray:
        return self._data.observed_nt

    @property
    def sigma_nt(self) -> FloatArray:
        return self._data.sigma_nt

    @property
    def out_of_plane_fraction(self) -> float:
        return self._data.out_of_plane_fraction

    @property
    def fractional_floor(self) -> float:
        return self._data.config.fractional_floor

    @property
    def core_m(self) -> float:
        return self._data.config.core_m

    @property
    def vertical_lower_bound_m(self) -> float:
        return self._data.config.vertical_lower_bound_m

    @property
    def oriented_plane_basis(self) -> np.ndarray:
        return _basis(self._data, self.fit.handedness)

    def predict_nt(self, params: np.ndarray) -> np.ndarray:
        return vector_predict(params, self._data, self.fit.handedness)


def make_vector_data(
    collection: defs.MeasurementCollectionInput,
    config: FitConfig | None = None,
) -> VectorData:
    settings = config or FitConfig()
    positions = collection.positions_m
    observed = collection.phasors_nt
    stderr = collection.stderr_nt
    norms = np.linalg.norm(observed, axis=1)
    sigma = np.maximum(
        np.hypot(stderr, settings.fractional_floor * norms), settings.min_sigma_nt
    )
    observed_real: NDArray[np.float64] = np.real(observed)
    observed_imag = np.imag(observed)
    covariance: NDArray[np.float64] = (
        observed_real.T @ observed_real + observed_imag.T @ observed_imag
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return VectorData(
        collection_name=collection.name,
        horizontal_m=positions[:, 0],
        vertical_m=positions[:, 1],
        observed_nt=observed,
        sigma_nt=sigma,
        plane_basis=eigenvectors[:, order[:2]],
        plane_eigenvalues=eigenvalues[order],
        config=settings,
        shared_reference=collection.shared_reference,
    )


def _compiled_pair(data: VectorData, handedness: int) -> CompiledPair | None:
    return _vector_kernel.compiled(
        _basis(data, handedness),
        horizontal_m=data.horizontal_m,
        vertical_m=data.vertical_m,
        observed_nt=data.observed_nt,
        sigma_nt=data.sigma_nt,
        core_m=data.config.core_m,
        pair_spacing_m=data.config.pair_spacing_m,
        shared_reference=data.shared_reference,
    )


def _basis(data: VectorData, handedness: int) -> np.ndarray:
    basis = np.array(data.plane_basis, copy=True)
    basis[:, 1] *= handedness
    return basis


def _leak_coefficient(params: np.ndarray) -> complex:
    return complex(
        params[PARAMETER_INDEX["c_inphase"]], params[PARAMETER_INDEX["c_quadrature"]]
    )


def _coherent_from_params(params: np.ndarray) -> np.ndarray:
    vector = np.zeros(3, dtype=complex)
    for parameter in PARAMETERS:
        coefficient = params[PARAMETER_INDEX[parameter.name]]
        if parameter.kind is ParameterKind.COHERENT_REAL:
            vector[parameter.axis] += coefficient
        elif parameter.kind is ParameterKind.COHERENT_IMAG:
            vector[parameter.axis] += 1j * coefficient
    return vector


def vector_predict(params: np.ndarray, data: VectorData, handedness: int) -> np.ndarray:
    basis = _basis(data, handedness)
    gamma = params[PARAMETER_INDEX["gamma"]]
    theta = params[PARAMETER_INDEX["theta"]]
    amplitude = math.exp(params[PARAMETER_INDEX["log_a"]])
    coefficient = _leak_coefficient(params)
    dx = data.horizontal_m - params[PARAMETER_INDEX["x0"]]
    dy = data.vertical_m - params[PARAMETER_INDEX["y0"]]
    core2 = data.config.core_m**2
    half = 0.5 * data.config.pair_spacing_m
    offset_x, offset_y = half * math.cos(theta), half * math.sin(theta)
    live_x, live_y = dx - offset_x, dy - offset_y
    return_x, return_y = dx + offset_x, dy + offset_y
    live = np.maximum(live_x * live_x + live_y * live_y + core2, MIN_RADIUS_M**2)
    back = np.maximum(
        return_x * return_x + return_y * return_y + core2, MIN_RADIUS_M**2
    )
    scale = amplitude / data.config.pair_spacing_m
    bx = (scale * (return_y / back - live_y / live)).astype(complex)
    by = (scale * (live_x / live - return_x / back)).astype(complex)
    radius = np.maximum(np.sqrt(dx * dx + dy * dy + core2), MIN_RADIUS_M)
    phi = np.arctan2(dy, dx)
    monopole = 1.0 / radius
    bx += coefficient * (-monopole * np.sin(phi))
    by += coefficient * (monopole * np.cos(phi))
    cos_g, sin_g = math.cos(gamma), math.sin(gamma)
    plane_x = cos_g * bx - sin_g * by
    plane_y = sin_g * bx + cos_g * by
    prediction = plane_x[:, None] * basis[:, 0] + plane_y[:, None] * basis[:, 1]
    prediction += _coherent_from_params(params)
    return prediction


def _optimal_phase_gauges(
    observed: np.ndarray,
    prediction: np.ndarray,
    sigma: np.ndarray,
    shared_reference: bool = False,
) -> np.ndarray:
    cross = np.sum(np.conj(prediction) * observed / sigma[:, None] ** 2, axis=1)
    if shared_reference:
        cross = np.full(cross.shape, cross.sum())
    magnitude = np.abs(cross)
    return np.where(magnitude > 1e-15, cross / magnitude, 1.0 + 0.0j)


def vector_residual(
    params: np.ndarray, data: VectorData, handedness: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = vector_predict(params, data, handedness)
    gauges = _optimal_phase_gauges(
        data.observed_nt, prediction, data.sigma_nt, data.shared_reference
    )
    residual = (data.observed_nt - gauges[:, None] * prediction) / data.sigma_nt[
        :, None
    ]
    flattened = np.concatenate((residual.real.ravel(), residual.imag.ravel()))
    return flattened, prediction, gauges


def _bounds(data: VectorData) -> tuple[np.ndarray, np.ndarray]:
    span = max(float(np.ptp(data.horizontal_m)), float(np.ptp(data.vertical_m)), 0.02)
    max_field = max(float(np.max(np.linalg.norm(data.observed_nt, axis=1))), 1.0)
    lower: list[float] = []
    upper: list[float] = []
    for parameter in PARAMETERS:
        lo: float
        hi: float
        if parameter.kind is ParameterKind.ANGLE:
            lo, hi = -4.0 * np.pi, 4.0 * np.pi
        elif parameter.kind is ParameterKind.LOG_COEFFICIENT:
            lo, hi = -25.0, 25.0
        elif parameter.kind is ParameterKind.LINEAR_COEFFICIENT:
            reach = 2.0 * max_field * (span + data.config.core_m)
            lo, hi = -reach, reach
        elif parameter.kind is ParameterKind.HORIZONTAL_POSITION:
            lo = float(data.horizontal_m.min() - span)
            hi = float(data.horizontal_m.max() + span)
        elif parameter.kind is ParameterKind.VERTICAL_POSITION:
            lo = max(
                float(data.vertical_m.min() - span), data.config.vertical_lower_bound_m
            )
            hi = float(data.vertical_m.max() + span)
        elif parameter.kind in {
            ParameterKind.COHERENT_REAL,
            ParameterKind.COHERENT_IMAG,
        }:
            lo, hi = -2.0 * max_field, 2.0 * max_field
        else:
            raise KeyError(parameter.kind)
        if not lo < hi:
            raise ValueError(
                f"empty bound for {parameter.name}: {lo:g} >= {hi:g}; "
                "check the position frame and vertical lower bound"
            )
        lower.append(lo)
        upper.append(hi)
    return np.asarray(lower), np.asarray(upper)


def _cold_starts(data: VectorData) -> list[FloatArray]:
    norms = np.linalg.norm(data.observed_nt, axis=1)
    brightest = int(np.argmax(norms))
    x_guess = float(data.horizontal_m[brightest])
    depths = tuple(
        dict.fromkeys(
            (*data.config.vertical_position_seeds_m, data.config.vertical_lower_bound_m)
        )
    )
    positions: tuple[Position, ...] = tuple((x_guess, depth) for depth in depths)
    gammas = np.linspace(0.0, np.pi, 4, endpoint=False)
    thetas = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)
    grid = [
        (float(gamma), theta, position)
        for gamma in gammas
        for theta in thetas
        for position in positions
    ]
    if len(grid) > MAX_STARTS:
        raise RuntimeError(f"{len(grid)} start points exceeds {MAX_STARTS}")
    starts: list[FloatArray] = []
    for gamma, theta, (x0, y0) in grid:
        radius = math.sqrt(
            (data.horizontal_m[brightest] - x0) ** 2
            + (data.vertical_m[brightest] - y0) ** 2
            + data.config.core_m**2
        )
        values: dict[str, float] = {
            "gamma": gamma,
            "theta": theta,
            "log_a": math.log(max(norms[brightest] * radius**2, 1e-10)),
            "c_inphase": data.config.leak_seed_coeff_nt_m,
            "c_quadrature": 0.0,
            "x0": x0,
            "y0": y0,
        }
        for parameter in PARAMETERS:
            if parameter.axis is not None:
                values[parameter.name] = 0.0
        starts.append(np.array([values[parameter.name] for parameter in PARAMETERS]))
    return starts


def fit_model(
    data: VectorData,
    handedness: int,
    *,
    starts: Sequence[FloatArray] | None = None,
    pins: Mapping[str, float] | None = None,
) -> VectorFit:
    pin_values = pins or {}
    try:
        pinned = {
            PARAMETER_INDEX[name]: float(value) for name, value in pin_values.items()
        }
    except KeyError as error:
        raise ValueError(f"unknown pinned parameter {error.args[0]!r}") from error
    free_mask = np.ones(N_PARAMETERS, dtype=bool)
    free_mask[list(pinned)] = False
    pinned_values = np.zeros(N_PARAMETERS)
    for index, value in pinned.items():
        pinned_values[index] = value

    def expand(free_params: FloatArray) -> FloatArray:
        full = pinned_values.copy()
        full[free_mask] = free_params
        return full

    lower_full, upper_full = _bounds(data)
    lower, upper = lower_full[free_mask], upper_full[free_mask]
    compiled = _compiled_pair(data, handedness)
    jacobian_arg: JacobianArg
    if compiled is None:

        def residual_fn(free_params: FloatArray, /) -> FloatArray:
            return vector_residual(expand(free_params), data, handedness)[0]

        jacobian_arg = "2-point"
    else:
        full_residual, full_jacobian = compiled
        if pinned:

            def residual_fn(free_params: FloatArray, /) -> FloatArray:
                return full_residual(expand(free_params))

            def reduced_jacobian(free_params: FloatArray) -> FloatArray:
                return full_jacobian(expand(free_params))[:, free_mask]

            jacobian_arg = reduced_jacobian
        else:
            residual_fn = full_residual
            jacobian_arg = full_jacobian
    seeds = starts if starts is not None else _cold_starts(data)
    candidates = [np.asarray(seed, dtype=float)[free_mask] for seed in seeds]

    def solve(candidate: FloatArray) -> _LeastSquaresResult:
        start = np.clip(candidate, lower + 1e-9, upper - 1e-9)
        return least_squares(
            residual_fn,
            start,
            jac=jacobian_arg,
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=12000,
        )

    solutions = [solve(candidate) for candidate in candidates]
    best_success: tuple[float, _LeastSquaresResult] | None = None
    best_any: tuple[float, _LeastSquaresResult] | None = None
    for solution in solutions:
        rss = float(np.sum(solution.fun**2))
        if best_any is None or rss < best_any[0]:
            best_any = rss, solution
        if solution.success and (best_success is None or rss < best_success[0]):
            best_success = rss, solution
    selected = best_success if best_success is not None else best_any
    if selected is None:
        raise RuntimeError("least-squares fitting produced no candidates")
    rss, solution = selected
    solution_params = expand(solution.x)
    if not pin_values:
        c_real = solution_params[PARAMETER_INDEX["c_inphase"]]
        c_imag = solution_params[PARAMETER_INDEX["c_quadrature"]]
        if c_real < 0.0 or (c_real == 0.0 and c_imag < 0.0):
            solution_params[PARAMETER_INDEX["theta"]] = (
                solution_params[PARAMETER_INDEX["theta"]] + 2.0 * math.pi
            ) % (2.0 * math.pi) - math.pi
            solution_params[PARAMETER_INDEX["c_inphase"]] *= -1.0
            solution_params[PARAMETER_INDEX["c_quadrature"]] *= -1.0
            for parameter in PARAMETERS:
                if parameter.axis is not None:
                    solution_params[PARAMETER_INDEX[parameter.name]] *= -1.0
    _, prediction, gauges = vector_residual(solution_params, data, handedness)
    return VectorFit(
        handedness,
        solution_params,
        prediction,
        gauges,
        rss,
        bool(solution.success),
        str(solution.message),
        len(pinned),
        data.shared_reference,
    )


def select_handedness(
    data: VectorData,
    *,
    pins: Mapping[str, float] | None = None,
) -> VectorFit | None:
    converged = [
        candidate
        for candidate in (
            fit_model(data, handedness, pins=pins) for handedness in (1, -1)
        )
        if candidate.success
    ]
    if not converged:
        return None
    return min(converged, key=lambda candidate: candidate.weighted_rss)


def fit(
    collection: defs.MeasurementCollectionInput,
    config: FitConfig | None = None,
) -> FitResult:
    data = make_vector_data(collection, config)
    selected = select_handedness(data)
    if selected is None:
        raise RuntimeError("no converged handedness fit")
    return FitResult(selected, data)
