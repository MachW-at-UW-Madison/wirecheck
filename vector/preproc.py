from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Callable, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import (
    minimize_scalar as _minimize_scalar,
)

from . import defs

DEFAULT_SEGMENT_SECONDS = 4.0
NOMINAL_MAINS_HZ = (50.0, 60.0)
MAINS_SEARCH_HZ = 2.0
MIN_BLOCK_SAMPLES = 256
NOISE_PROBE_HZ = 5.0
MIN_BLOCK_STDERR_NT = 0.5


class _ScalarMinimum(Protocol):
    x: float


minimize_scalar: Callable[..., _ScalarMinimum] = cast(
    Callable[..., _ScalarMinimum], _minimize_scalar
)


@dataclass(frozen=True)
class PreprocessConfig:
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS
    nominal_mains_hz: tuple[float, ...] = NOMINAL_MAINS_HZ
    search_half_width_hz: float = MAINS_SEARCH_HZ

    def __post_init__(self) -> None:
        if self.segment_seconds <= 0.0:
            raise ValueError("lock-in segment length must be positive")
        if not self.nominal_mains_hz:
            raise ValueError("at least one nominal mains frequency is required")
        if self.search_half_width_hz <= 0.0:
            raise ValueError("mains search width must be positive")


@dataclass(frozen=True, eq=False)
class LineSpectrum:
    line_hz: float
    phasor_nt: NDArray[np.complex128]
    stderr_nt: float
    quadrature_ratio: float
    block_count: int
    drift_db: float
    circularity: float

    def __post_init__(self) -> None:
        phasor = np.array(self.phasor_nt, dtype=complex, copy=True)
        if phasor.shape != (3,):
            raise ValueError("line phasor must contain exactly three axes")
        phasor.setflags(write=False)
        object.__setattr__(self, "phasor_nt", phasor)


def _mains_family(
    frequencies: NDArray[np.floating[Any]],
    spectrum: NDArray[np.complex128],
    config: PreprocessConfig,
) -> float:
    best, best_power = None, -1.0
    for nominal in config.nominal_mains_hz:
        near = np.abs(frequencies - nominal) <= config.search_half_width_hz
        if not near.any():
            continue
        power = float(np.max(np.sum(np.abs(spectrum[near]) ** 2, axis=1)))
        if power > best_power:
            best, best_power = nominal, power
    if best is None:
        choices = " or ".join(f"{value:g}" for value in config.nominal_mains_hz)
        raise ValueError(
            f"sample rate too low: no bin within {config.search_half_width_hz:g} "
            f"Hz of {choices} Hz"
        )
    return float(best)


def _block_phasor(
    t: NDArray[np.float64],
    axes_nt: NDArray[np.float64],
    fs_hz: float,
    nominal_hz: float | None = None,
    config: PreprocessConfig | None = None,
) -> tuple[float, NDArray[np.complex128]]:
    settings = config or PreprocessConfig()
    centered = axes_nt - np.mean(axes_nt, axis=0)
    window = np.hanning(t.size)
    weighted = centered * window[:, None]
    frequencies = np.fft.rfftfreq(t.size, 1.0 / fs_hz)
    spectrum = np.fft.rfft(weighted, axis=0)
    nominal = (
        _mains_family(frequencies, spectrum, settings)
        if nominal_hz is None
        else nominal_hz
    )
    near = np.abs(frequencies - nominal) <= settings.search_half_width_hz
    if not near.any():
        raise ValueError(
            f"no spectral bin within {settings.search_half_width_hz:g} Hz of "
            f"{nominal:g} Hz at {fs_hz:g} Hz sample rate"
        )
    coarse = float(
        frequencies[near][np.argmax(np.sum(np.abs(spectrum[near]) ** 2, axis=1))]
    )
    df = fs_hz / t.size

    def objective(frequency: float) -> float:
        oscillator = np.exp(-2j * np.pi * frequency * t)
        candidate: NDArray[np.complex128] = (
            2.0 / window.sum() * np.sum(oscillator[:, None] * weighted, axis=0)
        )
        inner = np.vdot(candidate, candidate)
        return -float(inner.real)

    optimum = minimize_scalar(
        objective,
        bounds=(coarse - df, coarse + df),
        method="bounded",
        options={"xatol": 2e-7, "maxiter": 40},
    )
    frequency = float(optimum.x)
    oscillator = np.exp(-2j * np.pi * frequency * t)
    phasor = (
        2.0
        / window.sum()
        * np.sum(oscillator[:, None] * weighted, axis=0)
        / math.sqrt(2.0)
    )
    common_phase = 0.5 * np.angle(np.sum(phasor * phasor))
    return frequency, phasor * np.exp(-1j * common_phase)


def _align_block_phasors(
    phasors: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    reference = phasors[int(np.argmax(np.linalg.norm(phasors, axis=1)))]
    overlap = phasors @ np.conj(reference)
    magnitude = np.abs(overlap)
    gauge = np.where(magnitude > 1e-15, overlap / magnitude, 1.0 + 0.0j)
    return phasors * np.conj(gauge)[:, None]


def _circularity(phasors: NDArray[np.complex128]) -> float:
    squared = np.abs(np.sum(phasors * phasors, axis=1))
    energy = np.sum(np.abs(phasors) ** 2, axis=1)
    return float(np.median(squared / np.maximum(energy, 1e-30)))


def preprocess_recording(
    recording: defs.RecordingInput, config: PreprocessConfig | None = None
) -> LineSpectrum:
    settings = config or PreprocessConfig()
    fs_hz = recording.fs_hz
    block_size = max(MIN_BLOCK_SAMPLES, int(round(settings.segment_seconds * fs_hz)))
    if recording.n_samples < 2 * block_size:
        usable = recording.n_samples / fs_hz
        block_s = block_size / fs_hz
        remedy = (
            f"record at least {2 * block_s:.1f} s"
            if block_size <= MIN_BLOCK_SAMPLES
            else f"record at least {2 * block_s:.1f} s or use blocks no longer "
            f"than {max(usable / 2.0, MIN_BLOCK_SAMPLES / fs_hz):.1f} s"
        )
        floor_note = (
            f" (the {MIN_BLOCK_SAMPLES}-sample floor pins the block at "
            f"{block_s:g} s)"
            if block_size <= MIN_BLOCK_SAMPLES
            else ""
        )
        raise ValueError(
            f"needs two lock-in blocks of {block_s:g} s but only "
            f"{usable:.1f} s survived the settle crop{floor_note}; {remedy}"
        )

    raw_axes = recording.field_nt
    starts = range(0, recording.n_samples - block_size + 1, block_size)
    nominal_hz: float | None = None
    phasors: list[NDArray[np.complex128]] = []
    frequencies: list[float] = []
    for start in starts:
        stop = start + block_size
        frequency, phasor = _block_phasor(
            recording.time_s[start:stop] - recording.time_s[start],
            raw_axes[start:stop],
            fs_hz,
            nominal_hz,
            settings,
        )
        if nominal_hz is None:
            nominal_hz = min(
                settings.nominal_mains_hz, key=lambda value: abs(frequency - value)
            )
        frequencies.append(frequency)
        phasors.append(phasor)

    phasors_array: NDArray[np.complex128] = np.asarray(phasors)
    aligned = _align_block_phasors(phasors_array)
    mean_phasor = np.mean(aligned, axis=0)
    component_variance = np.var(aligned.real, axis=0, ddof=1) + np.var(
        aligned.imag, axis=0, ddof=1
    )
    independent_components = 2 * aligned.shape[1] - 1
    stderr = math.sqrt(
        float(np.sum(component_variance)) / (independent_components * aligned.shape[0])
    )
    real_norm = np.linalg.norm(phasors_array.real, axis=1)
    quadrature_ratio = float(
        np.median(
            np.linalg.norm(phasors_array.imag, axis=1) / np.maximum(real_norm, 1e-9)
        )
    )
    magnitudes: NDArray[np.float64] = np.linalg.norm(aligned, axis=1)
    half = magnitudes.size // 2
    first = float(np.mean(magnitudes[:half]))
    second = float(np.mean(magnitudes[half:]))
    drift_db = (
        20.0 * math.log10(max(first, second) / min(first, second))
        if min(first, second) > 0.0
        else float("inf")
    )
    return LineSpectrum(
        line_hz=float(np.median(frequencies)),
        phasor_nt=mean_phasor,
        stderr_nt=stderr,
        quadrature_ratio=quadrature_ratio,
        block_count=len(phasors_array),
        drift_db=drift_db,
        circularity=_circularity(phasors_array),
    )


def _windowed_phasor(
    t: NDArray[np.float64],
    weighted: NDArray[np.float64],
    window_sum: float,
    frequency: float,
) -> NDArray[np.complex128]:
    oscillator = np.exp(-2j * np.pi * frequency * t)
    return (
        2.0
        / window_sum
        * np.sum(oscillator[:, None] * weighted, axis=0)
        / math.sqrt(2.0)
    )


def preprocess_block(
    recording: defs.RecordingInput, config: PreprocessConfig | None = None
) -> LineSpectrum:
    settings = config or PreprocessConfig()
    fs_hz = recording.fs_hz
    block_size = max(MIN_BLOCK_SAMPLES, int(round(settings.segment_seconds * fs_hz)))
    if recording.n_samples < block_size:
        raise ValueError(
            f"block needs {block_size} samples ({block_size / fs_hz:g} s) but "
            f"the capture has {recording.n_samples}"
        )
    stop = block_size
    t = recording.time_s[:stop] - recording.time_s[0]
    axes = recording.field_nt[:stop]
    frequency, phasor = _block_phasor(t, axes, fs_hz, None, settings)
    centered = axes - np.mean(axes, axis=0)
    window = np.hanning(t.size)
    weighted = centered * window[:, None]
    noise = float(
        np.linalg.norm(
            _windowed_phasor(
                t, weighted, float(window.sum()), frequency + NOISE_PROBE_HZ
            )
        )
    )
    quadrature_ratio = float(
        np.linalg.norm(phasor.imag) / max(float(np.linalg.norm(phasor.real)), 1e-9)
    )
    return LineSpectrum(
        line_hz=frequency,
        phasor_nt=phasor,
        stderr_nt=max(noise, MIN_BLOCK_STDERR_NT),
        quadrature_ratio=quadrature_ratio,
        block_count=1,
        drift_db=0.0,
        circularity=_circularity(phasor[None, :]),
    )
