from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import calibrate, core, defs, meas, parse, preproc

_VERTICAL_LOWER_BOUND_M = -0.054


@dataclass(frozen=True)
class Config:
    segment_seconds: float = preproc.DEFAULT_SEGMENT_SECONDS
    block_seconds: float | None = None
    core_m: float = core.CORE_MM / 1000.0
    pair_spacing_m: float = core.PAIR_SPACING_MM / 1000.0
    vertical_lower_bound_m: float = _VERTICAL_LOWER_BOUND_M
    fractional_floor: float = core.DEFAULT_FRACTIONAL_FLOOR


@dataclass(frozen=True)
class CalibrationConfig:
    segment_seconds: float = preproc.DEFAULT_SEGMENT_SECONDS
    block_seconds: float | None = None
    core_m: float = core.CORE_MM / 1000.0
    pair_spacing_m: float = core.PAIR_SPACING_MM / 1000.0
    vertical_lower_bound_m: float = _VERTICAL_LOWER_BOUND_M
    seed_floor: float = core.DEFAULT_FRACTIONAL_FLOOR
    search_min: float = calibrate.MIN_FLOOR
    search_max: float = calibrate.MAX_FLOOR


@dataclass(frozen=True)
class Analysis:
    report: parse.Report
    _fit: core.FitResult

    def make_figure(self):
        from . import plot

        return plot.make_figure(self._fit, self.report["health"])


def _measure_collection(
    captures: defs.CaptureCollectionInput,
    *,
    block_seconds: float | None,
    segment_seconds: float,
) -> meas.Collection:
    try:
        raw_captures = captures.captures
    except AttributeError as error:
        raise TypeError("source must expose a raw captures collection") from error
    reference_hz = getattr(captures, "reference_hz", None)
    rows: list[meas.Measurement] = []
    for capture in raw_captures:
        try:
            if reference_hz is not None:
                spectrum = preproc.preprocess_reference_block(capture, reference_hz)
            elif block_seconds is not None:
                spectrum = preproc.preprocess_block(
                    capture, preproc.PreprocessConfig(block_seconds)
                )
            else:
                spectrum = preproc.preprocess_recording(
                    capture, preproc.PreprocessConfig(segment_seconds)
                )
        except ValueError as error:
            raise ValueError(f"{capture.source}: {error}") from error
        rows.append(meas.Measurement(capture.position, spectrum))
    return meas.Collection(
        captures.name, tuple(rows), shared_reference=reference_hz is not None
    )


def _quality_observations(
    collection: meas.Collection,
) -> parse.QualityObservations:
    rows = collection.measurements
    return parse.QualityObservations(
        positions_m=collection.positions_m,
        drift_db=np.array([row.spectrum.drift_db for row in rows]),
        circularity=np.array([row.spectrum.circularity for row in rows]),
        quadrature_ratio=np.array([row.spectrum.quadrature_ratio for row in rows]),
        stderr_nt=collection.stderr_nt,
    )


def analyze(
    source: defs.CaptureCollectionInput,
    config: Config | None = None,
) -> Analysis:
    settings = config or Config()
    collection = _measure_collection(
        source,
        block_seconds=settings.block_seconds,
        segment_seconds=settings.segment_seconds,
    )
    fitted = core.fit(
        collection,
        core.FitConfig(
            core_m=settings.core_m,
            pair_spacing_m=settings.pair_spacing_m,
            vertical_lower_bound_m=settings.vertical_lower_bound_m,
            fractional_floor=settings.fractional_floor,
        ),
    )
    report = parse.interpret(
        fitted,
        _quality_observations(collection),
        config=parse.ParseConfig(pair_spacing_mm=settings.pair_spacing_m * 1000.0),
    )
    return Analysis(report, fitted)


def calibrate_fractional_floor(
    sources: Sequence[defs.CaptureCollectionInput],
    config: CalibrationConfig | None = None,
) -> calibrate.FloorCalibration:
    settings = config or CalibrationConfig()
    collections = [
        _measure_collection(
            source,
            block_seconds=settings.block_seconds,
            segment_seconds=settings.segment_seconds,
        )
        for source in sources
    ]
    return calibrate.calibrate_fractional_floor(
        collections,
        fit_config=core.FitConfig(
            core_m=settings.core_m,
            pair_spacing_m=settings.pair_spacing_m,
            vertical_lower_bound_m=settings.vertical_lower_bound_m,
            fractional_floor=settings.seed_floor,
        ),
        search_min=settings.search_min,
        search_max=settings.search_max,
    )
