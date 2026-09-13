from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from . import core

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from matplotlib.figure import Figure
    from matplotlib.transforms import Transform

FloatArray = NDArray[np.float64]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS_RULE = "#c3c2b7"
PAIR = "#2a78d6"
LEAK = "#eb6834"
MODEL = "#4a3aa7"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
_HEALTH_BADGE: dict[str, tuple[str, str]] = {
    "no_cable": ("NO CABLE", "warning"),
    "healthy": ("HEALTHY", "good"),
    "leakage_suspected": ("LEAK SUSPECTED", "warning"),
    "leakage_detected": ("LEAK DETECTED", "critical"),
}
_BADGE_COLOUR: dict[str, str] = {
    "no_cable": "#8a8a8a",
    "healthy": STATUS["good"],
    "leakage_suspected": STATUS["warning"],
    "leakage_detected": STATUS["critical"],
}
_AXIS_LABELS = ("sensor x", "sensor y", "sensor z")
_AXIS_COLOURS = (PAIR, LEAK, MODEL)

_STYLE: dict[str, object] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS_RULE,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8.0,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "legend.labelcolor": INK_SOFT,
    "font.size": 9.0,
}

_MEASURED: dict[str, object] = {
    "marker": "o",
    "ms": 5.0,
    "color": PAIR,
    "markeredgecolor": SURFACE,
    "markeredgewidth": 0.6,
    "linestyle": "none",
    "label": "measured field",
    "zorder": 4,
}
_LEFTOVER: dict[str, object] = {
    "marker": "s",
    "ms": 4.8,
    "color": LEAK,
    "markeredgecolor": SURFACE,
    "markeredgewidth": 0.6,
    "linestyle": "none",
    "label": "after pair/background removal",
    "zorder": 5,
}
_GUIDE: dict[str, object] = {"ls": (0, (3, 2)), "lw": 1.0, "zorder": 2}


class _Axis(Protocol):
    @property
    def transAxes(self) -> Transform:
        ...
    def scatter(self, x: object, y: object, **kwargs: object) -> object: ...
    def plot(self, *args: object, **kwargs: object) -> object: ...
    def fill_between(
        self,
        x: object,
        y1: object,
        y2: object,
        **kwargs: object,
    ) -> object: ...
    def text(
        self,
        x: float,
        y: float,
        s: str,
        **kwargs: object,
    ) -> object: ...
    def annotate(
        self,
        text: str,
        xy: tuple[float, float],
        **kwargs: object,
    ) -> object: ...
    def set(self, **kwargs: object) -> object: ...
    def legend(self, **kwargs: object) -> object: ...
    def grid(self, visible: bool, **kwargs: object) -> object: ...
    def set_xscale(self, value: str, **kwargs: object) -> None: ...
    def set_yscale(self, value: str, **kwargs: object) -> None: ...
    def set_xlim(self, left: float, right: float) -> object: ...
    def set_ylim(self, bottom: float, top: float) -> object: ...
    def set_aspect(self, aspect: str, **kwargs: object) -> object: ...
    def axhline(self, y: float, **kwargs: object) -> object: ...


class _BoundingBox(Protocol):
    @property
    def x1(self) -> float: ...


class _TextArtist(Protocol):
    def get_position(self) -> tuple[float, float]: ...
    def get_window_extent(self) -> _BoundingBox: ...
    def set_position(self, xy: tuple[float, float]) -> None: ...


class _FigureOperations(Protocol):
    def colorbar(
        self,
        mappable: object,
        *,
        ax: _Axis,
        label: str,
        **kwargs: object,
    ) -> object: ...
    def suptitle(self, t: str, **kwargs: object) -> _TextArtist: ...
    def text(self, x: float, y: float, s: str, **kwargs: object) -> _TextArtist: ...


class _Plotting(Protocol):
    def subplots(
        self,
        nrows: int,
        ncols: int,
        *,
        figsize: tuple[float, float],
        constrained_layout: bool,
    ) -> tuple[Figure, NDArray[np.object_]]: ...
    def rc_context(
        self,
        rc: dict[str, object],
    ) -> AbstractContextManager[None]: ...


def _bin_medians(
    coordinates: FloatArray,
    values: FloatArray,
    count: int,
    *,
    logarithmic: bool,
) -> tuple[FloatArray, FloatArray]:
    if coordinates.size == 0:
        return np.array([]), np.array([])
    if logarithmic:
        positive = coordinates[coordinates > 0.0]
        if positive.size == 0:
            return np.array([]), np.array([])
        edges = np.geomspace(
            float(positive.min()) * 0.999,
            float(positive.max()) * 1.001,
            count + 1,
        )
    else:
        edges = np.linspace(
            float(coordinates.min()) - 1e-9,
            float(coordinates.max()) + 1e-9,
            count + 1,
        )
    centres: list[float] = []
    medians: list[float] = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (coordinates >= low) & (coordinates < high)
        if not np.any(inside):
            continue
        centres.append(
            float(
                np.exp(np.mean(np.log(coordinates[inside])))
                if logarithmic
                else np.mean(coordinates[inside])
            )
        )
        medians.append(float(np.median(values[inside])))
    return np.asarray(centres), np.asarray(medians)


def _principal_angle(
    bx: NDArray[np.complex128], by: NDArray[np.complex128]
) -> FloatArray:
    turned = np.exp(-0.5j * np.angle(bx * bx + by * by))
    return np.arctan2(np.real(turned * by), np.real(turned * bx))


def _to_cable_plane(
    vectors: NDArray[np.complex128], result: core.FitResult
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    basis = result.oriented_plane_basis
    plane_x = vectors @ basis[:, 0]
    plane_y = vectors @ basis[:, 1]
    gamma = result.fit.value("gamma")
    cos_g, sin_g = math.cos(gamma), math.sin(gamma)
    return cos_g * plane_x + sin_g * plane_y, -sin_g * plane_x + cos_g * plane_y


def _fit_components(
    result: core.FitResult,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    fitted = result.fit
    pair_params = fitted.params.copy()
    pair_params[core.PARAMETER_INDEX["c_inphase"]] = 0.0
    pair_params[core.PARAMETER_INDEX["c_quadrature"]] = 0.0
    for parameter in core.PARAMETERS:
        if parameter.axis is not None:
            pair_params[core.PARAMETER_INDEX[parameter.name]] = 0.0
    pair = result.predict_nt(pair_params)
    background = fitted.coherent_vector()[None, :]
    leak = fitted.prediction_nt - pair - background
    aligned = result.observed_nt * np.conj(fitted.capture_gauges)[:, None]
    return aligned, pair, leak


def _fit_quality_panels(
    radial_axis: _Axis,
    angular_axis: _Axis,
    result: core.FitResult,
) -> None:
    fitted = result.fit
    aligned, pair, leak = _fit_components(result)
    background = fitted.coherent_vector()[None, :]
    leftover = aligned - pair - background
    wire = fitted.wire_position_m()
    dx = result.horizontal_m - wire[0]
    dy = result.vertical_m - wire[1]
    radius_m = np.maximum(np.hypot(dx, dy), core.MIN_RADIUS_M)
    radius_cm = 100.0 * radius_m
    bins = min(8, max(3, int(round(math.sqrt(result.n_positions)))))

    centres, measured_binned = _bin_medians(
        radius_cm, np.linalg.norm(aligned, axis=1), bins, logarithmic=True
    )
    _, leftover_binned = _bin_medians(
        radius_cm, np.linalg.norm(leftover, axis=1), bins, logarithmic=True
    )
    curve_cm = np.geomspace(
        max(float(radius_cm.min()) * 0.94, 1e-3),
        float(radius_cm.max()) * 1.06,
        120,
    )
    curve_radius = np.sqrt((curve_cm / 100.0) ** 2 + result.core_m**2)
    sample_radius = np.sqrt(radius_m**2 + result.core_m**2)
    pair_scale = float(np.median(np.linalg.norm(pair, axis=1) * sample_radius**2))
    leak_scale = float(np.median(np.linalg.norm(leak, axis=1) * sample_radius))
    radial_axis.plot(
        curve_cm,
        pair_scale / curve_radius**2,
        color=INK_MUTED,
        label=r"fitted pair $\propto r^{-2}$",
        **_GUIDE,
    )
    if leak_scale > 0.0:
        radial_axis.plot(
            curve_cm,
            leak_scale / curve_radius,
            color=LEAK,
            label=r"fitted leak $\propto r^{-1}$",
            **_GUIDE,
        )
    radial_axis.plot(centres, measured_binned, **_MEASURED)
    radial_axis.plot(centres, leftover_binned, **_LEFTOVER)
    radial_axis.set_xscale("log")
    radial_axis.set_yscale("log")
    radial_axis.grid(True, which="major", zorder=0)
    radial_axis.set(
        title="Fit separation: radial falloff",
        xlabel="radius from fitted cable [cm]",
        ylabel="|B| [nT]",
    )
    radial_axis.legend(loc="best", ncol=1)

    phi = np.arctan2(dy, dx)
    far = np.flatnonzero(radius_m >= min(0.10, float(np.median(radius_m))))
    if far.size < 4:
        far = np.arange(result.n_positions)
    far = far[np.argsort(phi[far])]
    observation_x, observation_y = _to_cable_plane(aligned[far], result)
    leftover_x, leftover_y = _to_cable_plane(leftover[far], result)
    observation_direction = (
        np.unwrap(2.0 * _principal_angle(observation_x, observation_y)) / 2.0
    )
    leftover_direction = np.unwrap(2.0 * _principal_angle(leftover_x, leftover_y)) / 2.0
    azimuth = np.unwrap(phi[far])
    angle, observation_turn = _bin_medians(
        np.degrees(azimuth),
        np.degrees(observation_direction),
        bins,
        logarithmic=False,
    )
    _, leftover_turn = _bin_medians(
        np.degrees(azimuth),
        np.degrees(leftover_direction),
        bins,
        logarithmic=False,
    )
    if angle.size:
        swing = angle - angle[0]
        observation_turn -= observation_turn[0]
        leftover_turn -= leftover_turn[0]
        span = np.array([0.0, max(float(swing.max()) * 1.08, 1.0)])
        angular_axis.plot(
            span,
            2.0 * span,
            color=INK_MUTED,
            label=r"pair direction $2\phi$",
            **_GUIDE,
        )
        angular_axis.plot(
            span, span, color=LEAK, label=r"leak direction $\phi$", **_GUIDE
        )
        angular_axis.plot(swing, observation_turn, **_MEASURED)
        angular_axis.plot(swing, leftover_turn, **_LEFTOVER)
    angular_axis.grid(True, which="major", zorder=0)
    angular_axis.set(
        title="Fit separation: angular turning",
        xlabel="observation azimuth change [deg]",
        ylabel="field direction change [deg]",
    )
    angular_axis.legend(loc="best", ncol=1)


def _aperture_outline(points_mm: FloatArray) -> tuple[FloatArray, float]:
    if points_mm.shape[0] < 2:
        return points_mm.copy(), 0.0
    centre = points_mm.mean(axis=0)
    centred = points_mm - centre
    _, singular, basis = np.linalg.svd(centred, full_matrices=False)
    ratio = float(singular[1] / singular[0]) if singular[0] > 0.0 else 0.0
    scale = singular / math.sqrt(max(points_mm.shape[0] - 1, 1))
    turn = np.linspace(0.0, 2.0 * math.pi, 181)
    outline = (
        centre
        + np.cos(turn)[:, None] * scale[0] * basis[0]
        + np.sin(turn)[:, None] * scale[1] * basis[1]
    )
    return outline, ratio


def _geometry_panel(
    figure: _FigureOperations,
    axis: _Axis,
    result: core.FitResult,
    observed_nt: FloatArray,
    level: str,
) -> None:
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    ramp = mpl.colormaps["Blues"]
    shades = cast(
        Sequence[tuple[float, float, float, float]], ramp(np.linspace(0.30, 0.97, 256))
    )
    field_map = LinearSegmentedColormap.from_list("field", shades)

    points_mm = np.column_stack(
        (1000.0 * result.horizontal_m, 1000.0 * result.vertical_m)
    )
    outline, ratio = _aperture_outline(points_mm)
    axis.plot(
        outline[:, 0], outline[:, 1], "-", lw=1.0, color=INK_MUTED, alpha=0.5, zorder=1
    )
    axis.text(
        0.98,
        0.98,
        f"one-sigma capture spread\naperture ratio {ratio:.2f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        size=7.5,
        color=INK_MUTED,
        linespacing=1.5,
    )

    scatter = axis.scatter(
        points_mm[:, 0],
        points_mm[:, 1],
        c=observed_nt,
        s=70,
        cmap=field_map,
        norm=LogNorm(float(observed_nt.min()), float(observed_nt.max())),
        edgecolors=SURFACE,
        linewidths=1.3,
        zorder=3,
    )

    wire_mm = 1000.0 * result.fit.wire_position_m()
    core_mm = 1000.0 * result.core_m
    if core_mm > 0.0:
        turn = np.linspace(0.0, 2.0 * math.pi, 121)
        axis.plot(
            wire_mm[0] + core_mm * np.cos(turn),
            wire_mm[1] + core_mm * np.sin(turn),
            "-",
            lw=0.9,
            color=STATUS[level],
            alpha=0.55,
            zorder=4,
        )
    axis.plot(
        [wire_mm[0]],
        [wire_mm[1]],
        marker="P",
        ms=14.0,
        color=SURFACE,
        markeredgewidth=0.0,
        zorder=5,
    )
    axis.plot(
        [wire_mm[0]],
        [wire_mm[1]],
        marker="P",
        ms=10.0,
        color=STATUS[level],
        markeredgewidth=0.0,
        zorder=6,
    )
    axis.annotate(
        "fitted cable",
        (float(wire_mm[0]), float(wire_mm[1])),
        xytext=(11.0, 0.0),
        textcoords="offset points",
        ha="left",
        va="center",
        size=8.0,
        weight="bold",
        color=STATUS[level],
    )

    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, zorder=0)
    axis.set(
        title="Aperture and fitted cable",
        xlabel="horizontal [mm]",
        ylabel="vertical [mm]",
    )
    figure.colorbar(
        scatter, ax=axis, label="observed |B| [nT]", fraction=0.05, pad=0.02
    )


def _residual_panel(
    axis: _Axis,
    distance_mm: FloatArray,
    residual: NDArray[np.complex128],
    reduced_chi_square: float,
    dof: int,
) -> None:
    left = float(distance_mm.min()) * 0.7
    right = float(distance_mm.max()) * 1.18
    for band, shade in ((2.0, "#f1f0ea"), (1.0, "#e3e2da")):
        axis.fill_between([left, right], -band, band, color=shade, lw=0.0, zorder=0)
    axis.axhline(0.0, lw=0.9, color=AXIS_RULE, zorder=1)
    for index in range(residual.shape[1]):
        column = residual[:, index]
        axis.scatter(
            np.tile(distance_mm, 2),
            np.concatenate((column.real, column.imag)),
            s=24,
            color=_AXIS_COLOURS[index],
            alpha=0.85,
            edgecolors=SURFACE,
            linewidths=0.5,
            zorder=3,
            label=_AXIS_LABELS[index],
        )
    axis.set_xscale("log")
    axis.set_xlim(left, right)
    reach = float(np.abs(residual).max()) if residual.size else 1.0
    axis.set_ylim(-max(2.6, reach * 1.25), max(2.6, reach * 1.25))
    axis.grid(True, axis="x", zorder=0)
    axis.text(
        0.985,
        0.04,
        f"reduced chi-square {reduced_chi_square:.2f}   dof {dof}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        size=8.0,
        color=INK_SOFT,
    )
    axis.set(
        title="Normalised residual",
        xlabel="distance from fitted cable [mm]",
        ylabel=r"(observed - model) / $\sigma$",
    )
    axis.legend(loc="upper left", ncol=3)


def make_figure(result: core.FitResult, health: str) -> Figure:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    fitted = result.fit
    badge = _HEALTH_BADGE.get(
        health, (health.replace("_", " ").upper(), "warning")
    )
    observed = np.linalg.norm(result.observed_nt, axis=1)
    residual = (
        result.observed_nt - fitted.capture_gauges[:, None] * fitted.prediction_nt
    ) / result.sigma_nt[:, None]
    wire = fitted.wire_position_m()
    distance = 1000.0 * np.hypot(
        result.horizontal_m - wire[0], result.vertical_m - wire[1]
    )
    dof = fitted.residual_dof
    reduced = fitted.weighted_rss / dof if dof > 0 else math.inf

    plotting = cast(_Plotting, plt)
    with plotting.rc_context(_STYLE):
        figure = Figure(figsize=(13.5, 8.8), constrained_layout=True)
        axes = figure.subplots(2, 2)
        radial = cast(_Axis, axes[0, 0])
        geometry = cast(_Axis, axes[0, 1])
        angular = cast(_Axis, axes[1, 0])
        residual_axis = cast(_Axis, axes[1, 1])
        operations = cast(_FigureOperations, figure)
        _fit_quality_panels(radial, angular, result)
        _geometry_panel(operations, geometry, result, observed, badge[1])
        _residual_panel(residual_axis, distance, residual, reduced, dof)
        badge_colour = _BADGE_COLOUR.get(health, INK_MUTED)
        prefix = f"{result.collection_name}    WireCheck"
        title = operations.suptitle(
            prefix,
            x=0.006,
            ha="left",
            size=12.0,
            weight="bold",
            color=INK,
        )
        badge_artist = operations.text(
            0.006,
            0.98,
            badge[0],
            ha="left",
            va="top",
            size=12.0,
            weight="bold",
            color=badge_colour,
        )

        figure.draw_without_rendering()
        title_right = float(
            figure.transFigure.inverted().transform(
                (title.get_window_extent().x1, 0.0)
            )[0]
        )
        gap = 12.0 / (72.0 * float(figure.get_figwidth()))
        badge_artist.set_position((title_right + gap, float(title.get_position()[1])))
    return figure
