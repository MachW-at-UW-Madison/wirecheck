from __future__ import annotations

import math
from typing import Callable, NamedTuple, Protocol, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit as _numba_njit
except ImportError:
    _numba_njit = None

__all__ = ["compiled", "HAVE_NUMBA"]

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]
ResidualFunction = Callable[[FloatArray], FloatArray]
JacobianFunction = Callable[[FloatArray], FloatArray]
CompiledFunctions = tuple[ResidualFunction, JacobianFunction]

_Function = TypeVar("_Function", bound=Callable[..., object])


class _Decorator(Protocol):
    def __call__(self, function: _Function, /) -> _Function: ...


def _identity_decorator(
    function: _Function,
) -> _Function:
    return function


MIN_RADIUS2 = 1e-12

HAVE_NUMBA = _numba_njit is not None
if _numba_njit is not None:
    _jit = cast(_Decorator, _numba_njit(cache=True, nogil=True))
else:
    _jit = cast(_Decorator, _identity_decorator)


@_jit
def capture_terms(
    params: FloatArray,
    ig: np.int64,
    it: np.int64,
    ila: np.int64,
    ilc: np.int64,
    ips: np.int64,
    ix0: np.int64,
    iy0: np.int64,
    iur: IntArray,
    iui: IntArray,
    horizontal_m: FloatArray,
    vertical_m: FloatArray,
    b0: FloatArray,
    b1: FloatArray,
    core2: float,
    half_spacing: float,
    inverse_spacing: float,
    derivatives: bool,
    pred_re: FloatArray,
    pred_im: FloatArray,
    d_re: FloatArray,
    d_im: FloatArray,
) -> None:
    gamma = params[ig]
    theta = params[it]
    amp = math.exp(params[ila])
    lp_re = params[ilc]
    lp_im = params[ips]
    cg = math.cos(gamma)
    sg = math.sin(gamma)
    ox = half_spacing * math.cos(theta)
    oy = half_spacing * math.sin(theta)
    scale = amp * inverse_spacing
    n = horizontal_m.shape[0]
    for i in range(n):
        dx = horizontal_m[i] - params[ix0]
        dy = vertical_m[i] - params[iy0]
        rho2 = dx * dx + dy * dy
        r2 = rho2 + core2
        if r2 < MIN_RADIUS2:
            r2 = MIN_RADIUS2
        lx = dx - ox
        ly = dy - oy
        rx = dx + ox
        ry = dy + oy
        lr = lx * lx + ly * ly + core2
        rr = rx * rx + ry * ry + core2
        if lr < MIN_RADIUS2:
            lr = MIN_RADIUS2
        if rr < MIN_RADIUS2:
            rr = MIN_RADIUS2
        phi = math.atan2(dy, dx)
        monopole = 1.0 / math.sqrt(r2)
        sp = math.sin(phi)
        cp = math.cos(phi)
        mx = -monopole * sp
        my = monopole * cp
        pair_x = scale * (ry / rr - ly / lr)
        pair_y = scale * (lx / lr - rx / rr)
        bx_re = pair_x + lp_re * mx
        bx_im = lp_im * mx
        by_re = pair_y + lp_re * my
        by_im = lp_im * my
        px_re = cg * bx_re - sg * by_re
        px_im = cg * bx_im - sg * by_im
        py_re = sg * bx_re + cg * by_re
        py_im = sg * bx_im + cg * by_im
        for axis in range(3):
            pred_re[i, axis] = px_re * b0[axis] + py_re * b1[axis] + params[iur[axis]]
            pred_im[i, axis] = px_im * b0[axis] + py_im * b1[axis] + params[iui[axis]]
        if not derivatives:
            continue
        if rho2 > 0.0:
            dphi_x = dy / rho2
            dphi_y = -dx / rho2
        else:
            dphi_x = 0.0
            dphi_y = 0.0
        dm_x = monopole * dx / r2
        dm_y = monopole * dy / r2
        lr2 = lr * lr
        rr2 = rr * rr
        pair_x_x0 = scale * 2.0 * (rx * ry / rr2 - lx * ly / lr2)
        pair_y_x0 = scale * (
            2.0 * lx * lx / lr2 - 1.0 / lr + 1.0 / rr - 2.0 * rx * rx / rr2
        )
        pair_x_y0 = scale * (
            1.0 / lr - 2.0 * ly * ly / lr2 - 1.0 / rr + 2.0 * ry * ry / rr2
        )
        pair_y_y0 = -pair_x_x0
        dlr_t = 2.0 * (lx * oy - ly * ox)
        drr_t = 2.0 * (ry * ox - rx * oy)
        pair_x_t = scale * (ox / lr + ly * dlr_t / lr2 + ox / rr - ry * drr_t / rr2)
        pair_y_t = scale * (oy / lr - lx * dlr_t / lr2 + oy / rr + rx * drr_t / rr2)
        for term in range(7):
            column = -1
            dbx_re = 0.0
            dbx_im = 0.0
            dby_re = 0.0
            dby_im = 0.0
            rotate_only = False
            if term == 0:
                column = ila
                dbx_re = pair_x
                dby_re = pair_y
            elif term == 1:
                column = ilc
                dbx_re = mx
                dby_re = my
            elif term == 2:
                column = it
                dbx_re = pair_x_t
                dby_re = pair_y_t
            elif term == 3:
                column = ips
                dbx_im = mx
                dby_im = my
            elif term == 4:
                column = ig
                rotate_only = True
            elif term == 5:
                column = ix0
                dbx_re = pair_x_x0 - lp_re * (dm_x * sp + monopole * cp * dphi_x)
                dbx_im = -lp_im * (dm_x * sp + monopole * cp * dphi_x)
                dby_re = pair_y_x0 + lp_re * (dm_x * cp - monopole * sp * dphi_x)
                dby_im = lp_im * (dm_x * cp - monopole * sp * dphi_x)
            elif term == 6:
                column = iy0
                dbx_re = pair_x_y0 - lp_re * (dm_y * sp + monopole * cp * dphi_y)
                dbx_im = -lp_im * (dm_y * sp + monopole * cp * dphi_y)
                dby_re = pair_y_y0 + lp_re * (dm_y * cp - monopole * sp * dphi_y)
                dby_im = lp_im * (dm_y * cp - monopole * sp * dphi_y)
            if rotate_only:
                dpx_re = -py_re
                dpx_im = -py_im
                dpy_re = px_re
                dpy_im = px_im
            else:
                dpx_re = cg * dbx_re - sg * dby_re
                dpx_im = cg * dbx_im - sg * dby_im
                dpy_re = sg * dbx_re + cg * dby_re
                dpy_im = sg * dbx_im + cg * dby_im
            for axis in range(3):
                d_re[i, column, axis] = dpx_re * b0[axis] + dpy_re * b1[axis]
                d_im[i, column, axis] = dpx_im * b0[axis] + dpy_im * b1[axis]
        for component in range(3):
            d_re[i, iur[component], component] = 1.0
            d_im[i, iui[component], component] = 1.0


@_jit
def _cross(
    re: FloatArray, im: FloatArray, obs_re: FloatArray, obs_im: FloatArray, i: int
) -> tuple[float, float]:
    cross_re = 0.0
    cross_im = 0.0
    for axis in range(3):
        cross_re += re[axis] * obs_re[i, axis] + im[axis] * obs_im[i, axis]
        cross_im += re[axis] * obs_im[i, axis] - im[axis] * obs_re[i, axis]
    return cross_re, cross_im


@_jit
def residual_from_terms(
    pred_re: FloatArray,
    pred_im: FloatArray,
    obs_re: FloatArray,
    obs_im: FloatArray,
    inv_sig: FloatArray,
    shared: bool,
    out: FloatArray,
) -> FloatArray:
    n = pred_re.shape[0]
    g_re = 1.0
    g_im = 0.0
    if shared:
        total_re = 0.0
        total_im = 0.0
        for i in range(n):
            cross_re, cross_im = _cross(pred_re[i], pred_im[i], obs_re, obs_im, i)
            w2 = inv_sig[i] * inv_sig[i]
            total_re += w2 * cross_re
            total_im += w2 * cross_im
        magnitude = math.sqrt(total_re * total_re + total_im * total_im)
        if magnitude > 1e-15:
            g_re = total_re / magnitude
            g_im = total_im / magnitude
    for i in range(n):
        if not shared:
            cross_re, cross_im = _cross(pred_re[i], pred_im[i], obs_re, obs_im, i)
            magnitude = math.sqrt(cross_re * cross_re + cross_im * cross_im)
            if magnitude > 1e-15:
                g_re = cross_re / magnitude
                g_im = cross_im / magnitude
            else:
                g_re = 1.0
                g_im = 0.0
        w = inv_sig[i]
        real_row = i * 3
        imag_row = 3 * n + i * 3
        for axis in range(3):
            gr = g_re * pred_re[i, axis] - g_im * pred_im[i, axis]
            gi = g_re * pred_im[i, axis] + g_im * pred_re[i, axis]
            out[real_row + axis] = (obs_re[i, axis] - gr) * w
            out[imag_row + axis] = (obs_im[i, axis] - gi) * w
    return out


@_jit
def jacobian_from_terms(
    pred_re: FloatArray,
    pred_im: FloatArray,
    d_re: FloatArray,
    d_im: FloatArray,
    obs_re: FloatArray,
    obs_im: FloatArray,
    inv_sig: FloatArray,
    shared: bool,
    jac: FloatArray,
) -> FloatArray:
    n = pred_re.shape[0]
    columns = d_re.shape[1]
    phase_derivative = np.zeros(columns, dtype=np.float64)
    g_re = 1.0
    g_im = 0.0
    if shared:
        total_re = 0.0
        total_im = 0.0
        d_total_re = np.zeros(columns, dtype=np.float64)
        d_total_im = np.zeros(columns, dtype=np.float64)
        for i in range(n):
            w2 = inv_sig[i] * inv_sig[i]
            cross_re, cross_im = _cross(pred_re[i], pred_im[i], obs_re, obs_im, i)
            total_re += w2 * cross_re
            total_im += w2 * cross_im
            for column in range(columns):
                dc_re, dc_im = _cross(d_re[i, column], d_im[i, column], obs_re, obs_im, i)
                d_total_re[column] += w2 * dc_re
                d_total_im[column] += w2 * dc_im
        magnitude = math.sqrt(total_re * total_re + total_im * total_im)
        if magnitude > 1e-15:
            g_re = total_re / magnitude
            g_im = total_im / magnitude
            for column in range(columns):
                phase_derivative[column] = (
                    total_re * d_total_im[column] - total_im * d_total_re[column]
                ) / (magnitude * magnitude)
    for i in range(n):
        if not shared:
            cross_re, cross_im = _cross(pred_re[i], pred_im[i], obs_re, obs_im, i)
            magnitude = math.sqrt(cross_re * cross_re + cross_im * cross_im)
            if magnitude > 1e-15:
                g_re = cross_re / magnitude
                g_im = cross_im / magnitude
            else:
                g_re = 1.0
                g_im = 0.0
            for column in range(columns):
                phase_derivative[column] = 0.0
                if magnitude > 1e-15:
                    dc_re, dc_im = _cross(d_re[i, column], d_im[i, column], obs_re, obs_im, i)
                    phase_derivative[column] = (
                        cross_re * dc_im - cross_im * dc_re
                    ) / (magnitude * magnitude)
        w = inv_sig[i]
        real_row = i * 3
        imag_row = 3 * n + i * 3
        for column in range(columns):
            for axis in range(3):
                aligned_re = d_re[i, column, axis] - pred_im[i, axis] * phase_derivative[column]
                aligned_im = d_im[i, column, axis] + pred_re[i, axis] * phase_derivative[column]
                jac[real_row + axis, column] = -(g_re * aligned_re - g_im * aligned_im) * w
                jac[imag_row + axis, column] = -(g_re * aligned_im + g_im * aligned_re) * w
    return jac


class _Geometry(NamedTuple):
    gamma_index: np.int64
    theta_index: np.int64
    log_amplitude_index: np.int64
    leak_inphase_index: np.int64
    leak_quadrature_index: np.int64
    horizontal_position_index: np.int64
    vertical_position_index: np.int64
    coherent_real_indices: IntArray
    coherent_imag_indices: IntArray
    horizontal_m: FloatArray
    vertical_m: FloatArray
    basis_0: FloatArray
    basis_1: FloatArray
    core_squared_m2: float
    half_pair_spacing_m: float
    inverse_pair_spacing_m: float


class _Observations(NamedTuple):
    observed_real_nt: FloatArray
    observed_imag_nt: FloatArray
    inverse_sigma_nt: FloatArray
    shared_reference: bool


_PARAMETER_COUNT = 13
_GAMMA = np.int64(0)
_THETA = np.int64(1)
_LOG_AMPLITUDE = np.int64(2)
_LEAK_INPHASE = np.int64(3)
_LEAK_QUADRATURE = np.int64(4)
_HORIZONTAL_POSITION = np.int64(5)
_VERTICAL_POSITION = np.int64(6)
_COHERENT_REAL = np.array((7, 8, 9), dtype=np.int64)
_COHERENT_IMAG = np.array((10, 11, 12), dtype=np.int64)


def _kernel_args(
    basis: FloatArray,
    horizontal_m: FloatArray,
    vertical_m: FloatArray,
    observed_nt: ComplexArray,
    sigma_nt: FloatArray,
    core_m: float,
    pair_spacing_m: float,
    shared_reference: bool,
) -> tuple[_Geometry, _Observations]:
    geometry = _Geometry(
        _GAMMA,
        _THETA,
        _LOG_AMPLITUDE,
        _LEAK_INPHASE,
        _LEAK_QUADRATURE,
        _HORIZONTAL_POSITION,
        _VERTICAL_POSITION,
        _COHERENT_REAL,
        _COHERENT_IMAG,
        np.ascontiguousarray(horizontal_m, dtype=np.float64),
        np.ascontiguousarray(vertical_m, dtype=np.float64),
        np.ascontiguousarray(basis[:, 0], dtype=np.float64),
        np.ascontiguousarray(basis[:, 1], dtype=np.float64),
        float(core_m) ** 2,
        0.5 * float(pair_spacing_m),
        1.0 / float(pair_spacing_m),
    )
    observations = _Observations(
        np.ascontiguousarray(observed_nt.real, dtype=np.float64),
        np.ascontiguousarray(observed_nt.imag, dtype=np.float64),
        np.ascontiguousarray(1.0 / sigma_nt, dtype=np.float64),
        bool(shared_reference),
    )
    return geometry, observations


def compiled(
    basis: FloatArray,
    *,
    horizontal_m: FloatArray,
    vertical_m: FloatArray,
    observed_nt: ComplexArray,
    sigma_nt: FloatArray,
    core_m: float,
    pair_spacing_m: float,
    shared_reference: bool = False,
) -> CompiledFunctions | None:
    if not HAVE_NUMBA:
        return None
    geometry, observations = _kernel_args(
        basis,
        horizontal_m,
        vertical_m,
        observed_nt,
        sigma_nt,
        core_m,
        pair_spacing_m,
        shared_reference,
    )
    n = horizontal_m.size
    rows = 6 * n
    columns = _PARAMETER_COUNT
    no_derivatives = np.zeros((0, 0, 3), dtype=np.float64)

    def residual(parameters: FloatArray) -> FloatArray:
        pred_re = np.empty((n, 3), dtype=np.float64)
        pred_im = np.empty((n, 3), dtype=np.float64)
        capture_terms(
            parameters, *geometry, False, pred_re, pred_im, no_derivatives, no_derivatives
        )
        return residual_from_terms(
            pred_re, pred_im, *observations, np.empty(rows, dtype=np.float64)
        )

    def jacobian(parameters: FloatArray) -> FloatArray:
        pred_re = np.empty((n, 3), dtype=np.float64)
        pred_im = np.empty((n, 3), dtype=np.float64)
        d_re = np.zeros((n, columns, 3), dtype=np.float64)
        d_im = np.zeros((n, columns, 3), dtype=np.float64)
        capture_terms(parameters, *geometry, True, pred_re, pred_im, d_re, d_im)
        return jacobian_from_terms(
            pred_re,
            pred_im,
            d_re,
            d_im,
            *observations,
            np.empty((rows, columns), dtype=np.float64),
        )

    return residual, jacobian
