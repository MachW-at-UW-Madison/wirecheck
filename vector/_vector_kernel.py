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
def residual_full(
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
    obs_re: FloatArray,
    obs_im: FloatArray,
    inv_sig: FloatArray,
    b0: FloatArray,
    b1: FloatArray,
    core2: float,
    half_spacing: float,
    inverse_spacing: float,
    out: FloatArray,
) -> FloatArray:
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
        r2 = dx * dx + dy * dy + core2
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
        bx_re = scale * (ry / rr - ly / lr)
        bx_im = 0.0
        by_re = scale * (lx / lr - rx / rr)
        by_im = 0.0
        monopole = 1.0 / math.sqrt(r2)
        mx = -monopole * math.sin(phi)
        my = monopole * math.cos(phi)
        bx_re += lp_re * mx
        bx_im += lp_im * mx
        by_re += lp_re * my
        by_im += lp_im * my
        px_re = cg * bx_re - sg * by_re
        px_im = cg * bx_im - sg * by_im
        py_re = sg * bx_re + cg * by_re
        py_im = sg * bx_im + cg * by_im
        p0r = px_re * b0[0] + py_re * b1[0]
        p1r = px_re * b0[1] + py_re * b1[1]
        p2r = px_re * b0[2] + py_re * b1[2]
        p0i = px_im * b0[0] + py_im * b1[0]
        p1i = px_im * b0[1] + py_im * b1[1]
        p2i = px_im * b0[2] + py_im * b1[2]
        p0r += params[iur[0]]
        p1r += params[iur[1]]
        p2r += params[iur[2]]
        p0i += params[iui[0]]
        p1i += params[iui[1]]
        p2i += params[iui[2]]
        cross_re = (
            p0r * obs_re[i, 0]
            + p0i * obs_im[i, 0]
            + p1r * obs_re[i, 1]
            + p1i * obs_im[i, 1]
            + p2r * obs_re[i, 2]
            + p2i * obs_im[i, 2]
        )
        cross_im = (
            p0r * obs_im[i, 0]
            - p0i * obs_re[i, 0]
            + p1r * obs_im[i, 1]
            - p1i * obs_re[i, 1]
            + p2r * obs_im[i, 2]
            - p2i * obs_re[i, 2]
        )
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
        gr = g_re * p0r - g_im * p0i
        gi = g_re * p0i + g_im * p0r
        out[real_row] = (obs_re[i, 0] - gr) * w
        out[imag_row] = (obs_im[i, 0] - gi) * w
        gr = g_re * p1r - g_im * p1i
        gi = g_re * p1i + g_im * p1r
        out[real_row + 1] = (obs_re[i, 1] - gr) * w
        out[imag_row + 1] = (obs_im[i, 1] - gi) * w
        gr = g_re * p2r - g_im * p2i
        gi = g_re * p2i + g_im * p2r
        out[real_row + 2] = (obs_re[i, 2] - gr) * w
        out[imag_row + 2] = (obs_im[i, 2] - gi) * w
    return out


@_jit
def jacobian_full(
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
    obs_re: FloatArray,
    obs_im: FloatArray,
    inv_sig: FloatArray,
    b0: FloatArray,
    b1: FloatArray,
    core2: float,
    half_spacing: float,
    inverse_spacing: float,
    jac: FloatArray,
) -> FloatArray:
    gamma = params[ig]
    theta = params[it]
    amp = math.exp(params[ila])
    lp_re = params[ilc]
    lp_im = params[ips]
    cg = math.cos(gamma)
    sg = math.sin(gamma)
    jac[:, :] = 0.0
    n = horizontal_m.shape[0]
    pred_re = np.empty(3, dtype=np.float64)
    pred_im = np.empty(3, dtype=np.float64)
    deriv_re = np.empty(3, dtype=np.float64)
    deriv_im = np.empty(3, dtype=np.float64)
    ox = half_spacing * math.cos(theta)
    oy = half_spacing * math.sin(theta)
    scale = amp * inverse_spacing
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
        cross_re = 0.0
        cross_im = 0.0
        for axis in range(3):
            pred_re[axis] = px_re * b0[axis] + py_re * b1[axis]
            pred_im[axis] = px_im * b0[axis] + py_im * b1[axis]
            pred_re[axis] += params[iur[axis]]
            pred_im[axis] += params[iui[axis]]
            cross_re += (
                pred_re[axis] * obs_re[i, axis] + pred_im[axis] * obs_im[i, axis]
            )
            cross_im += (
                pred_re[axis] * obs_im[i, axis] - pred_im[axis] * obs_re[i, axis]
            )
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
            cross_derivative_re = 0.0
            cross_derivative_im = 0.0
            for axis in range(3):
                deriv_re[axis] = dpx_re * b0[axis] + dpy_re * b1[axis]
                deriv_im[axis] = dpx_im * b0[axis] + dpy_im * b1[axis]
                cross_derivative_re += (
                    deriv_re[axis] * obs_re[i, axis] + deriv_im[axis] * obs_im[i, axis]
                )
                cross_derivative_im += (
                    deriv_re[axis] * obs_im[i, axis] - deriv_im[axis] * obs_re[i, axis]
                )
            phase_derivative = 0.0
            if magnitude > 1e-15:
                phase_derivative = (
                    cross_re * cross_derivative_im - cross_im * cross_derivative_re
                ) / (magnitude * magnitude)
            for axis in range(3):
                aligned_re = deriv_re[axis] - pred_im[axis] * phase_derivative
                aligned_im = deriv_im[axis] + pred_re[axis] * phase_derivative
                jac[real_row + axis, column] = (
                    -(g_re * aligned_re - g_im * aligned_im) * w
                )
                jac[imag_row + axis, column] = (
                    -(g_re * aligned_im + g_im * aligned_re) * w
                )
        for component in range(3):
            for quadrature in range(2):
                cross_derivative_re = 0.0
                cross_derivative_im = 0.0
                for axis in range(3):
                    deriv_re[axis] = 0.0
                    deriv_im[axis] = 0.0
                if quadrature == 0:
                    column = iur[component]
                    deriv_re[component] = 1.0
                else:
                    column = iui[component]
                    deriv_im[component] = 1.0
                for axis in range(3):
                    cross_derivative_re += (
                        deriv_re[axis] * obs_re[i, axis]
                        + deriv_im[axis] * obs_im[i, axis]
                    )
                    cross_derivative_im += (
                        deriv_re[axis] * obs_im[i, axis]
                        - deriv_im[axis] * obs_re[i, axis]
                    )
                phase_derivative = 0.0
                if magnitude > 1e-15:
                    phase_derivative = (
                        cross_re * cross_derivative_im - cross_im * cross_derivative_re
                    ) / (magnitude * magnitude)
                for axis in range(3):
                    aligned_re = deriv_re[axis] - pred_im[axis] * phase_derivative
                    aligned_im = deriv_im[axis] + pred_re[axis] * phase_derivative
                    jac[real_row + axis, column] = (
                        -(g_re * aligned_re - g_im * aligned_im) * w
                    )
                    jac[imag_row + axis, column] = (
                        -(g_re * aligned_im + g_im * aligned_re) * w
                    )
    return jac


class _KernelArgs(NamedTuple):
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
    observed_real_nt: FloatArray
    observed_imag_nt: FloatArray
    inverse_sigma_nt: FloatArray
    basis_0: FloatArray
    basis_1: FloatArray
    core_squared_m2: float
    half_pair_spacing_m: float
    inverse_pair_spacing_m: float


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
) -> _KernelArgs:
    return _KernelArgs(
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
        np.ascontiguousarray(observed_nt.real, dtype=np.float64),
        np.ascontiguousarray(observed_nt.imag, dtype=np.float64),
        np.ascontiguousarray(1.0 / sigma_nt, dtype=np.float64),
        np.ascontiguousarray(basis[:, 0], dtype=np.float64),
        np.ascontiguousarray(basis[:, 1], dtype=np.float64),
        float(core_m) ** 2,
        0.5 * float(pair_spacing_m),
        1.0 / float(pair_spacing_m),
    )


def compiled(
    basis: FloatArray,
    *,
    horizontal_m: FloatArray,
    vertical_m: FloatArray,
    observed_nt: ComplexArray,
    sigma_nt: FloatArray,
    core_m: float,
    pair_spacing_m: float,
) -> CompiledFunctions | None:
    if not HAVE_NUMBA:
        return None
    args = _kernel_args(
        basis, horizontal_m, vertical_m, observed_nt, sigma_nt, core_m, pair_spacing_m
    )
    rows = 6 * horizontal_m.size
    columns = _PARAMETER_COUNT

    def residual(parameters: FloatArray) -> FloatArray:
        return residual_full(parameters, *args, np.empty(rows, dtype=np.float64))

    def jacobian(parameters: FloatArray) -> FloatArray:
        return jacobian_full(
            parameters, *args, np.empty((rows, columns), dtype=np.float64)
        )

    return residual, jacobian
