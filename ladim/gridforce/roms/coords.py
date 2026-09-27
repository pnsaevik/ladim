"""
Coordinate conversion for ROMS grids (no forcing data involved).

* Vertical: particle depth -> s-level index and weight, computed column by
  column from the bottom depth and the stretching curve, without the full
  3-D depth array.
* Horizontal: nearest grid cell lookups and bilinear interpolation of static
  2-D fields (e.g. longitude and latitude).

Horizontal particle positions X, Y are global grid coordinates (rho points
at integer values). The 2-D arrays are subgrid arrays, where index [0, 0]
corresponds to the global point (j0, i0).
"""

import numba
import numpy as np
from numba import prange


# --------------------------------------------------------------------------
# Vertical
# --------------------------------------------------------------------------

def s_stretch(N, theta_s, theta_b, stagger="rho", Vstretching=1):
    """
    Compute an s-level stretching array.

    :param int N: Number of vertical levels.
    :param float theta_s: Surface stretching factor.
    :param float theta_b: Bottom stretching factor.
    :param str stagger: Grid staggering option ("rho" or "w").
    :param int Vstretching: Stretching function choice (1, 2, 3, 4, or 5).
    :return: Computed s-level stretching array.
    """

    # if stagger == "rho":
    #     S = -1.0 + (0.5 + np.arange(N)) / N
    # elif stagger == "w":
    #     S = np.linspace(-1.0, 0.0, N + 1)
    if stagger == "rho":
        K = np.arange(0.5, N)
    elif stagger == "w":
        K = np.arange(N + 1)
    else:
        raise ValueError("stagger must be 'rho' or 'w'")
    S = -1 + K / N

    if Vstretching == 1:
        cff1 = 1.0 / np.sinh(theta_s)
        cff2 = 0.5 / np.tanh(0.5 * theta_s)
        return (1.0 - theta_b) * cff1 * np.sinh(theta_s * S) + theta_b * (
            cff2 * np.tanh(theta_s * (S + 0.5)) - 0.5
        )

    elif Vstretching == 2:
        a, b = 1.0, 1.0
        Csur = (1 - np.cosh(theta_s * S)) / (np.cosh(theta_s) - 1)
        Cbot = np.sinh(theta_b * (S + 1)) / np.sinh(theta_b) - 1
        mu = (S + 1) ** a * (1 + (a / b) * (1 - (S + 1) ** b))
        return mu * Csur + (1 - mu) * Cbot

    elif Vstretching == 3:
        gamma_ = 3.0
        Csur = -np.log(np.cosh(gamma_ * (-S) ** theta_s)) / np.log(np.cosh(gamma_))
        Cbot = (
            np.log(np.cosh(gamma_ * (S + 1) ** theta_b)) / np.log(np.cosh(gamma_)) - 1
        )
        mu = 0.5 * (1 - np.tanh(gamma_ * (S + 0.5)))
        return mu * Csur + (1 - mu) * Cbot

    elif Vstretching == 4:
        C = (1 - np.cosh(theta_s * S)) / (np.cosh(theta_s) - 1)
        C = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))
        return C

    elif Vstretching == 5:
        S1 = (K * K - 2 * K * N + K + N * N - N) / (N * N - N)
        S2 = (K * K - K * N) / (1 - N)
        S = -S1 - 0.01 * S2

        C = (1 - np.cosh(theta_s * S)) / (np.cosh(theta_s) - 1)
        C = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))
        return C

    else:
        raise ValueError(f"Unknown Vstretching: {Vstretching}")


def sdepth(H, Hc, C, stagger="rho", Vtransform=1):
    """
    Return depth of rho-points in s-levels.

    :param arraylike H: Bottom depths [meter, positive].
    :param float Hc: Critical depth.
    :param numpy.ndarray cs_r: 1D array of s-level stretching curve.
    :param str stagger: Grid staggering option ("rho" or "w").
    :param int Vtransform: Defines the transform used.
        Defaults to 1 = Song-Haidvogel.

    :return: Depths of mid-points in the s-levels with
        ``ndim = H.ndim + 1`` and
        ``shape = cs_r.shape + H.shape``.
    :rtype: numpy.ndarray

    Typical usage::

        fid = Dataset(roms_file)
        H = fid.variables['h'][:, :]
        C = fid.variables['Cs_r'][:]
        Hc = fid.variables['hc'].getValue()
        z_rho = sdepth(H, Hc, C)
    """
    H = np.asarray(H)
    Hshape = H.shape  # Save the shape of H
    H = H.ravel()  # and make H 1D for easy shape maniplation
    C = np.asarray(C)
    N = len(C)
    outshape = (N,) + Hshape  # Shape of output
    if stagger == "rho":
        S = -1.0 + (0.5 + np.arange(N)) / N  # Unstretched coordinates
    elif stagger == "w":
        S = np.linspace(-1.0, 0.0, N)
    else:
        raise ValueError("stagger must be 'rho' or 'w'")

    if Vtransform == 1:  # Default transform by Song and Haidvogel
        A = Hc * (S - C)[:, None]
        B = np.outer(C, H)
        return (A + B).reshape(outshape)

    if Vtransform == 2:  # New transform by Shchepetkin
        N = Hc * S[:, None] + np.outer(C, H)
        D = 1.0 + Hc / H
        return (N / D).reshape(outshape)

    # else:
    raise ValueError("Unknown Vtransform")



@numba.njit(inline="always")
def _zlevel(k, h, C, S, hc, vtransform):
    """Depth (negative) of s-level k in a column of bottom depth h"""
    if vtransform == 1:
        return hc * (S[k] - C[k]) + C[k] * h
    return (hc * S[k] + C[k] * h) / (1.0 + hc / h)


@numba.njit(inline="always")
def _near(x, n):
    """Nearest index (round half to even), clipped to [0, n-1]"""
    if not x >= 0.0:  # Also catches NaN
        return 0
    if x >= n:
        return n - 1
    return min(int(np.rint(x)), n - 1)


@numba.njit(inline="always")
def _near_global(x, i0, n):
    """Subgrid index of the nearest global grid point, clipped to [0, n-1]"""
    if not x >= i0 - 1.0:  # Also catches NaN
        return 0
    if x >= i0 + n:
        return n - 1
    return min(max(int(np.rint(x)) - i0, 0), n - 1)


@numba.njit(parallel=True, nogil=True, cache=True)
def _z2s(X, Y, Z, i0, j0, H, C, S, hc, vtransform, monotonic, K, A, J, I):
    """K, A as in legacy.z2s. If J and I have the same size as X, the
    (global) indices of the water column are stored there as well."""
    jmax, imax = H.shape
    kmax = C.size
    store_ji = I.size == X.size
    for p in prange(X.size):
        # Nearest rho point, rounded in subgrid coordinates as legacy.z2s
        i = _near(X[p] - i0, imax)
        j = _near(Y[p] - j0, jmax)
        if store_ji:
            I[p] = i + i0
            J[p] = j + j0
        h = H[j, i]
        mz = -Z[p]
        if monotonic:
            # Count levels below -Z, searching down from the surface
            k = kmax
            while k > 0 and not _zlevel(k - 1, h, C, S, hc, vtransform) < mz:
                k -= 1
        else:
            k = 0
            for kk in range(kmax):
                if _zlevel(kk, h, C, S, hc, vtransform) < mz:
                    k += 1
        k = min(max(k, 1), kmax - 1)
        zk = _zlevel(k, h, C, S, hc, vtransform)
        zk1 = _zlevel(k - 1, h, C, S, hc, vtransform)
        a = (zk + Z[p]) / (zk - zk1)
        K[p] = k
        A[p] = min(max(a, 0.0), 1.0)


@numba.njit(parallel=True, cache=True)
def _count_nonmonotonic(H, C, S, hc, vtransform):
    jmax, imax = H.shape
    bad = 0
    for j in prange(jmax):
        for i in range(imax):
            h = H[j, i]
            for k in range(1, C.size):
                if not (_zlevel(k, h, C, S, hc, vtransform)
                        >= _zlevel(k - 1, h, C, S, hc, vtransform)):
                    bad += 1
                    break
    return bad


class SCoordinate:
    """Vertical s-coordinate at rho points of a (sub)grid

    :param H: Bottom depth, 2-D subgrid array
    :param C: Stretching curve at rho points (Cs_r)
    :param hc: Critical depth
    :param vtransform: 1 (Song and Haidvogel) or 2 (Shchepetkin)
    :param i0, j0: Global indices of subgrid point [0, 0]
    """

    def __init__(self, H, C, hc, vtransform, i0=0, j0=0):
        if int(vtransform) not in (1, 2):
            raise ValueError("Unknown Vtransform")
        self.H = np.ascontiguousarray(H, dtype=np.float64)
        self.C = np.ascontiguousarray(C, dtype=np.float64)
        n = len(self.C)
        self.S = -1.0 + (0.5 + np.arange(n)) / n
        self.hc = float(hc)
        self.vtransform = int(vtransform)
        self.i0 = int(i0)
        self.j0 = int(j0)
        self.monotonic = _count_nonmonotonic(
            self.H, self.C, self.S, self.hc, self.vtransform) == 0

    @property
    def N(self):
        return len(self.C)

    def z2s(self, X, Y, Z):
        """s-level and weight for vertical interpolation (see roms.legacy.z2s)

        :return: (K, A), integer and float arrays with 1 <= K <= N-1 and
            0 <= A <= 1, such that -Z = A * z_r[K-1] + (1-A) * z_r[K] in the
            nearest water column (with constant extension outside the levels).
        """
        K, A, _, _ = self._z2s(X, Y, Z, cell=False)
        return K, A

    def cell(self, X, Y, Z):
        """Grid cell of the particles

        :return: (K, J, I), integer arrays. K is the s-level from z2s (the
            level just above the particle), J and I are the global indices of
            the nearest rho point within the subgrid.
        """
        K, _, J, I = self._z2s(X, Y, Z, cell=True)
        return K, J, I

    def _z2s(self, X, Y, Z, cell):
        X, Y, Z = (np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
                   for v in np.broadcast_arrays(X, Y, Z))
        n = X.size if cell else 0
        K = np.empty(X.size, dtype=np.int64)
        A = np.empty(X.size, dtype=np.float64)
        J = np.empty(n, dtype=np.int64)
        I = np.empty(n, dtype=np.int64)
        _z2s(X, Y, Z, self.i0, self.j0, self.H, self.C, self.S, self.hc,
             self.vtransform, self.monotonic, K, A, J, I)
        return K, A, J, I

    def level(self, X, Y, Z):
        """Fractional s-level index of depth Z (0 = lowest rho level)

        Linear interpolation between levels floor(k) and floor(k)+1 at this
        index gives the same result as the (K, A) weights from z2s.
        """
        K, A = self.z2s(X, Y, Z)
        return K - A


# --------------------------------------------------------------------------
# Horizontal
# --------------------------------------------------------------------------

@numba.njit(parallel=True, nogil=True, cache=True)
def _lookup(F, X, Y, i0, j0, imax, jmax, out):
    for p in prange(X.size):
        out[p] = F[_near_global(Y[p], j0, jmax), _near_global(X[p], i0, imax)]


def lookup(F, X, Y, i0=0, j0=0, clip=None):
    """Value of a 2-D subgrid field in the nearest grid cell

    The global coordinates are rounded (half to even) before subtracting the
    subgrid offset, as in the original ROMS module.

    :param F: 2-D subgrid array
    :param X, Y: Global grid coordinates
    :param i0, j0: Global indices of F[0, 0]
    :param clip: Optional (jmax, imax); indices are clipped to [0, max-1].
        Defaults to the shape of F.
    """
    shape = np.broadcast(X, Y).shape
    X, Y = (np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
            for v in np.broadcast_arrays(X, Y))
    jmax, imax = F.shape if clip is None else clip
    out = np.empty(X.size, dtype=F.dtype)
    _lookup(F, X, Y, i0, j0, imax, jmax, out)
    return out.reshape(shape)


@numba.njit(parallel=True, nogil=True, cache=True)
def _bilinear(F, X, Y, out):
    jmax, imax = F.shape
    nbad = 0
    for p in prange(X.size):
        x = X[p]
        y = Y[p]
        if not (0.0 <= x < imax - 1 and 0.0 <= y < jmax - 1):
            nbad += 1
            out[p] = np.nan
            continue
        i = int(x)
        j = int(y)
        u = x - i
        v = y - j
        out[p] = ((1 - u) * (1 - v) * F[j, i] + (1 - u) * v * F[j + 1, i]
                  + u * (1 - v) * F[j, i + 1] + u * v * F[j + 1, i + 1])
    return nbad


def bilinear(F, X, Y):
    """Bilinear interpolation of a 2-D array at local grid coordinates

    Raises ValueError if any point is outside the array (as
    :func:`ladim.sample.sample2D`).
    """
    shape = np.broadcast(X, Y).shape
    X, Y = (np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
            for v in np.broadcast_arrays(X, Y))
    F = np.ascontiguousarray(F, dtype=np.float64)
    out = np.empty(X.size, dtype=np.float64)
    if _bilinear(F, X, Y, out):
        raise ValueError("point outside grid")
    return float(out[0]) if shape == () else out.reshape(shape)


@numba.njit(parallel=True, nogil=True, cache=True)
def _inside(X, Y, xmin, xmax, ymin, ymax, out):
    for p in prange(X.size):
        out[p] = xmin < X[p] < xmax and ymin < Y[p] < ymax


def inside(X, Y, xmin, xmax, ymin, ymax):
    """True for points strictly inside the rectangle"""
    shape = np.broadcast(X, Y).shape
    X, Y = (np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
            for v in np.broadcast_arrays(X, Y))
    out = np.empty(X.size, dtype=np.bool_)
    _inside(X, Y, float(xmin), float(xmax), float(ymin), float(ymax), out)
    return out.reshape(shape)
