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



def z2s(z_rho, X, Y, Z):
    """
    Find s-level and coefficients for vertical interpolation.

    :param z_rho: 3D array with vertical s-coordinate structure at rho-points.
    :param X: 1D array, horizontal position in grid coordinates.
    :param Y: 1D array, horizontal position in grid coordinates.
    :param Z: 1D array, particle depth [meters, positive].

    :return:  (K, A), where K is integer and A is float, both 1D arrays.

    Notes
    -----
    With:

    * ``1 <= K < kmax = z_rho.shape[0]``
    * ``z_rho[K-1] < -Z < z_rho[K]`` for ``1 < K < kmax - 1``
    * ``-Z < z_rho[1]`` for ``K = 1``
    * ``z_rho[-1] < -Z`` for ``K = kmax - 1``
    * ``0.0 <= A <= 1``

    Interior linear interpolation::

        A * z_rho[K - 1] + (1 - A) * z_rho[K] = -Z
        for z_rho[0] < -Z < z_rho[-1]

    Extend constant below lowest::

        A * z_rho[K - 1] + (1 - A) * z_rho[K] = z_rho[0]
        for -Z < z_rho[0]  (K=1, A=1)

    Extend constant above highest::

        A * z_rho[K - 1] + (1 - A) * z_rho[K] = z_rho[-1]
        for -Z > z_rho[-1]  (K=kmax-1, A=0)
    """

    kmax = z_rho.shape[0]  # Number of vertical levels

    # Find rho-based horizontal grid cell (rho-point)
    I = np.around(X).astype("int")
    J = np.around(Y).astype("int")

    # Constrain to valid indices
    I = np.minimum(np.maximum(I, 0), z_rho.shape[-1] - 1)
    J = np.minimum(np.maximum(J, 0), z_rho.shape[-2] - 1)

    # Vectorized searchsorted
    K = np.sum(z_rho[:, J, I] < -Z, axis=0)
    K = K.clip(1, kmax - 1)

    A = (z_rho[K, J, I] + Z) / (z_rho[K, J, I] - z_rho[K - 1, J, I])
    A = A.clip(0, 1)  # Extend constantly

    return K, A


# Dense-array reference implementations of 3-D sampling, from the original
# ROMS module. Not used by Forcing, which samples through the chunk loader.
# With (K, A) from z2s, sample3D(method="bilinear") equals linear
# interpolation at the fractional s-level K - A. sample3DUV samples u and v
# on their staggered grid points.


def sample3D(F, X, Y, K, A, method="bilinear"):
    """
    Sample a 3D field on the (sub)grid.

    :param F: 3D field.
    :param S: Depth structure matrix.
    :param X: 1D array of horizontal grid coordinates.
    :param Y: 1D array of horizontal grid coordinates.
    :param Z: 1D array of depth [m, positive downwards].
    :param interpolation: Interpolation method.
        - ``'bilinear'`` for trilinear interpolation.
        - ``'nearest'`` for value in 3D grid cell.

    :return: Sampled values on the (sub)grid.

    Notes
    -----
    Everything is in rho-points.

    Shapes:

    * ``F.shape = (kmax, jmax, imax)``
    * ``S.shape = (kmax, jmax, imax)``
    * ``X.shape = (pmax,)``
    * ``Y.shape = (pmax,)``
    * ``Z.shape = (pmax,)``
    """

    if method == "bilinear":
        # Find rho-point as lower left corner
        I = X.astype("int")
        J = Y.astype("int")

        # Constrain to valid indices
        I = np.minimum(np.maximum(I, 0), F.shape[-1] - 2)
        J = np.minimum(np.maximum(J, 0), F.shape[-2] - 2)

        P = X - I
        Q = Y - J
        W000 = (1 - P) * (1 - Q) * (1 - A)
        W010 = (1 - P) * Q * (1 - A)
        W100 = P * (1 - Q) * (1 - A)
        W110 = P * Q * (1 - A)
        W001 = (1 - P) * (1 - Q) * A
        W011 = (1 - P) * Q * A
        W101 = P * (1 - Q) * A
        W111 = P * Q * A

        return (
            W000 * F[K, J, I]
            + W010 * F[K, J + 1, I]
            + W100 * F[K, J, I + 1]
            + W110 * F[K, J + 1, I + 1]
            + W001 * F[K - 1, J, I]
            + W011 * F[K - 1, J + 1, I]
            + W101 * F[K - 1, J, I + 1]
            + W111 * F[K - 1, J + 1, I + 1]
        )

    # else:  method == 'nearest'
    I = X.round().astype("int")
    J = Y.round().astype("int")

    # Constrain to valid indices
    I = np.minimum(np.maximum(I, 0), F.shape[-1] - 1)
    J = np.minimum(np.maximum(J, 0), F.shape[-2] - 1)

    # Below the lowest level (A == 1), use the lowest level
    K = K - (A == 1)

    return F[K, J, I]


def sample3DUV(U, V, X, Y, K, A, method="bilinear"):
    return (
        sample3D(U, X + 0.5, Y, K, A, method=method),
        sample3D(V, X, Y + 0.5, K, A, method=method),
    )


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
def _z2s(X, Y, Z, i0, j0, H, C, S, hc, vtransform, monotonic, frac, K, A, J, I):
    """K, A as in z2s (or A = K - A if frac).

    Cell mode (J and I have the same size as X): K is the level just above
    the particle, or the lowest level if the particle is below it, and the
    (global) indices of the water column are stored in J and I. A is not set.
    """
    jmax, imax = H.shape
    kmax = C.size
    cell = I.size == X.size
    for p in prange(X.size):
        # Nearest rho point, rounded in subgrid coordinates as z2s
        i = _near(X[p] - i0, imax)
        j = _near(Y[p] - j0, jmax)
        if cell:
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
        if cell:
            K[p] = min(k, kmax - 1)
            continue
        k = min(max(k, 1), kmax - 1)
        zk = _zlevel(k, h, C, S, hc, vtransform)
        zk1 = _zlevel(k - 1, h, C, S, hc, vtransform)
        a = (zk + Z[p]) / (zk - zk1)
        K[p] = k
        a = min(max(a, 0.0), 1.0)
        A[p] = k - a if frac else a


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
    """Vertical s-coordinate of a (sub)grid, at rho or w levels

    :param H: Bottom depth, 2-D subgrid array
    :param C: Stretching curve (Cs_r for rho levels, Cs_w for w levels)
    :param hc: Critical depth
    :param vtransform: 1 (Song and Haidvogel) or 2 (Shchepetkin)
    :param i0, j0: Global indices of subgrid point [0, 0]
    :param stagger: "rho" (levels in the middle of the layers) or "w" (levels
        at the layer interfaces, from the bottom to the surface)
    """

    def __init__(self, H, C, hc, vtransform, i0=0, j0=0, stagger="rho"):
        if int(vtransform) not in (1, 2):
            raise ValueError("Unknown Vtransform")
        self.H = np.ascontiguousarray(H, dtype=np.float64)
        self.C = np.ascontiguousarray(C, dtype=np.float64)
        n = len(self.C)
        if stagger == "rho":
            self.S = -1.0 + (0.5 + np.arange(n)) / n
        elif stagger == "w":
            self.S = np.linspace(-1.0, 0.0, n)
        else:
            raise ValueError("stagger must be 'rho' or 'w'")
        self.stagger = stagger
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
        """s-level and weight for vertical interpolation (see z2s)

        :return: (K, A), integer and float arrays with 1 <= K <= N-1 and
            0 <= A <= 1, such that -Z = A * z[K-1] + (1-A) * z[K] in the
            nearest water column, where z are the level depths (with constant
            extension outside the levels).
        """
        K, A, _, _ = self._z2s(X, Y, Z, cell=False, frac=False)
        return K, A

    def cell(self, X, Y, Z):
        """Grid cell of the particles

        :return: (K, J, I), integer arrays. K is the s-level just above the
            particle, or the lowest level if the particle is below it
            (constant extrapolation). J and I are the global indices of the
            nearest rho point within the subgrid.
        """
        K, _, J, I = self._z2s(X, Y, Z, cell=True, frac=False)
        return K, J, I

    def _z2s(self, X, Y, Z, cell, frac):
        X, Y, Z = (np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
                   for v in np.broadcast_arrays(X, Y, Z))
        n = X.size if cell else 0
        K = np.empty(X.size, dtype=np.int64)
        A = np.empty(X.size, dtype=np.float64)
        J = np.empty(n, dtype=np.int64)
        I = np.empty(n, dtype=np.int64)
        _z2s(X, Y, Z, self.i0, self.j0, self.H, self.C, self.S, self.hc,
             self.vtransform, self.monotonic, frac, K, A, J, I)
        return K, A, J, I

    def level(self, X, Y, Z):
        """Fractional s-level index of depth Z (0 = lowest level)

        Linear interpolation between levels floor(k) and floor(k)+1 at this
        index gives the same result as the (K, A) weights from z2s.
        """
        return self._z2s(X, Y, Z, cell=False, frac=True)[1]


# --------------------------------------------------------------------------
# Horizontal
# --------------------------------------------------------------------------

@numba.njit(parallel=True, nogil=True, cache=True)
def _nearest_cell(X, Y, i0, j0, imax, jmax, J, I):
    for p in prange(X.size):
        I[p] = _near(X[p] - i0, imax) + i0
        J[p] = _near(Y[p] - j0, jmax) + j0


def nearest_cell(X, Y, i0, j0, shape):
    """Global indices (J, I) of the nearest rho point within a subgrid

    The coordinates are rounded (half to even) in subgrid coordinates, as in
    SCoordinate.cell.

    :param shape: (jmax, imax) of the subgrid
    """
    X, Y = (np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
            for v in np.broadcast_arrays(X, Y))
    J = np.empty(X.size, dtype=np.int64)
    I = np.empty(X.size, dtype=np.int64)
    jmax, imax = shape
    _nearest_cell(X, Y, i0, j0, imax, jmax, J, I)
    return J, I


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
