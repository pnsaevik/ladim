"""
Numba kernels of the chunk loader: interpolation, chunk marking and HDF5
byte unshuffling.

Conventions
-----------
Spatial arrays are always 3-D (variables of lower rank are padded with leading
axes of length 1). A position is a set of fractional, grid-native indices
``(k, j, i)``, i.e. ``F[k, j, i]`` is the value at integer coordinates.
Along each axis the interpolation is either

* linear: lower index ``floor(x)`` clipped to ``[0, n-2]``, weight of the
  upper index ``x - lower`` clipped to ``[0, 1]`` (constant extension outside
  the array), or
* nearest: index ``rint(x)`` clipped to ``[0, n-1]``.

The chunks holding the needed grid points must be loaded, as flagged in the
``loaded`` arrays (one entry per chunk). Particles that need a chunk that is
not loaded are flagged in ``miss`` and skipped.
"""

import numba
import numpy as np
from numba import prange


@numba.njit(inline="always")
def _axis(x, n, linear):
    """Lower index, upper index and upper weight along one axis"""
    # Guard against NaN and huge values before converting to int
    if not x >= -1.0:
        x = -1.0
    if x > n:
        x = float(n)
    if linear and n > 1:
        lo = min(max(int(np.floor(x)), 0), n - 2)
        f = min(max(x - lo, 0.0), 1.0)
        return lo, lo + 1, f
    lo = min(max(int(np.rint(x)), 0), n - 1)
    return lo, lo, 0.0


@numba.njit(inline="always")
def _is_loaded(L, k0, k1, j0, j1, i0, i1, cz, cy, cx):
    a0 = k0 // cz
    a1 = k1 // cz
    b0 = j0 // cy
    b1 = j1 // cy
    c0 = i0 // cx
    c1 = i1 // cx
    return (
        L[a0, b0, c0] != 0 and L[a0, b0, c1] != 0
        and L[a0, b1, c0] != 0 and L[a0, b1, c1] != 0
        and L[a1, b0, c0] != 0 and L[a1, b0, c1] != 0
        and L[a1, b1, c0] != 0 and L[a1, b1, c1] != 0
    )


@numba.njit(inline="always")
def _get(A, p):
    """Element p of a coordinate array, broadcasting arrays of length 1"""
    return A[p] if A.size > 1 else A[0]


@numba.njit(parallel=True, nogil=True, cache=True)
def interp(B0, L0, B1, L1, w1, K, J, I, linear, chunks, scale, offset,
           mask, use_mask, out, miss):
    """Interpolate a (time-interpolated) 3-D field at fractional indices

    :param B0, B1: Raw data of the two time frames (same shape)
    :param L0, L1: Loaded-flags of the chunks of B0 and B1
    :param w1: Weight of frame B1 (B1 is not accessed if w1 == 0)
    :param K, J, I: Fractional indices, float64 arrays of length N (or 1)
    :param linear: Boolean array of length 3, linear or nearest for each axis
    :param chunks: Integer array of length 3, the chunk shape
    :param scale, offset: Decoded value = offset + scale * raw
    :param mask: 2-D array over the (j, i) axes. Grid points with mask == 0
        contribute zero to the result. Only used if use_mask is True.
    :param out: Output float64 array of length N
    :param miss: Output uint8 array of length N, set to 1 for particles whose
        chunks are not loaded (their output is not computed)
    :return: The number of particles with missing chunks
    """
    nk, nj, ni = B0.shape
    cz, cy, cx = chunks[0], chunks[1], chunks[2]
    lk, lj, li = linear[0], linear[1], linear[2]
    two = w1 != 0.0
    w0 = 1.0 - w1
    nmiss = 0
    for p in prange(out.size):
        k0, k1, fk = _axis(_get(K, p), nk, lk)
        j0, j1, fj = _axis(_get(J, p), nj, lj)
        i0, i1, fi = _axis(_get(I, p), ni, li)

        ok = _is_loaded(L0, k0, k1, j0, j1, i0, i1, cz, cy, cx)
        if two:
            ok = ok and _is_loaded(L1, k0, k1, j0, j1, i0, i1, cz, cy, cx)
        if not ok:
            miss[p] = 1
            nmiss += 1
            continue
        miss[p] = 0

        acc = 0.0
        for a in range(2):
            k = k1 if a else k0
            wa = fk if a else 1.0 - fk
            if wa == 0.0:
                continue
            for b in range(2):
                j = j1 if b else j0
                wb = wa * (fj if b else 1.0 - fj)
                if wb == 0.0:
                    continue
                for c in range(2):
                    i = i1 if c else i0
                    w = wb * (fi if c else 1.0 - fi)
                    if w == 0.0:
                        continue
                    if use_mask and mask[j, i] == 0:
                        continue
                    r = np.float64(B0[k, j, i])
                    if two:
                        r = w0 * r + w1 * np.float64(B1[k, j, i])
                    acc += w * (offset + scale * r)
        out[p] = acc
    return nmiss


@numba.njit(parallel=True, nogil=True, cache=True)
def mark_chunks(idx, K, J, I, linear, shape, chunks, M):
    """Set M[chunk] = 1 for the chunks needed by particles idx"""
    nk, nj, ni = shape[0], shape[1], shape[2]
    cz, cy, cx = chunks[0], chunks[1], chunks[2]
    lk, lj, li = linear[0], linear[1], linear[2]
    for q in prange(idx.size):
        p = idx[q]
        k0, k1, _ = _axis(_get(K, p), nk, lk)
        j0, j1, _ = _axis(_get(J, p), nj, lj)
        i0, i1, _ = _axis(_get(I, p), ni, li)
        for k in (k0 // cz, k1 // cz):
            for j in (j0 // cy, j1 // cy):
                for i in (i0 // cx, i1 // cx):
                    M[k, j, i] = 1


# --------------------------------------------------------------------------
# HDF5 shuffle filter inverse. The source holds byte plane b of all n
# elements at src[b*n : (b+1)*n], little-endian byte order.
# --------------------------------------------------------------------------

@numba.njit(nogil=True, cache=True)
def unshuffle2(src, dst):
    n = dst.size
    for e in range(n):
        dst[e] = np.uint16(src[e]) | (np.uint16(src[n + e]) << np.uint16(8))


@numba.njit(nogil=True, cache=True)
def unshuffle4(src, dst):
    n = dst.size
    for e in range(n):
        dst[e] = np.uint32(src[e])
    for b in range(1, 4):
        s = np.uint32(8 * b)
        for e in range(n):
            dst[e] |= np.uint32(src[b * n + e]) << s


@numba.njit(nogil=True, cache=True)
def unshuffle8(src, dst):
    n = dst.size
    for e in range(n):
        dst[e] = np.uint64(src[e])
    for b in range(1, 8):
        s = np.uint64(8 * b)
        for e in range(n):
            dst[e] |= np.uint64(src[b * n + e]) << s


UNSHUFFLE = {2: (unshuffle2, np.uint16), 4: (unshuffle4, np.uint32),
             8: (unshuffle8, np.uint64)}
