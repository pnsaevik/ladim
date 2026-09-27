"""
Dense-array (numpy) sampling functions of the original ROMS module.

The forcing module itself uses the chunk loader and the kernels in
:mod:`.coords`. These functions operate on full 3-D arrays (e.g.
``grid.z_r`` and dense forcing fields) and are kept for plugins that
build on them.
"""

import numpy as np


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

    return F[K, J, I]


def sample3DUV(U, V, X, Y, K, A, method="bilinear"):
    return (
        sample3D(U, X + 0.5, Y, K, A, method=method),
        sample3D(V, X, Y + 0.5, K, A, method=method),
    )
