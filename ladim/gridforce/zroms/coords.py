"""
Vertical coordinate of z-level (depth interpolated) ROMS output.
"""

import numpy as np

from ..roms.coords import nearest_cell


class ZCoordinate:
    """Vertical coordinate given by fixed depths

    Has the same interface as :class:`ladim.gridforce.roms.coords.SCoordinate`,
    but the levels do not depend on the horizontal position.

    :param z_levels: Depths of the levels [m, positive downwards], strictly
        increasing. Level number 0 is the shallowest level.
    :param H: Bottom depth, 2-D subgrid array (only used for its shape)
    :param i0, j0: Global indices of subgrid point [0, 0]
    """

    def __init__(self, z_levels, H, i0=0, j0=0):
        z = np.ascontiguousarray(z_levels, dtype=np.float64).reshape(-1)
        if z.size == 0 or np.any(np.diff(z) <= 0):
            raise ValueError("z_levels must be strictly increasing")
        self.z_levels = z
        self.shape = np.shape(H)
        self.i0 = int(i0)
        self.j0 = int(j0)
        self._index = np.arange(z.size, dtype=np.float64)

    @property
    def N(self):
        return self.z_levels.size

    def level(self, X, Y, Z):
        """Fractional level index of depth Z (0 = shallowest level)

        Linear interpolation between levels floor(k) and floor(k)+1 at this
        index interpolates linearly in depth. Constant extension above the
        first and below the last level.
        """
        Z = np.asarray(Z, dtype=np.float64)
        Z = np.broadcast_to(Z, np.broadcast(X, Y, Z).shape)
        return np.interp(Z, self.z_levels, self._index)

    def cell(self, X, Y, Z):
        """Grid cell of the particles

        :return: (K, J, I), integer arrays. K is the level just above the
            particle (the shallowest level if the particle is above it, the
            deepest if it is below). J and I are the global indices of the
            nearest rho point within the subgrid.
        """
        k = self.level(X, Y, Z)
        K = np.floor(k).astype(np.int64)
        J, I = nearest_cell(X, Y, self.i0, self.j0, self.shape)
        return K, J, I
