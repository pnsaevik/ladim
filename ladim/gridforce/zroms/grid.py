"""
zROMS grid: static grid fields and coordinate conversions.
"""

import logging

import numpy as np

from ..roms.grid import Grid as RomsGrid
from .coords import ZCoordinate

logger = logging.getLogger(__name__)

# Depths of the levels in the forcing files, unless specified otherwise
DEFAULT_Z_LEVELS = (0, 1, 2, 3, 5, 10, 15, 20)


class Grid(RomsGrid):
    """
    ROMS grid object for forcing files interpolated to z levels.

    The horizontal grid is as for the ROMS grid (see
    :class:`ladim.gridforce.roms.grid.Grid`). The depths of the vertical
    levels are taken from, in order of precedence, the gridforce setting
    ``z_levels``, the variable ``zlevels`` of the grid file, and the default
    ``[0, 1, 2, 3, 5, 10, 15, 20]``.
    """

    def _read_angle(self, ncid):
        if "angle" in ncid.variables:
            return ncid.variables["angle"][self.J, self.I]
        return None

    def _read_vertical(self, ncid, gconf):
        if gconf.get("z_levels") is not None:
            z_levels = gconf["z_levels"]
        elif "zlevels" in ncid.variables:
            z_levels = ncid.variables["zlevels"][:]
        else:
            z_levels = DEFAULT_Z_LEVELS
        self.z_levels = np.array(z_levels, dtype=np.float64)
        self.N = len(self.z_levels)
        logger.info(f"Vertical levels = {self.z_levels.tolist()}")

    def _make_coordinates(self):
        self.scoord = ZCoordinate(self.z_levels, self.H, self.i0, self.j0)
        self.scoord_w = None

    # Depths of the levels, shape (N, jmax, imax). Same as ``z_r`` of the
    # ROMS grid: negative below the surface.
    @property
    def z_r(self):
        return np.broadcast_to(-self.z_levels[:, None, None], (self.N,) + self.H.shape)

    z_w = z_r
