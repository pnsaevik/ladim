"""
zROMS grid: static grid fields and coordinate conversions.
"""

import glob
import logging

import numpy as np
from netCDF4 import Dataset

from ..roms.grid import Grid as RomsGrid
from .coords import ZCoordinate

logger = logging.getLogger(__name__)


class Grid(RomsGrid):
    """
    ROMS grid object for forcing files interpolated to z levels.

    The horizontal grid is as for the ROMS grid (see
    :class:`ladim.gridforce.roms.grid.Grid`). The depths of the vertical
    levels are taken from, in order of precedence, the gridforce setting
    ``z_levels``, the variable ``zlevels`` of the grid file, and the variable
    ``zlevels`` of the first forcing file. If none of these are given,
    ``z_levels`` is None, which is an error in :class:`.Forcing`.
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
            z_levels = _forcing_z_levels(gconf)
        if z_levels is None:
            # Not needed by the horizontal grid. The forcing requires them.
            self.z_levels = None
            self.N = None
            return
        self.z_levels = np.array(z_levels, dtype=np.float64)
        self.N = len(self.z_levels)
        logger.info(f"Vertical levels = {self.z_levels.tolist()}")

    def _make_coordinates(self):
        self.scoord = None
        if self.z_levels is not None:
            self.scoord = ZCoordinate(self.z_levels, self.H, self.i0, self.j0)
        self.scoord_w = None

    # Depths of the levels, shape (N, jmax, imax). Same as ``z_r`` of the
    # ROMS grid: negative below the surface.
    @property
    def z_r(self):
        if self.z_levels is None:
            raise ValueError("The depths of the z levels are unknown")
        return np.broadcast_to(-self.z_levels[:, None, None], (self.N,) + self.H.shape)

    @property
    def z_w(self):
        raise NotImplementedError("zROMS grids have no w levels")


def _forcing_z_levels(gconf):
    """The variable zlevels of the first forcing file, or None"""
    if "input_file" not in gconf:
        return None
    files = sorted(glob.glob(gconf["input_file"]))
    if not files:
        raise FileNotFoundError("No input file: {}".format(gconf["input_file"]))
    with Dataset(files[0]) as nc:
        if "zlevels" in nc.variables:
            return nc.variables["zlevels"][:]
    return None
