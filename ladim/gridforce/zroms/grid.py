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
    levels are taken from the gridforce setting ``z_levels`` if given, and
    otherwise from the coordinate variable of the vertical dimension of ``u``
    in the first forcing file (see :func:`read_z_levels`). If the forcing
    file is not available (the grid file alone is used), ``z_levels`` is None.
    """

    def _read_angle(self, ncid):
        if "angle" in ncid.variables:
            return ncid.variables["angle"][self.J, self.I]
        return None

    def _read_vertical(self, ncid, gconf):
        if gconf.get("z_levels") is not None:
            z_levels = gconf["z_levels"]
            logger.info("Using the z levels given in the config")
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
    """Depths of the levels in the first forcing file, or None if no u in it"""
    if "input_file" not in gconf:
        return None
    files = sorted(glob.glob(gconf["input_file"]))
    if not files:
        raise FileNotFoundError("No input file: {}".format(gconf["input_file"]))
    with Dataset(files[0]) as nc:
        if "u" not in nc.variables:
            return None  # The input file is the grid file
        return read_z_levels(nc, files[0])


_METER_UNITS = {"m", "meter", "meters", "metre", "metres"}


def read_z_levels(nc, name="file"):
    """Depths of the z levels of an open netCDF dataset

    The levels are the coordinate variable of the vertical dimension of the
    velocity ``u`` (dimensions time, level, y, x). The variable must have
    units in meters and ``positive = "down"``, so that the values are
    depths below the surface. Otherwise, specify the depths with the gridforce
    setting ``z_levels``.
    """
    u = nc.variables["u"]
    if len(u.dimensions) != 4:
        raise ValueError(f"Expected u to have dimensions (time, z, y, x) in {name}, "
                         f"got {u.dimensions}")
    dim = u.dimensions[1]
    hint = " Specify the depths with gridforce.z_levels."
    if dim not in nc.variables:
        raise ValueError(f"The vertical dimension '{dim}' of u in {name} has no "
                         f"coordinate variable." + hint)
    var = nc.variables[dim]
    if var.dimensions != (dim,):
        raise ValueError(f"The variable '{dim}' in {name} is not a coordinate "
                         f"variable of the dimension '{dim}'." + hint)
    units = str(getattr(var, "units", "")).strip().lower()
    if units not in _METER_UNITS:
        raise ValueError(f"The units of '{dim}' in {name} must be meters, got "
                         f"{getattr(var, 'units', None)!r}." + hint)
    positive = str(getattr(var, "positive", "")).strip().lower()
    if positive != "down":
        raise ValueError(f"The attribute positive of '{dim}' in {name} must be "
                         f"'down', got {getattr(var, 'positive', None)!r}." + hint)
    logger.info(f"Reading z levels from variable '{dim}' in {name}")
    return np.array(var[:], dtype=np.float64)
