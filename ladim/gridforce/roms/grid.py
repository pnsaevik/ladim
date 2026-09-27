"""
ROMS grid: static grid fields and coordinate conversions.
"""

import glob
import logging

import numpy as np
from netCDF4 import Dataset

from ladim.sample import bilin_inv

from .. import parallel
from . import coords

logger = logging.getLogger(__name__)


class Grid:
    """
    ROMS grid object.

    Possible grid arguments are subgrid and Vinfo. subgrid = [i0, i1, j0, j1].
    Ordinary python style, start points included, not end points. Each of the
    elements can be replaced with None, for no limitation. Vinfo: dictionary
    with N, hc, theta_s and theta_b (and optionally Vstretching, Vtransform).

    Horizontal positions X, Y are global grid coordinates, while the 2-D
    arrays (H, M, dx, ...) cover the subgrid only.
    """

    def __init__(self, config):
        logger.info("Initializing ROMS-type grid object")
        gconf = config["gridforce"]
        parallel.configure(gconf.get("num_threads"))

        if "grid_file" in gconf:
            grid_file = gconf["grid_file"]
        elif "input_file" in gconf:
            grid_file = sorted(glob.glob(gconf["input_file"]))[0]
        else:
            logger.error("No grid file specified")
            raise SystemExit(1)

        try:
            ncid = Dataset(grid_file)
        except OSError:
            logger.error("Could not open grid file " + grid_file)
            raise SystemExit(1)

        with ncid:
            ncid.set_auto_mask(False)
            self._read_grid(ncid, gconf)

        self.scoord = coords.SCoordinate(
            self.H, self.Cs_r, self.hc, self.Vtransform, self.i0, self.j0
        )
        self._M8 = np.ascontiguousarray(self.M, dtype=np.int8)
        self._z_r = None
        self._z_w = None

    def _read_grid(self, ncid, gconf):
        # Subgrid, only considers internal grid cells
        # 1 <= i0 < i1 <= imax-1, default=end points
        # 1 <= j0 < j1 <= jmax-1, default=end points
        jmax, imax = ncid.variables["h"].shape
        whole_grid = [1, imax - 1, 1, jmax - 1]
        limits = list(gconf.get("subgrid") or whole_grid)
        limits = [w if v is None else v for v, w in zip(limits, whole_grid)]

        self.i0, self.i1, self.j0, self.j1 = limits
        self.imax = self.i1 - self.i0
        self.jmax = self.j1 - self.j0

        # Limits for where velocities are defined
        self.xmin = float(self.i0)
        self.xmax = float(self.i1 - 1)
        self.ymin = float(self.j0)
        self.ymax = float(self.j1 - 1)

        # Slices of rho-, u- and v-points
        self.I = slice(self.i0, self.i1)
        self.J = slice(self.j0, self.j1)
        self.Iu = slice(self.i0 - 1, self.i1)
        self.Ju = self.J
        self.Iv = self.I
        self.Jv = slice(self.j0 - 1, self.j1)

        # Vertical grid
        if "Vinfo" in gconf:
            Vinfo = gconf["Vinfo"]
            self.N = Vinfo["N"]
            self.hc = Vinfo["hc"]
            self.Vstretching = Vinfo.get("Vstretching", 1)
            self.Vtransform = Vinfo.get("Vtransform", 1)
            args = (self.N, Vinfo["theta_s"], Vinfo["theta_b"])
            self.Cs_r = coords.s_stretch(*args, stagger="rho", Vstretching=self.Vstretching)
            self.Cs_w = coords.s_stretch(*args, stagger="w", Vstretching=self.Vstretching)
        else:
            self.hc = ncid.variables["hc"].getValue()
            self.Cs_r = ncid.variables["Cs_r"][:]
            self.Cs_w = ncid.variables["Cs_w"][:]
            self.N = len(self.Cs_r)
            if "Vtransform" in ncid.variables:
                self.Vtransform = ncid.variables["Vtransform"].getValue()
            else:
                self.Vtransform = 1  # Default = old way

        J, I = self.J, self.I
        self.H = ncid.variables["h"][J, I]
        self.M = ncid.variables["mask_rho"][J, I].astype(int)
        self.dx = 1.0 / ncid.variables["pm"][J, I]
        self.dy = 1.0 / ncid.variables["pn"][J, I]
        self.lon = ncid.variables["lon_rho"][J, I]
        self.lat = ncid.variables["lat_rho"][J, I]
        self.angle = ncid.variables["angle"][J, I]

        # Land masks at u- and v-points, boundary points copied from rho
        M = self.M
        self.Mu = np.zeros((self.jmax, self.imax + 1), dtype=int)
        self.Mu[:, 1:-1] = M[:, :-1] * M[:, 1:]
        self.Mu[:, 0] = M[:, 0]
        self.Mu[:, -1] = M[:, -1]
        self.Mv = np.zeros((self.jmax + 1, self.imax), dtype=int)
        self.Mv[1:-1, :] = M[:-1, :] * M[1:, :]
        self.Mv[0, :] = M[0, :]
        self.Mv[-1, :] = M[-1, :]

    # Full 3-D depth arrays (1 GB each for large grids), computed on access
    @property
    def z_r(self):
        if self._z_r is None:
            self._z_r = coords.sdepth(self.H, self.hc, self.Cs_r, stagger="rho",
                                      Vtransform=self.Vtransform)
        return self._z_r

    @z_r.setter
    def z_r(self, value):
        self._z_r = value

    @property
    def z_w(self):
        if self._z_w is None:
            self._z_w = coords.sdepth(self.H, self.hc, self.Cs_w, stagger="w",
                                      Vtransform=self.Vtransform)
        return self._z_w

    @z_w.setter
    def z_w(self, value):
        self._z_w = value

    # ---------- Cell lookups ----------

    def _lookup(self, F, X, Y, clip=None):
        return coords.lookup(F, X, Y, self.i0, self.j0, clip)

    def sample_metric(self, X, Y):
        """Grid spacing (dx, dy) in the nearest cell.

        The grid is assumed conformal (dx == dy), as for polar stereographic
        grids, so dx is returned for both.
        """
        jmax, imax = self.dx.shape
        A = self._lookup(self.dx, X, Y, clip=(jmax - 1, imax - 1))
        return A, A

    def sample_depth(self, X, Y):
        """Bottom depth of the nearest grid cell"""
        return self._lookup(self.H, X, Y)

    def onland(self, X, Y):
        """Returns True for points on land"""
        return self._lookup(self._M8, X, Y) < 1

    def atsea(self, X, Y):
        """Returns True for points at sea"""
        return self._lookup(self._M8, X, Y) > 0

    def ingrid(self, X, Y):
        """Returns True for points inside the subgrid"""
        return coords.inside(X, Y, self.xmin + 0.5, self.xmax - 0.5,
                             self.ymin + 0.5, self.ymax - 0.5)

    # ---------- Geographical coordinates ----------

    def lonlat(self, X, Y, method="bilinear"):
        """Return the longitude and latitude from grid coordinates"""
        if method == "bilinear":  # More accurate
            return self.xy2ll(X, Y)
        # else: containing grid cell, less accurate
        return self._lookup(self.lon, X, Y), self._lookup(self.lat, X, Y)

    def xy2ll(self, X, Y):
        x = np.asarray(X) - self.i0
        y = np.asarray(Y) - self.j0
        return coords.bilinear(self.lon, x, y), coords.bilinear(self.lat, x, y)

    def ll2xy(self, lon, lat):
        Y, X = bilin_inv(lon, lat, self.lon, self.lat)
        return X + self.i0, Y + self.j0
