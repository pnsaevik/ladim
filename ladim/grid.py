from .model import Module
import numpy as np
from typing import Sequence
from scipy.ndimage import map_coordinates


class Grid(Module):
    """
    The grid class represents the coordinate system used for particle tracking.
    It contains methods for converting between global coordinates (latitude,
    longitude, depth and posix time) and internal coordinates.
    """

    def ingrid(self, X, Y):
        raise NotImplementedError

    def sample_metric(self, X, Y):
        raise NotImplementedError

    def atsea(self, X, Y):
        raise NotImplementedError

    def ll2xy(self, lon, lat):
        raise NotImplementedError

    def xy2ll(self, x, y):
        raise NotImplementedError

    # --- MODERN METHODS ---

    def dx(self, x: Sequence, y: Sequence) -> np.ndarray:
        """
        Metric scale factor in the X direction

        The metric scale factor is defined such that if one moves
        a small increment delta along the axis, then the distance
        (in meters) equals scale_factor * delta.

        :param x: X positions
        :param y: Y positions
        :return: Metric scale factor [in meters per grid unit]
        """
        raise NotImplementedError

    def dy(self, x: Sequence, y: Sequence) -> np.ndarray:
        """
        Metric scale factor in the Y direction

        The metric scale factor is defined such that if one moves
        a small increment delta along the axis, then the distance
        (in meters) equals scale_factor * delta.

        :param x: X positions
        :param y: Y positions
        :return: Metric scale factor [in meters per grid unit]
        """
        raise NotImplementedError

    def from_bearing(
            self, x: Sequence, y: Sequence, b: Sequence
    ) -> np.ndarray:
        """
        Azimutal angles from compass bearings.

        A compass bearing (in degrees) is defined such that 0 is north, 90
        is east, 180 is south and 270 is west. An azimutal vector angle is
        defined such that 0 is pointing along the X axis, 90 is pointing
        along the Y axis, 180 is pointing opposite the X axis and 270 is
        pointing opposite the Y axis.

        This function computes a set of azimutal vector angles from a set
        of compass bearings and horizontal positions.

        :param x: X positions
        :param y: Y positions
        :param b: Compass bearings [degrees]
        :return: Azimutal vector angles [degrees]
        """
        raise NotImplementedError

    def to_bearing(
            self, x: Sequence, y: Sequence, az: Sequence
    ) -> np.ndarray:
        """
        Azimutal angles from compass bearings.

        A compass bearing (in degrees) is defined such that 0 is north, 90
        is east, 180 is south and 270 is west. An azimutal vector angle is
        defined such that 0 is pointing along the X axis, 90 is pointing
        along the Y axis, 180 is pointing opposite the X axis and 270 is
        pointing opposite the Y axis.

        This function computes a set of compass bearings from a set
        of azimutal vector angles and horizontal positions.

        :param x: X positions
        :param y: Y positions
        :param az: Azimutal vector angles [degrees]
        :return: Compass bearings [degrees]
        """
        raise NotImplementedError

    def from_depth(
            self, x: Sequence, y: Sequence, z: Sequence
    ) -> np.ndarray:
        """
        Vertical coordinates from depth and horizontal coordinates.

        :param x: X positions
        :param y: Y positions
        :param z: Depth below surface [m, positive downwards]
        :return: Vertical coordinates
        """
        raise NotImplementedError

    def to_latlon(
            self, x: Sequence, y: Sequence,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Latitude and longitude from horizontal coordinates

        :param x: X positions
        :param y: Y positions
        :return: A tuple (lat, lon) of latitude [degrees north] and longitude [degrees east]
        """
        raise NotImplementedError

    def from_latlon(
            self, lat: Sequence, lon: Sequence,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Horizontal coordinates from latitude and longitude

        :param lat: Latitude [degrees north]
        :param lon: Longitude [degrees east]
        :return: A tuple (x, y) of horizontal coordinates
        """
        raise NotImplementedError

    def to_depth(
            self, x: Sequence, y: Sequence, s: Sequence
    ) -> np.ndarray:
        """
        Depth from horizontal and vertical coordinates.

        :param x: X positions
        :param y: Y positions
        :param s: Vertical coordinates
        :return: Depth below surface [m, positive downwards]
        """
        raise NotImplementedError

    def from_epoch(self, p: Sequence) -> np.ndarray:
        """
        Time coordinates from posix time

        :param p: Posix time [seconds since 1970-01-01]
        :return: Time coordinates
        """
        raise NotImplementedError

    def to_epoch(self, t: Sequence) -> np.ndarray:
        """
        Posix time from time coordinates

        :param t: Time coordinates
        :return: Posix time [seconds since 1970-01-01]
        """
        raise NotImplementedError


class RomsGrid(Grid):
    def __init__(
            self,
            file: str,
            start_time=None,
            subgrid=None,
            legacy_module='ladim.gridforce.ROMS.Grid',
            **_,
    ):

        legacy_conf = dict(
            gridforce=dict(
                input_file=file,
            ),
            start_time=start_time,
        )
        if subgrid is not None:
            legacy_conf['gridforce']['subgrid'] = subgrid

        from .model import load_class
        LegacyGrid = load_class(legacy_module)

        # Allow gridforce module in current directory
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        # Import correct gridforce_module
        # gridforce_module = importlib.import_module(config["gridforce_module"])
        self.grid = LegacyGrid(legacy_conf)
        self.xmin = self.grid.xmin
        self.xmax = self.grid.xmax
        self.ymin = self.grid.ymin
        self.ymax = self.grid.ymax

    def sample_metric(self, X, Y):
        """Sample the metric coefficients"""
        return self.grid.sample_metric(X, Y)

    def sample_depth(self, X, Y):
        """Return the depth of grid cells"""
        return self.grid.sample_depth(X, Y)

    def lonlat(self, X, Y, method=None):
        """Return the longitude and latitude from grid coordinates"""
        return self.grid.lonlat(X, Y, method=method)

    def ingrid(self, X, Y):
        """Returns True for points inside the subgrid"""
        return self.grid.ingrid(X, Y)

    def onland(self, X, Y):
        """Returns True for points on land"""
        return self.grid.onland(X, Y)

    # Error if point outside
    def atsea(self, X, Y):
        """Returns True for points at sea"""
        return self.grid.atsea(X, Y)

    def ll2xy(self, lon, lat):
        return self.grid.ll2xy(lon, lat)

    def xy2ll(self, X, Y):
        return self.grid.xy2ll(X, Y)


class ArrayGrid(Grid):
    def __init__(
            self,
            lat: np.ndarray | tuple[tuple] = ((), ),
            lon: np.ndarray | tuple[tuple] = ((), ),
            depth: np.ndarray | tuple[tuple[tuple]] = (((), ), ),
            time: np.ndarray | tuple = (),
    ):
        """
        Define an array grid

        The number of lattice points in the T (time), Z (depth), Y and X
        dimensions are NT, NZ, NY and NX, respectively. It is assumed that

        - The lat/lon coordinates are independent of the T and Z dimensions
        - The depth is independent of the T dimension
        - The time is independent of the X, Y and Z timensions

        :param lat: Latitude coordinates [degrees, NY * NX array]
        :param lon: Longitude coordinates [degrees, NY * NX array]
        :param depth: Depth below surface [meters, positive downwards, NZ * NY * NX array]
        :param time: Time since 1970-01-01 [seconds, NT array]
        """
        self.lat = np.asarray(lat).astype('f8')
        self.lon = np.asarray(lon).astype('f8')
        self.depth = np.asarray(depth).astype('f4')
        self.time = np.asarray(time).astype('datetime64[s]').astype('int64')

    def from_epoch(self, p: Sequence) -> np.ndarray:
        return np.interp(x=p, xp=self.time, fp=np.arange(len(self.time)))

    def to_epoch(self, t: Sequence) -> np.ndarray:
        return map_coordinates(self.time, (t, ), order=1, mode='nearest')

    def to_latlon(
            self, x: Sequence, y: Sequence,
    ) -> tuple[np.ndarray, np.ndarray]:
        lat = map_coordinates(self.lat, (y, x), order=1, mode='nearest')
        lon = map_coordinates(self.lon, (y, x), order=1, mode='nearest')
        return lat, lon

    def from_latlon(
            self, lat: Sequence, lon: Sequence,
    ) -> tuple[np.ndarray, np.ndarray]:
        y, x = bilin_inv(f=lat, g=lon, F=self.lat, G=self.lon)
        return x, y


def bilin_inv(f, g, F, G, maxiter=7, tol=1.0e-7) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse bilinear interpolation

    ``f, g`` should be scalars or arrays of same shape

    ``F, G`` should be 2D arrays of the same shape

    :param f: Desired f value
    :param g: Desired g value
    :param F: Tabulated f values
    :param G: Tabulated g values
    :param maxiter: Maximum number of Newton iterations
    :param tol: Maximum residual value
    :return: A tuple ``(x, y)`` such that ``F[x, y] = f`` and ``G[x, y] = g``, when
        linearly interpolated
    """

    imax, jmax = np.array(F.shape) - 1

    f = np.asarray(f)
    g = np.asarray(g)

    # initial guess
    x = np.zeros_like(f) + 0.5 * imax
    y = np.zeros_like(f) + 0.5 * jmax

    for t in range(maxiter):
        i = np.minimum(imax - 1, x.astype("i4"))
        j = np.minimum(jmax - 1, y.astype("i4"))

        p, q = x - i, y - j

        # Shorthands
        F00 = F[i, j]
        F01 = F[i, j+1]
        F10 = F[i+1, j]
        F11 = F[i+1, j+1]
        G00 = G[i, j]
        G01 = G[i, j+1]
        G10 = G[i+1, j]
        G11 = G[i+1, j+1]

        # Bilinear estimate of F[x,y] and G[x,y]
        Fs = (
            (1 - p) * (1 - q) * F00
            + p * (1 - q) * F10
            + (1 - p) * q * F01
            + p * q * F11
        )
        Gs = (
            (1 - p) * (1 - q) * G00
            + p * (1 - q) * G10
            + (1 - p) * q * G01
            + p * q * G11
        )

        H = (Fs - f) ** 2 + (Gs - g) ** 2

        if np.all(H < tol**2):
            break

        # Estimate Jacobi matrix
        Fx = (1 - q) * (F10 - F00) + q * (F11 - F01)
        Fy = (1 - p) * (F01 - F00) + p * (F11 - F10)
        Gx = (1 - q) * (G10 - G00) + q * (G11 - G01)
        Gy = (1 - p) * (G01 - G00) + p * (G11 - G10)

        # Newton-Raphson step
        # Jinv = np.linalg.inv([[Fx, Fy], [Gx, Gy]])
        # incr = - np.dot(Jinv, [Fs-f, Gs-g])
        # x = x + incr[0], y = y + incr[1]
        det = Fx * Gy - Fy * Gx
        x -= (Gy * (Fs - f) - Fy * (Gs - g)) / det
        y -= (-Gx * (Fs - f) + Fx * (Gs - g)) / det

        x = np.maximum(0, np.minimum(imax, x))
        y = np.maximum(0, np.minimum(jmax, y))

    return x, y
