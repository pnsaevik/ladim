import numpy as np
from typing import Sequence
from scipy.ndimage import map_coordinates


class Grid:
    """
    The grid class represents the coordinate system used for particle tracking.
    It contains methods for converting between global coordinates (latitude,
    longitude, depth and posix time) and internal coordinates.
    """

    @staticmethod
    def from_roms(**conf):
        return RomsGrid(**conf)

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

        from .utilities import load_class
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
            mask: np.ndarray | tuple[tuple] = ((), ),
    ):
        """
        Define an array grid

        The number of lattice points in the T (time), Z (depth), Y and X
        dimensions are NT, NZ, NY and NX, respectively. It is assumed that

        - The lat/lon coordinates are independent of the T and Z dimensions
        - The depth is independent of the T dimension
        - The time is independent of the X, Y and Z timensions
        - Time values must be increasing
        - Depth values must be decreasing with Z

        :param lat: Latitude coordinates [degrees, NY * NX array]
        :param lon: Longitude coordinates [degrees, NY * NX array]
        :param depth: Depth below surface [meters, positive downwards, NZ * NY * NX array]
        :param time: Time since 1970-01-01 [seconds, NT array]
        :param mask: Zero at land positions (default all ones) [NY * NX array]
        """
        self.lat = np.asarray(lat, dtype='f8')
        self.lon = np.asarray(lon, dtype='f8')
        self.depth = np.asarray(depth, dtype='f4')
        self.time = np.asarray(time, dtype='datetime64[s]').astype('int64')

        if (not mask) or np.size(mask) == 0:
            self.mask = np.ones(self.depth.shape[-2:], dtype='i2')
        else:
            self.mask = np.asarray(mask, dtype='i2')

        self._cache_dict = dict()

        if np.any(np.diff(self.time) <= 0):
            raise ValueError('Time values must be increasing')
        if np.any(self.depth[0] < self.depth[-1]):
            raise ValueError('Depth values must be decreasing with Z')

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

    def to_depth(
            self, x: Sequence, y: Sequence, s: Sequence
    ) -> np.ndarray:
        mask = map_coordinates(self.mask, (y, x), order=0, mode='nearest')
        depth = map_coordinates(self.depth, (s, y, x), order=1, mode='nearest')
        depth[mask == 0] = 0
        return depth

    def from_depth(
            self, x: Sequence, y: Sequence, z: Sequence
    ) -> np.ndarray:
        depths = bilinear_interp(self.depth, y, x)
        idx, frac = array_lookup(
            arr=-depths,
            values=-np.asarray(z),
            return_frac=True,
        )

        s = idx + frac
        mask = map_coordinates(self.mask, (y, x), order=0, mode='nearest')
        s[mask == 0] = 0
        return s

    def compute(self, key):
        """
        Cached computation of key variables

        :param key: The variable to compute
        :return: The computed variables
        """
        if key in self._cache_dict:
            return self._cache_dict[key]

        if key in ['dx', 'dy']:
            dx, dy = compute_dx_dy(lat=self.lat, lon=self.lon)
            self._cache_dict['dx'] = dx
            self._cache_dict['dy'] = dy

        elif key in ['latdiff_x', 'latdiff_y', 'londiff_x', 'londiff_y']:
            lax = (self.lat[:, 1:] - self.lat[:, :-1]) * (np.pi / 180)
            lox = (self.lon[:, 1:] - self.lon[:, :-1]) * (np.pi / 180)
            lay = (self.lat[1:, :] - self.lat[:-1, :]) * (np.pi / 180)
            loy = (self.lon[1:, :] - self.lon[:-1, :]) * (np.pi / 180)
            self._cache_dict['latdiff_x'] = lax
            self._cache_dict['londiff_x'] = lox
            self._cache_dict['latdiff_y'] = lay
            self._cache_dict['londiff_y'] = loy

        return self._cache_dict[key]

    def dx(self, x: Sequence, y: Sequence) -> np.ndarray:
        x = np.asarray(x)
        y = np.asarray(y)
        dx = self.compute('dx')
        coords = (y, x - 0.5)  # Convert from 'rho' to 'u' coordinates
        return map_coordinates(dx, coords, order=1, mode='nearest')

    def dy(self, x: Sequence, y: Sequence) -> np.ndarray:
        x = np.asarray(x)
        y = np.asarray(y)
        dy = self.compute('dy')
        coords = (y - 0.5, x)  # Convert from 'rho' to 'v' coordinates
        return map_coordinates(dy, coords, order=1, mode='nearest')

    def _latlondiff(self, x: Sequence, y: Sequence):
        """
        Compute latitude and longitude unit difference at selected points

        Returns a tuple latdiff_x, latdiff_y, londiff_x, londiff_y. Together,
        these variables tell how much the latitude and longitude increases when
        moving by one grid cell in either the X or Y direction.

        :param x: X coordinates of starting points
        :param y: Y coordinates of starting points
        :return: A tuple latdiff_x, latdiff_y, londiff_x, londiff_y
        """
        x = np.asarray(x)
        y = np.asarray(y)

        latdiff_xdir_grid = self.compute('latdiff_x')
        londiff_xdir_grid = self.compute('londiff_x')
        latdiff_ydir_grid = self.compute('latdiff_y')
        londiff_ydir_grid = self.compute('londiff_y')

        crd_x = (y, x - 0.5)  # Convert from 'rho' to 'u' coordinates
        crd_y = (y - 0.5, x)  # Convert from 'rho' to 'v' coordinates
        latdiff_xdir = map_coordinates(latdiff_xdir_grid, crd_x, order=1, mode='nearest')
        londiff_xdir = map_coordinates(londiff_xdir_grid, crd_x, order=1, mode='nearest')
        latdiff_ydir = map_coordinates(latdiff_ydir_grid, crd_y, order=1, mode='nearest')
        londiff_ydir = map_coordinates(londiff_ydir_grid, crd_y, order=1, mode='nearest')

        return latdiff_xdir, latdiff_ydir, londiff_xdir, londiff_ydir

    def to_bearing(
            self, x: Sequence, y: Sequence, az: Sequence
    ) -> np.ndarray:
        # Compute unit lat/lon difference in the x and y directions
        latdiff_x, latdiff_y, londiff_x, londiff_y = self._latlondiff(x, y)

        # Define directional vector 'p' which is defined on the x/y grid
        az_radians = np.asarray(az) * (np.pi / 180)
        p_x = np.cos(az_radians)
        p_y = np.sin(az_radians)

        # Define new vector 'q' which is defined on the lon/lat grid
        # and has the same direction as 'p'
        q_lat = p_x * latdiff_x + p_y * latdiff_y
        q_lon = p_x * londiff_x + p_y * londiff_y

        # Compute bearing
        bearing_radians = np.atan2(q_lon, q_lat)
        bearing = (bearing_radians * (180 / np.pi)) % 360
        return bearing

    def from_bearing(
            self, x: Sequence, y: Sequence, b: Sequence
    ) -> np.ndarray:
        # Compute unit lat/lon difference in the x and y directions
        latdiff_x, latdiff_y, londiff_x, londiff_y = self._latlondiff(x, y)

        # Define directional vector 'q' which is defined on the lat/lon grid
        bearing_radians = np.asarray(b) * (np.pi / 180)
        q_lat = np.cos(bearing_radians)
        q_lon = np.sin(bearing_radians)

        # Define new vector 'p' which is defined on the x/y grid
        # and has the same direction as 'q'
        p_x = q_lat * londiff_y - q_lon * latdiff_y
        p_y = -q_lat * londiff_x + q_lon * latdiff_x

        # Compute azimuth
        az_radians = np.atan2(p_y, p_x)
        az = (az_radians * (180 / np.pi)) % 360
        return az


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


def array_lookup(arr, values, return_frac=False):
    """
    Find indices of a set of values

    The lookup table "arr" has dimensions N * M, and should be sorted
    along the M axis (i.e., arr[i, j] <= arr[i, j + 1] for all i, j)

    The search value array "values" should have dimensions N. The values
    are clipped by the minimum and maximum values given by "arr".

    The function returns and index array "idx" such that arr[i, idx[i]] <=
    values[i] < arr[i, idx[i] + 1] for all i. All values in "idx" are
    values between 0 and M - 2.

    If the parameter "return_frac" is set to True, the function returns an
    additional array "frac" with values in the range [0, 1] such that
    values == arr[i, idx[i]] * (1 - frac[i]) + arr[i, idx[i + 1]] * frac[i]
    for all i.

    :param arr: Lookup table, shape N * M
    :param values: Values to search for, shape N
    :param return_frac: True if interpolation index "frac" should be returned
    :return: A tuple ("idx", "frac"), or just "idx" if return_frac is set to False
    """

    arr = np.asarray(arr)
    values = np.asarray(values)
    n, m = arr.shape

    assert (n, ) == values.shape
    assert np.all(arr[:, 0] <= arr[:, -1])

    idx_raw = np.sum(arr.T <= values, axis=0) - 1
    idx = np.maximum(0, np.minimum(idx_raw, m - 2))

    if not return_frac:
        return idx

    i = np.arange(n)
    values_0 = arr[i, idx]
    values_1 = arr[i, idx + 1]
    frac_raw = (values - values_0) / (values_1 - values_0)
    frac = np.maximum(0, np.minimum(frac_raw, 1))
    return idx, frac


def bilinear_interp(arr: np.ndarray, y: Sequence, x: Sequence):
    """
    Bilinear interpolation of a multi-dimensional array

    The function interpolates the input array in the second last and last
    dimensions, but leaves the first dimensions unchanged.

    :param arr: Input array
    :param y: Fractional coordinates of the second last dimension
    :param x: Fractional coordinates of the last dimension
    :return:
    """
    nx = arr.shape[-1]
    ny = arr.shape[-2]

    x = np.minimum(nx - 1, np.maximum(0, x))
    y = np.minimum(ny - 1, np.maximum(0, y))

    x0 = np.minimum(nx - 2, np.int32(x))
    y0 = np.minimum(ny - 2, np.int32(y))

    xf = x - x0
    yf = y - y0

    z00 = arr[..., y0, x0]
    z01 = arr[..., y0, x0 + 1]
    z10 = arr[..., y0 + 1, x0]
    z11 = arr[..., y0 + 1, x0 + 1]

    z = (
            z00 * (1 - xf) * (1 - yf)
            + z01 * xf * (1 - yf)
            + z10 * (1 - xf) * yf
            + z11 * xf * yf
    )
    return z.T


def compute_dx_dy(lat, lon):
    """
    Compute scale factors and bearings from grid

    The grid is assumed to be a structured grid with lat/lon coordinates
    at every grid point. The function computes two variables:

    dx:  The distance (in meters) when moving one grid cell along the X axis
    dy:  The distance (in meters) when moving one grid cell along the Y axis

    The shape of the returned arrays will be one less than the input arrays
    in the dimension where the differential is computed.

    :param lat: Latitude (in degrees) of grid points
    :param lon: Longitude (in degrees) of grid points
    :return: A tuple (dx, dy)
    """
    import pyproj

    geod = pyproj.Geod(ellps='WGS84')

    _, _, dist_x = geod.inv(
        lons1=lon[:, :-1], lats1=lat[:, :-1],
        lons2=lon[:, 1:], lats2=lat[:, 1:],
    )

    _, _, dist_y = geod.inv(
        lons1=lon[:-1, :], lats1=lat[:-1, :],
        lons2=lon[1:, :], lats2=lat[1:, :],
    )

    return dist_x, dist_y
