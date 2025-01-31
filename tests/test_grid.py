from ladim import grid
import numpy as np


class Test_ArrayGrid_from_epoch:
    def test_can_extract(self):
        g = grid.ArrayGrid(time=[0, 60, 120])
        assert g.from_epoch([60, 60, 120, 0]).tolist() == [1, 1, 2, 0]

    def test_can_interpolate(self):
        g = grid.ArrayGrid(time=[0, 60, 120])
        assert g.from_epoch([30, 90]).tolist() == [0.5, 1.5]

    def test_extrapolates_as_constant(self):
        g = grid.ArrayGrid(time=[0, 60, 120])
        assert g.from_epoch([-50, 200]).tolist() == [0, 2]


class Test_ArrayGrid_to_epoch:
    def test_can_extract(self):
        g = grid.ArrayGrid(time=[0, 60, 120])
        assert g.to_epoch([1, 1, 2, 0]).tolist() == [60, 60, 120, 0]

    def test_can_interpolate(self):
        g = grid.ArrayGrid(time=[0, 60, 120])
        assert g.to_epoch([0.5, 1.5]).tolist() == [30, 90]

    def test_extrapolates_as_constant(self):
        g = grid.ArrayGrid(time=[0, 60, 120])
        assert g.to_epoch([-1, 3]).tolist() == [0, 120]


class Test_ArrayGrid_to_latlon:
    def test_can_extract(self):
        g = grid.ArrayGrid(
            lat=[[60, 60, 60], [61, 61, 61]],
            lon=[[5, 6, 7], [5, 6, 7]],
        )
        lat, lon = g.to_latlon([0, 1, 1], [0, 0, 1])
        assert lat.tolist() == [60, 60, 61]
        assert lon.tolist() == [5, 6, 6]

    def test_can_interpolate(self):
        g = grid.ArrayGrid(
            lat=[[60, 60, 60], [61, 61, 61]],
            lon=[[5, 6, 7], [5, 6, 7]],
        )
        lat, lon = g.to_latlon([0.5, 1.5, 0], [0, 0, 0.5])
        assert lat.tolist() == [60, 60, 60.5]
        assert lon.tolist() == [5.5, 6.5, 5]

    def test_extrapolates_as_constant(self):
        g = grid.ArrayGrid(
            lat=[[60, 60, 60], [61, 61, 61]],
            lon=[[5, 6, 7], [5, 6, 7]],
        )
        lat, lon = g.to_latlon([-1, 5, 0], [0, 0, 5])
        assert lat.tolist() == [60, 60, 61]
        assert lon.tolist() == [5, 7, 5]


class Test_ArrayGrid_from_latlon:
    def test_can_extract(self):
        g = grid.ArrayGrid(
            lat=[[60, 60, 60], [61, 61, 61]],
            lon=[[5, 6, 7], [5, 6, 7]],
        )
        x, y = g.from_latlon([60, 60, 61, 61], [5, 6, 6, 7])
        assert x.tolist() == [0, 1, 1, 2]
        assert y.tolist() == [0, 0, 1, 1]

    def test_can_interpolate(self):
        g = grid.ArrayGrid(
            lat=[[60, 60, 60], [61, 61, 61]],
            lon=[[5, 6, 7], [5, 6, 7]],
        )
        x, y = g.from_latlon([60.5, 60.5, 61, 61], [5, 6, 5.5, 6.5])
        assert x.tolist() == [0, 1, 0.5, 1.5]
        assert y.tolist() == [0.5, 0.5, 1, 1]

    def test_extrapolates_as_constant(self):
        g = grid.ArrayGrid(
            lat=[[60, 60, 60], [61, 61, 61]],
            lon=[[5, 6, 7], [5, 6, 7]],
        )
        x, y = g.from_latlon([50, 70, 60, 60], [6, 6, 4, 9])
        assert x.tolist() == [1, 1, 0, 2]
        assert y.tolist() == [0, 1, 0, 0]


class Test_bilinear_interp:
    def test_can_extract(self):
        depth = np.flip(np.arange(24).reshape((2, 3, 4)), 0)
        x = [0, 1, 1]
        y = [0, 0, 1]
        z = grid.bilinear_interp(depth, y, x)
        assert z.tolist() == [
            [12, 0],
            [13, 1],
            [17, 5],
        ]

    def test_can_interpolate(self):
        depth = np.flip(np.arange(24).reshape((2, 3, 4)), 0)
        x = [0, 1, 0, 1, .5, 0, .5]
        y = [0, 0, 1, 1, 0, .5, .5]
        z = grid.bilinear_interp(depth, y, x)
        assert z.tolist() == [
            [12.0, 0.0],
            [13.0, 1.0],
            [16.0, 4.0],
            [17.0, 5.0],
            [12.5, 0.5],
            [14.0, 2.0],
            [14.5, 2.5],
        ]

    def test_extrapolates_as_constant(self):
        depth = np.flip(np.arange(24).reshape((2, 3, 4)), 0)
        x = [0, -1, 0, 3, 4, 3]
        y = [0, 0, -1, 2, 2, 3]
        z = grid.bilinear_interp(depth, y, x)
        assert z.tolist() == [
            [12, 0],
            [12, 0],
            [12, 0],
            [23, 11],
            [23, 11],
            [23, 11],
        ]


class Test_ArrayGrid_from_to_depth:
    def test_can_extract(self):
        g = grid.ArrayGrid(depth=np.flip(np.arange(24).reshape((2, 3, 4)), 0))
        x = [0, 1, 2, 3]
        y = [0, 1, 2, 1]
        s = [0, 1, 0, 1]
        z = [12, 5, 22, 7]
        assert g.to_depth(x, y, s).tolist() == z
        assert g.from_depth(x, y, z).tolist() == s

    def test_can_interpolate(self):
        g = grid.ArrayGrid(depth=np.flip(np.arange(24).reshape((2, 3, 4)), 0))
        x = [0, .5, 1, 1, 1]
        y = [0, 0, 0, .5, 1]
        s = [1, 1, 1, 1, 1]
        z = [0, 0.5, 1, 3, 5]
        assert g.to_depth(x, y, s).tolist() == z
        assert g.from_depth(x, y, z).tolist() == s

    def test_extrapolates_as_constant(self):
        g = grid.ArrayGrid(depth=np.flip(np.arange(24).reshape((2, 3, 4)), 0))
        x = [-1, 9, 1, 1]
        y = [1, 1, -1, 9]
        s = [1, 1, 1, 1]
        z = [4, 7, 1, 9]
        assert g.to_depth(x, y, s).tolist() == z
        assert g.from_depth(x, y, z).tolist() == s

    def test_depth_is_zero_on_land(self):
        g = grid.ArrayGrid(
            depth=np.flip(np.arange(24).reshape((2, 3, 4)), 0),
            mask=[[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]]
        )
        x = [0, .25, .75, 1, 1.25, 1.75, 2]
        y = [1] * len(x)
        s = [1] * len(x)
        z = [4, 4.25, 0, 0, 0, 5.75, 6]
        assert g.to_depth(x, y, s).tolist() == z


class Test_ArrayGrid_dx_dy:
    def test_can_interpolate(self):
        g = grid.ArrayGrid(
            lat=[[60, 60, 60, 60], [61, 61, 61, 61], [62, 62, 62, 62]],
            lon=[[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]],
        )

        # Move along inner dimension (longitude)
        dx = g.dx(x=[1, 1.5, 2], y=[1, 1, 1])
        dy = g.dy(x=[1, 1.5, 2], y=[1, 1, 1])
        assert dx.round(-1).tolist() == [54110, 54110, 54110]
        assert dy.round(-1).tolist() == [111430, 111430, 111430]

        # Move along outer dimension (latitude)
        dx = g.dx(x=[2, 2, 2], y=[0.5, 1, 1.5])
        dy = g.dy(x=[2, 2, 2], y=[0.5, 1, 1.5])
        assert dx.round(-1).tolist() == [54950, 54110, 53250]
        assert dy.round(-1).tolist() == [111420, 111430, 111440]

    def test_extrapolates_as_constant(self):
        g = grid.ArrayGrid(
            lat=[[60, 60, 60, 60], [61, 61, 61, 61], [62, 62, 62, 62]],
            lon=[[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]],
        )

        # Move along outer dimension (latitude)
        dx = g.dx(x=[2, 2, 2, 2, 2], y=[-1, 0, 1, 2, 3])
        dy = g.dy(x=[2, 2, 2, 2, 2], y=[-1, 0, 1, 2, 3])
        assert dx.round(-1).tolist() == [55800, 55800, 54110, 52400, 52400]
        assert dy.round(-1).tolist() == [111420, 111420, 111430, 111440, 111440]


class Test_ArrayGrid_to_bearing:
    def test_returns_bearings_in_counterclockwise_degrees(self):
        g = grid.ArrayGrid(
            lat=[[61, 62, 63], [60, 61, 62]],
            lon=[[5, 6, 7], [6, 7, 8]],
        )

        az = [
            0,    # Along  X direction (lat+, lon+)
            90,   # Along  Y direction (lat-, lon+)
            180,  # Along -X direction (lat-, lon-)
            270,  # Along -Y direction (lat+, lon-)
        ]

        b = [
            45,  # North-East along X direction
            135,  # South-East along Y direction
            225,  # South-West along -X direction
            315,  # North-West along -Y direction
        ]

        result = g.to_bearing(x=[0] * 4, y=[0] * 4, az=az)
        assert max_dist_angle(result, b) < 1


class Test_ArrayGrid_from_bearing:
    def test_returns_azimuths_in_clockwise_degrees(self):
        g = grid.ArrayGrid(
            lat=[[61, 62, 63], [60, 61, 62]],
            lon=[[5, 6, 7], [6, 7, 8]],
        )

        az = [
            0,    # Along  X direction (lat+, lon+)
            90,   # Along  Y direction (lat-, lon+)
            180,  # Along -X direction (lat-, lon-)
            270,  # Along -Y direction (lat+, lon-)
        ]

        b = [
            45,  # North-East along X direction
            135,  # South-East along Y direction
            225,  # South-West along -X direction
            315,  # North-West along -Y direction
        ]

        result = g.from_bearing(x=[0] * 4, y=[0] * 4, b=b)
        assert max_dist_angle(result, az) < 1


class Test_bilin_inv:
    def test_can_invert_single_cell(self):
        x, y = grid.bilin_inv(
            f=np.array([3, 3, 5, 5]),
            g=np.array([30, 50, 50, 30]),
            F=np.array([[0, 10], [0, 10]]),
            G=np.array([[0, 0], [100, 100]]),
        )
        assert x.tolist() == [0.3, 0.5, 0.5, 0.3]
        assert y.tolist() == [0.3, 0.3, 0.5, 0.5]

    def test_can_invert_when_integer_coordinate(self):
        f = np.array([0, 1, 0, 2])
        g = np.array([0, 0, 10, 10])
        F = np.array([[0, 1, 2], [0, 1, 2]])
        G = np.array([[0, 0, 0], [10, 10, 10]])
        x, y = grid.bilin_inv(f, g, F, G)
        i = x.round().astype(int)
        j = y.round().astype(int)

        # Assert i and j are practically equal to x and y
        assert np.abs(i - x).max() < 1e-7
        assert np.abs(j - y).max() < 1e-7

        # Check that F, G interpolated to i, j gives f, g
        assert F[i, j].tolist() == f.tolist()
        assert G[i, j].tolist() == g.tolist()


class Test_array_lookup:
    def test_correct_when_no_frac(self):
        arr = np.array([
            [0, 10, 20],
            [0, 10, 20],
            [0, 10, 20],
            [0, 10, 20],
            [0, 10, 20],
            [0, 10, 20],
            [0, 10, 20],
            [0, 100, 300],
        ])
        values = np.array([-5, 0, 5, 10, 15, 20, 25, 100])
        idx = grid.array_lookup(arr, values)
        assert idx.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]

    def test_correct_when_frac(self):
        arr = np.array([
            [0, 10, 20],
            [0, 10, 20],
            [0, 10, 20],
            [0, 10, 20],
            [0, 10, 20],
        ])
        values = np.array([-5, 5, 15, 20, 25])
        idx, frac = grid.array_lookup(arr, values, return_frac=True)
        assert idx.tolist() == [0, 0, 1, 1, 1]
        assert frac.tolist() == [0, .5, .5, 1, 1]


class Test_compute_dx_dy:
    def test_matches_test_value(self):
        lon = np.array([[5, 6], [5, 6], [5, 6]])
        lat = np.array([[60, 60], [61, 61], [62, 62]])
        dx, dy = grid.compute_dx_dy(lon=lon, lat=lat)

        assert np.round(dx, -1).tolist() == [[55800], [54110], [52400]]
        assert np.round(dy, -1).tolist() == [[111420] * 2, [111440] * 2]


def max_dist_angle(a, b):
    return max_norm_angle(a - b)


def max_norm_angle(a):
    b = zero_centric_angle(a)
    return np.max(np.abs(b))


def zero_centric_angle(a):
    return ((a % 360) + 180) % 360 - 180
