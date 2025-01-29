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
