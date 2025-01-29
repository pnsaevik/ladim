from ladim import grid


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

