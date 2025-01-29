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

