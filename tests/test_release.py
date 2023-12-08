from ladim import release
import numpy as np
import pytest
import io
import typing


class Test_sorted_interval:
    @pytest.mark.parametrize("ab", [
        (1.5, 4.5), (1, 4.5), (1.5, 5), (-1, 4.5), (1.5, 9), (-2, -1), (8, 9),
    ])
    def test_correct_interval_when_sorted_array(self, ab):
        v = [0, 1, 2, 3, 4, 5, 6]
        a, b = ab
        start, stop = release.sorted_interval(v, a, b)

        w = np.array(v)
        expected = w[(w >= a) & (w < b)].tolist()
        assert v[start:stop] == expected


class Test_TextFileReleaser_update:
    @pytest.fixture()
    def mock_model(self):
        import ladim.state

        model = MockObj()  # type: typing.Any

        model.state = ladim.state.DynamicState(model)

        model.grid = MockObj()
        model.grid.ll2xy = lambda lon, lat: (100*lon, 10*lat)
        model.grid.ingrid = lambda X, Y: np.ones(len(X), bool)

        model.solver = MockObj()
        model.solver.time = np.datetime64('2000-01-01', 's').astype('int64')
        model.solver.step = 60

        return model

    def test_converts_latlon_colnames_to_xy(self, mock_model):
        # Create mock release file
        buf = io.StringIO(
            'release_time lat lon\n'
            '2000-01-01 60 4\n'
            '2000-01-01 61 5\n'
        )

        # Run releaser update
        releaser = release.TextFileReleaser(model=mock_model, file=buf)
        releaser.update()

        # Confirm effect on release module
        assert 'lat' not in mock_model.state
        assert list(mock_model.state['X']) == [400, 500]
        assert list(mock_model.state['Y']) == [600, 610]


class MockObj:
    pass
