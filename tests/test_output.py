import pytest
import numpy as np
import typing
from ladim import output


class Test_RaggedOutput_update:
    @pytest.fixture()
    def mock_model(self):
        model = MockObj()  # type: typing.Any

        model.state = MockObj()

        model.grid = MockObj()
        model.grid.ll2xy = lambda lon, lat: (100*lon, 10*lat)
        model.grid.ingrid = lambda X, Y: np.ones(len(X), bool)

        model.solver = MockObj()

        return model

    def test_writes_release_time_of_new_particles(self):
        # Define model
        model = MockObj()  # type: typing.Any
        model.state = MockObj()
        model.state.released = 2
        model.state.size = 2
        model.state['pid'] = np.array([0, 1])
        model.solver = MockObj()
        model.solver.time = np.datetime64('2000-01-01', 's').astype('int64')
        model.solver.step = 60
        model.output = MockObj()
        model.output = output.RaggedOutput(model, variables=dict(), file="", frequency=0)

        try:
            # Run update
            model.output.update()

            # Confirm effect on output file
            dset = model.output.dataset
            assert 'release_time' in dset.variables
            assert dset['release_time'].units == "seconds since 1970-01-01"
            assert dset['release_time'].long_name == "particle release time"
            assert dset['release_time'][:].astype('datetime64[s]').astype(str).tolist() == [
                '2000-01-01T00:00:00', '2000-01-01T00:00:00',
            ]

            # Add 3 new particles and kill 1 old
            model.solver.time += model.solver.step
            model.state.released = 5
            model.state.size = 4
            model.state['pid'] = np.array([0, 2, 3, 4])
            model.output.update()

            # Confirm effect on output file
            assert dset['release_time'][:].astype('datetime64[s]').astype(str).tolist() == (
                ['2000-01-01T00:00:00'] * 2 + ['2000-01-01T00:01:00'] * 3
            )

        finally:
            model.output.close()


class MockObj:
    def __init__(self):
        self._dict = dict()

    def __getitem__(self, item):
        return self._dict[item]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __contains__(self, item):
        return item in self._dict
