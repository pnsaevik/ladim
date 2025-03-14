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

        out = output.Output(variables=dict(), file="", frequency=0)

        try:
            # Run update
            out.update(model)

            # Confirm effect on output file
            dset = out.dataset
            assert 'release_time' in dset.variables
            assert dset['release_time'].dimensions == ('particle', )
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
            out.update(model)

            # Confirm effect on output file
            assert dset['release_time'][:].astype('datetime64[s]').astype(str).tolist() == (
                ['2000-01-01T00:00:00'] * 2 + ['2000-01-01T00:01:00'] * 3
            )

        finally:
            out.close()

    def test_writes_time_and_particle_count_for_each_timestep(self):
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
        model.output = output.Output(variables=dict(), file="", frequency=0)

        try:
            # Run update
            model.output.update(model)

            # Confirm effect on output file
            dset = model.output.dataset
            assert 'particle_count' in dset.variables
            assert dset['particle_count'].dimensions == ('time',)
            assert dset['particle_count'].long_name == "number of particles in a given timestep"
            assert dset['particle_count'][:].tolist() == [2]

            assert 'time' in dset.variables
            assert dset['time'].dimensions == ('time',)
            assert dset['time'].units == "seconds since 1970-01-01"
            assert dset['time'].long_name == "time"
            assert dset['time'][:].astype('datetime64[s]').astype(str).tolist() == [
                '2000-01-01T00:00:00'
            ]

            # Add 3 new particles and kill 1 old
            model.solver.time += model.solver.step
            model.state.released = 5
            model.state.size = 4
            model.state['pid'] = np.array([0, 2, 3, 4])
            model.output.update(model)

            # Confirm effect on output file
            assert dset['particle_count'][:].tolist() == [2, 4]
            assert dset['time'][:].astype('datetime64[s]').astype(str).tolist() == [
                '2000-01-01T00:00:00', '2000-01-01T00:01:00',
            ]

        finally:
            model.output.close()

    @pytest.mark.parametrize("kind", ['instance', 'initial'])
    def test_writes_particles_as_instance_or_init_vars(self, kind):
        # Define model
        model = MockObj()  # type: typing.Any
        model.state = MockObj()
        model.state.released = 2
        model.state.size = 2
        model.state['pid'] = np.array([0, 1])
        model.state['X'] = np.array([10, 20])
        model.solver = MockObj()
        model.solver.time = np.datetime64('2000-01-01', 's').astype('int64')
        model.solver.step = 60
        model.output = MockObj()
        model.output = output.Output(
            variables=dict(X=dict(units='m', long_name='x coord', kind=kind)),
            file="",
            frequency=0,
        )

        try:
            # Run update
            model.output.update(model)

            # Confirm effect on output file
            dset = model.output.dataset
            assert 'X' in dset.variables
            assert dset['X'].units == "m"
            assert dset['X'].long_name == "x coord"
            assert dset['X'][:].tolist() == [10, 20]

            # Add 3 new particles and kill 1 old
            model.solver.time += model.solver.step
            model.state.released = 5
            model.state.size = 4
            model.state['pid'] = np.array([0, 2, 3, 4])
            model.state['X'] = np.array([100, 200, 300, 400])
            model.output.update(model)

            # Confirm effect on output file
            if kind == 'initial':
                assert dset['X'].dimensions == ('particle',)
                assert dset['X'][:].tolist() == [10, 20, 200, 300, 400]
            elif kind == 'instance':
                assert dset['X'].dimensions == ('particle_instance',)
                assert dset['X'][:].tolist() == [10, 20, 100, 200, 300, 400]
            else:
                raise AssertionError(f'Wrong kind: {kind}')

        finally:
            model.output.close()

    def test_writes_only_at_output_points(self):
        # Define model
        model = MockObj()  # type: typing.Any
        model.state = MockObj()
        model.state.released = 2
        model.state.size = 2
        model.state['pid'] = np.array([0, 1])
        model.state['X'] = np.array([10, 20])
        model.solver = MockObj()
        model.solver.time = np.datetime64('2000-01-01', 's').astype('int64')
        model.solver.step = 60
        model.output = MockObj()
        model.output = output.Output(
            variables=dict(X=dict(units='m', long_name='x coord')),
            file="",
            frequency=120,
        )

        try:
            # Writes output on first step (0 sec)
            model.output.update(model)
            assert model.output.dataset['X'][:].tolist() == [10, 20]

            # Does not write output on second step (60 sec)
            model.solver.time += model.solver.step
            model.output.update(model)
            assert model.output.dataset['X'][:].tolist() == [10, 20]

            # Writes output on third step (120 sec)
            model.solver.time += model.solver.step
            model.output.update(model)
            assert model.output.dataset['X'][:].tolist() == [10, 20, 10, 20]

        finally:
            model.output.close()

    def test_can_output_lat_and_lon(self):
        # Define model
        model = MockObj()  # type: typing.Any
        model.state = MockObj()
        model.state.released = 2
        model.state.size = 2
        model.state['pid'] = np.array([0, 1])
        model.state['X'] = np.array([1, 2])
        model.state['Y'] = np.array([3, 4])
        model.solver = MockObj()
        model.solver.time = np.datetime64('2000-01-01', 's').astype('int64')
        model.grid = MockObj()
        model.grid.xy2ll = lambda x, y: (x + 10, y + 70)
        model.output = MockObj()
        model.output = output.Output(
            variables=dict(lat=dict(), lon=dict()),
            file="",
            frequency=0,
        )

        try:
            # Run update
            model.output.update(model)

            # Confirm effect on output file
            dset = model.output.dataset
            assert dset['lat'].dimensions == ('particle_instance',)
            assert dset['lat'][:].tolist() == [73, 74]
            assert dset['lon'].dimensions == ('particle_instance',)
            assert dset['lon'][:].tolist() == [11, 12]

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
