import pytest
import numpy as np
import typing
from ladim import output
import netCDF4 as nc


class Test_Output_update:
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
            dset = out.writer.paths[0]
            assert 'release_time' in dset.variables
            assert dset['release_time'].dimensions == ('particle', )
            assert dset['release_time'].units == "seconds since 1970-01-01 00:00:00"
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
            dset = model.output.writer.paths[0]
            assert 'particle_count' in dset.variables
            assert dset['particle_count'].dimensions == ('time',)
            assert dset['particle_count'].long_name == "number of particles in a given timestep"
            assert dset['particle_count'][:].tolist() == [2]

            assert 'time' in dset.variables
            assert dset['time'].dimensions == ('time',)
            assert dset['time'].units == "seconds since 1970-01-01 00:00:00"
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
            dset = model.output.writer.paths[0]
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
            dset = model.output.writer.paths[0]
            assert dset['X'][:].tolist() == [10, 20]

            # Does not write output on second step (60 sec)
            model.solver.time += model.solver.step
            model.output.update(model)
            assert dset['X'][:].tolist() == [10, 20]

            # Writes output on third step (120 sec)
            model.solver.time += model.solver.step
            model.output.update(model)
            assert dset['X'][:].tolist() == [10, 20, 10, 20]

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
            dset = model.output.writer.paths[0]
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


class Test_Writer:
    def test_netcdf_writer(self):
        variables = dict(
            x=output.OutputFormat(ncformat='f4', dimensions='mydim')
        )
        w = output.Writer.netcdf(file="", formats=variables)
        w.write_instance(dict(x=np.array([1.0, 2.0, 3.0])))
        w.write_instance(dict(x=np.array([4.0, 5.0])))

        assert w.paths[0].variables['x'][:].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert w.sizes == {'mydim': 5}
        assert len(w.paths) == 1
        w.close()

    def test_mf_netcdf_writer_can_append_particles(self):
        variables = dict(x=output.OutputFormat(ncformat='f4', dimensions='xd'))
        
        # Write four times with numrec = 2
        w = output.Writer.mf_netcdf(file="", formats=variables, numrec=2)
        w.write_instance(dict(x=np.array([1, 2, 3])))
        w.write_instance(dict(x=np.array([4, 5])))
        w.write_instance(dict(x=np.array([6, 7, 8])))
        w.write_instance(dict(x=np.array([9])))

        # Read x values
        x_values = []
        for dset in w.paths:
            x_values += dset.variables['x'][:].tolist()
        
        assert x_values == [1, 2, 3, 4, 5, 6, 7, 8, 9]

        w.close()

    def test_mf_netcdf_writer_splits_output_on_multiple_files(self):
        variables = dict(x=output.OutputFormat(ncformat='f4', dimensions='xd'))
        
        # Write two times with numrec = 2; single file
        w = output.Writer.mf_netcdf(file="", formats=variables, numrec=2)
        w.write_instance(dict(x=np.array([1, 2, 3])))
        w.write_instance(dict(x=np.array([4, 5])))
        assert len(w.paths) == 1
        assert w.paths[0].variables['x'][:].tolist() == [1, 2, 3, 4, 5]

        # Write once more; two files
        w.write_instance(dict(x=np.array([6, 7, 8])))
        assert len(w.paths) == 2
        assert w.paths[1].variables['x'][:].tolist() == [6, 7, 8]

        w.close()

    def test_mf_netcdf_writer_does_not_split_on_init_writes(self):
        variables = dict(x=output.OutputFormat(ncformat='f4', dimensions='xd'))
        
        # Write three times with numrec = 2; still a single file
        w = output.Writer.mf_netcdf(file="", formats=variables, numrec=2)
        w.write_init(dict(x=np.array([1, 2, 3])))
        w.write_init(dict(x=np.array([4, 5])))
        w.write_init(dict(x=np.array([6, 7, 8])))
        assert len(w.paths) == 1
        assert w.paths[0].variables['x'][:].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]

        w.close()

    def test_mf_netcdf_writer_returns_total_sizes(self):
        variables = dict(x=output.OutputFormat(ncformat='f4', dimensions='xd'))
        
        # Write three times with numrec = 2; two files
        w = output.Writer.mf_netcdf(file="", formats=variables, numrec=2)
        w.write_instance(dict(x=np.array([1, 2, 3])))
        w.write_instance(dict(x=np.array([4, 5])))
        w.write_instance(dict(x=np.array([6, 7, 8])))

        assert w.sizes['xd'] == 8

        w.close()

    def test_mf_netcdf_writer_copies_particle_table(self):
        variables = dict(x=output.OutputFormat(ncformat='f4', dimensions='xd'))
        
        # Write three times with numrec = 2
        w = output.Writer.mf_netcdf(
            file="",
            formats=variables,
            numrec=2,
            copy_dims=('xd', ),
            )
        w.write_instance(dict(x=np.array([1, 2, 3])))
        w.write_instance(dict(x=np.array([4, 5])))
        w.write_instance(dict(x=np.array([6, 7, 8])))

        assert w.paths[0].variables['x'][:].tolist() == [1, 2, 3, 4, 5]
        assert w.paths[1].variables['x'][:].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
        assert w.sizes['xd'] == 8

        w.close()

    def test_mf_netcdf_writer_updates_offset_variable(self):
        variables = dict(
            x=output.OutputFormat(ncformat='f4', dimensions='xd'),
            xd_offset=output.OutputFormat(ncformat='i4', dimensions='')
            )
        
        # Write three times with numrec = 2
        w = output.Writer.mf_netcdf(
            file="",
            formats=variables,
            numrec=2,
            offset_variables={'xd': 'xd_offset'}
        )
        w.write_instance(dict(x=np.array([1, 2, 3])))
        w.write_instance(dict(x=np.array([4, 5])))
        w.write_instance(dict(x=np.array([6, 7, 8])))

        assert w.paths[0].variables['xd_offset'][...] == 0
        assert w.paths[1].variables['xd_offset'][...] == 5

        w.close()
