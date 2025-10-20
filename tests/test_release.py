from ladim import release
import numpy as np
import pytest
import io
import typing
import netCDF4 as nc
import pandas as pd


class Test_Releaser_create:
    def test_accepts_empty_parameter_list(self):
        # Create empty releaser
        r = release.Releaser.create()

        # Retrieve particles from large time span
        df = r.new_particles(start=0, stop=int(1e10))

        # Check that particle list is empty
        assert df.shape[0] == 0


    def test_singular_releases(self):
        # Create releaser with irregular release times
        table = {'release_time': [0, 2, 5, 7, 10]}
        r = release.Releaser.create(table)

        # Retrieve particles from various time spans and confirm correct count
        
        df = r.new_particles(start=-10, stop=-1)  # Before start
        assert df.shape[0] == 0     # No particles

        df = r.new_particles(start=0, stop=1)     # Particle at start of interval
        assert df.shape[0] == 1     # One particle

        df = r.new_particles(start=0, stop=6)     # Particles at start and middle
        assert df.shape[0] == 3     # Include all

        df = r.new_particles(start=0, stop=7)     # Particles at start, middle and end
        assert df.shape[0] == 3     # Include only start and middle

        df = r.new_particles(start=1, stop=6)     # Particles at middle of interval
        assert df.shape[0] == 2     # Include both

        df = r.new_particles(start=1, stop=7)     # Particles at middle and end
        assert df.shape[0] == 2     # Include only middle

        df = r.new_particles(start=6, stop=7)     # Particle at end
        assert df.shape[0] == 0     # Include nothing
        
        df = r.new_particles(start=20, stop=30 )  # After interval
        assert df.shape[0] == 0     # No particles
        
    def test_simple_continuous_release(self):
        # Create releaser with continuous release times
        table = {'release_time': [10], 'release_interval': [3]}
        r = release.Releaser.create(table)

        # No particles before first release time
        df = r.new_particles(start=5, stop=7)
        assert df.shape[0] == 0

        # Particles at every release interval afterwards
        df = r.new_particles(start=10, stop=15)  # Start 1, middle 1
        assert df.shape[0] == 2
        df = r.new_particles(start=10, stop=19)  # Start 1, middle 2, end 1
        assert df.shape[0] == 3


class Test_reset_range_start:
    def test_can_set_new_start(self):
        start = 3
        step = 2
        limits = np.array([0, 1, 2, 3, 4, 5])
        new_start = release.reset_range_start(start, step, limits)
        assert np.asarray(new_start).tolist() == [1, 1, 3, 3, 5, 5]


class Test_add_start_stop_step_to_release_table:
    def test_adds_columns_when_discrete_spec(self):
        tab = pd.DataFrame({'release_time': [0, 10, 30]})
        result = release.add_start_stop_step_to_release_table(tab)
        
        start = result['release_start'].values
        stop = result['release_stop'].values
        step = result['release_step'].values

        assert start.tolist() == [0, 10, 30]
        assert stop[0] == 10
        assert stop[1] == 30
        assert stop[2] > 1_000_000
        assert np.all(step > 1_000_000)

    def test_adds_columns_when_continuous_spec(self):
        tab = pd.DataFrame({
            'release_time': [0, 10, 30],
            'release_interval': [7, 2, 3]
            })
        result = release.add_start_stop_step_to_release_table(tab)
        
        start = result['release_start'].values
        stop = result['release_stop'].values
        step = result['release_step'].values

        assert start.tolist() == [0, 10, 30]
        assert stop[0] == 10
        assert stop[1] == 30
        assert stop[2] > 1_000_000
        assert step.tolist() == [7, 2, 3]

    def test_can_convert_string_datetimes(self):
        tab = pd.DataFrame({'release_time': ['1970-01-01', '1970-01-02']})
        result = release.add_start_stop_step_to_release_table(tab)
        
        start = result['release_start'].values
        assert start.tolist() == [0, 86400]


class Test_truncate_schedule:
    def test_can_truncate(self):
        tab = pd.DataFrame({
            'release_start': [0, 0, 15],
            'release_stop': [10, 20, 25],
            'release_step': [2, 3, 5],
        })
        result = release.truncate_schedule_period(tab, 5, 15)
        assert result.to_dict(orient='list') == {
            'release_start': [6, 6],
            'release_stop': [10, 15],
            'release_step': [2, 3],
        }

    def test_removes_rows_outside_interval(self):
        tab = pd.DataFrame({
            'release_start': [0, 10],
            'release_stop': [5, 15],
            'release_step': [1, 1],
        })
        result = release.truncate_schedule_period(tab, 5, 10)
        assert result.to_dict(orient='records') == []

    def test_do_nothing_to_rows_inside_interval(self):
        data = {
            'release_start': [0, 10],
            'release_stop': [5, 15],
            'release_step': [2, 3],
        }
        tab = pd.DataFrame(data)
        result = release.truncate_schedule_period(tab, 0, 15)
        assert result.to_dict(orient='list') == data

    def test_shifts_start_on_overlap(self):
        data = {
            'release_start': [0, 2],
            'release_stop': [5, 15],
            'release_step': [2, 3],
        }
        tab = pd.DataFrame(data)
        result = release.truncate_schedule_period(tab, 3, 15)
        assert result.to_dict(orient='list') == {
            'release_start': [4, 5],
            'release_stop': [5, 15],
            'release_step': [2, 3],
        }

    def test_truncates_stop_on_overlap(self):
        data = {
            'release_start': [0, 2],
            'release_stop': [10, 15],
            'release_step': [2, 3],
        }
        tab = pd.DataFrame(data)
        result = release.truncate_schedule_period(tab, 0, 5)
        assert result.to_dict(orient='list') == {
            'release_start': [0, 2],
            'release_stop': [5, 5],
            'release_step': [2, 3],
        }


class Test_expand_schedule:
    def test_expands_row_by_row(self):
        data = {
            'release_start': [0, 5],
            'release_stop': [6, 7],
            'release_step': [2, 1],
            'myfield': [10, 20],
        }
        tab = pd.DataFrame(data)
        result = release.expand_schedule_range(tab)
        assert result.to_dict(orient='list') == {
            'release_time': [0, 2, 4, 5, 6],
            'myfield': [10, 10, 10, 20, 20],
        }


class Test_TextFileReleaser_update:
    @pytest.fixture()
    def mock_model(self):
        import ladim.state

        model = MockObj()  # type: typing.Any

        model.state = ladim.state.State()

        model.grid = MockObj()
        model.grid.ll2xy = lambda lon, lat: (100*lon, 10*lat)
        model.grid.ingrid = lambda X, Y: np.ones(len(X), bool)

        model.solver = MockObj()
        model.solver.time = np.datetime64('2000-01-01', 's').astype('int64')
        model.solver.stop = np.datetime64('2000-01-02', 's').astype('int64')
        model.solver.step = 60

        return model

    def test_continuous_release_resets_when_new_release_time(self, mock_model):
        mock_model.solver.time = np.datetime64('1970-01-01', 's').astype('int64')
        mock_model.solver.step = 60

        # Create mock release file
        buf = io.StringIO(
            'release_time X Y\n'
            '1970-01-01T00:01:00 4 60\n'
            '1970-01-01T00:01:00 5 61\n'
            '1970-01-01T00:05:00 6 62\n'
        )

        # Create continuous releaser
        releaser = release.Releaser.create(
            file=buf,
            frequency=(2, 'm'),
        )

        # Time step 0: No particles yet
        releaser.update(mock_model)
        assert list(mock_model.state['X']) == []

        # Time step 1: First particle release
        mock_model.solver.time += mock_model.solver.step
        releaser.update(mock_model)
        assert list(mock_model.state['X']) == [4, 5]

        # Time step 2: Intermediate step, no additional particles
        mock_model.solver.time += mock_model.solver.step
        releaser.update(mock_model)
        assert list(mock_model.state['X']) == [4, 5]

        # Time step 3: Second release, two new particles
        # Using previous release instructions
        mock_model.solver.time += mock_model.solver.step
        releaser.update(mock_model)
        assert list(mock_model.state['X']) == [4, 5, 4, 5]

        # Time step 4: Intermediate step, no additional particles
        mock_model.solver.time += mock_model.solver.step
        releaser.update(mock_model)
        assert list(mock_model.state['X']) == [4, 5, 4, 5]

        # Time step 5: New release instructions
        # One new particle at a new position, previous instructions cleared
        mock_model.solver.time += mock_model.solver.step
        releaser.update(mock_model)
        assert list(mock_model.state['X']) == [4, 5, 4, 5, 6]

    def test_converts_latlon_colnames_to_xy(self, mock_model):
        # Create mock release file
        buf = io.StringIO(
            'release_time lat lon\n'
            '2000-01-01 60 4\n'
            '2000-01-01 61 5\n'
        )

        # Run releaser update
        releaser = release.Releaser.create(
            file=buf, lonlat_converter=mock_model.grid.ll2xy)
        releaser.update(mock_model)

        # Confirm effect on state module
        assert 'lat' not in mock_model.state
        assert list(mock_model.state['X']) == [400, 500]
        assert list(mock_model.state['Y']) == [600, 610]

    def test_adds_default_values(self, mock_model):
        # Create mock release file
        buf = io.StringIO(
            'release_time X Y\n'
            '2000-01-01 60 4\n'
            '2000-01-01 61 5\n'
        )

        # Run releaser update
        releaser = release.Releaser.create(
            file=buf,
            defaults=dict(myvar=23),
        )
        releaser.update(mock_model)

        # Confirm effect on state module
        assert list(mock_model.state['myvar']) == [23, 23]

    def test_expands_multiplicity_variable(self, mock_model):
        # Mock release file
        buf = io.StringIO(
            'mult release_time X Y\n'
            '   1   2000-01-01 60 4\n'
            '   2   2000-01-01 61 5\n'
        )

        # Run releaser update
        releaser = release.Releaser.create(file=buf)
        releaser.update(mock_model)

        # Confirm effect on state module
        assert list(mock_model.state['X']) == [60, 61, 61]

    def test_removes_dead_particles(self, mock_model):
        # Create mock release file
        buf = io.StringIO(
            'release_time X Y\n'
            '2000-01-01 60 4\n'
            '2000-01-01 61 5\n'
        )

        # Run releaser update
        releaser = release.Releaser.create(file=buf)
        releaser.update(mock_model)
        assert list(mock_model.state['X']) == [60, 61]

        # Mark particle 0 as dead and run releaser update
        mock_model.state['alive'][0] = False
        mock_model.solver.time += mock_model.solver.step
        releaser.update(mock_model)

        # Confirm effect on state module
        assert list(mock_model.state['X']) == [61]

    def test_warm_start_time(self, mock_model):
        # Create mock release file
        buf = io.StringIO(
            'release_time X Y\n'
            '2000-01-01 60 4\n'
            '2000-01-01 61 5\n'
        )

        # Create mock warm start file
        data_dict = dict(time=np.array([1441587720, 1441587780]))
        dset = nc.Dataset(filename='warm_start_time.nc', mode='w', format='NETCDF4', diskless=True)
        dset.createDimension('time', len(data_dict['time']))
        time_var = dset.createVariable('time', 'i8', ('time',))
        time_var[:] = data_dict['time']
        time_var.units = 'seconds since 1970-01-01 00:00:00'

        releaser = release.Releaser.create(file=buf, warm_start_file=dset)

        assert releaser.warm_start_time() == 1441587780

    def test_from_warm_start_file(self, mock_model):
        # Create mock release file
        buf = io.StringIO(
            'release_time X Y\n'
            '2000-01-01 60 4\n'
            '2000-01-01 61 5\n'
        )

        # Create mock warm start file
        data_dict = dict(time=np.array([1441587720, 1441587780, 1441587840]),
                         particle_count=np.array([1, 2, 3]),
                         release_time=np.array([1441587720, 1441587780, 1441587840, 1441587840]),
                         pid=np.array([0, 0, 1, 0, 2, 3]),
                         X=np.array([10, 20, 30, 40, 50, 60]))
        dset = nc.Dataset(filename='from_warm_start_file.nc', mode='w', format='NETCDF4', diskless=True)
        dset.createDimension('time', len(data_dict['time']))
        dset.createDimension('particle', len(data_dict['release_time']))
        dset.createDimension('particle_instance', len(data_dict['pid']))
        time_var = dset.createVariable('time', 'i8', ('time',))
        time_var[:] = data_dict['time']
        time_var.units = 'seconds since 1970-01-01 00:00:00'
        particle_count_var = dset.createVariable('particle_count', 'i4', ('time',))
        particle_count_var[:] = data_dict['particle_count']
        release_time_var = dset.createVariable('release_time', 'i8', ('particle',))
        release_time_var[:] = data_dict['release_time']
        release_time_var.units = 'seconds since 1970-01-01 00:00:00'
        pid_var = dset.createVariable('pid', 'i4', ('particle_instance',))
        pid_var[:] = data_dict['pid']
        x_var = dset.createVariable('X', 'f4', ('particle_instance',))
        x_var[:] = data_dict['X']

        releaser = release.Releaser.create(file=buf, warm_start_file=dset)
        releaser.from_warm_start_file(mock_model)

        assert mock_model.state['pid'].tolist() == [0, 2, 3]
        assert mock_model.state['X'].tolist() ==  [40, 50, 60]
        assert mock_model.state['release_time'].tolist() ==  [1441587720, 1441587840, 1441587840]
        assert mock_model.state['active'].tolist() == [True, True, True]
        assert mock_model.state['alive'].tolist() == [True, True, True]
        assert mock_model.state.released == 4


class MockObj:
    pass
