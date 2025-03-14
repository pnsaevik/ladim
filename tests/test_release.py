from ladim import release
import numpy as np
import pytest
import io
import typing


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
        releaser = release.Releaser.from_textfile(
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
        releaser = release.Releaser.from_textfile(
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
        releaser = release.Releaser.from_textfile(
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
        releaser = release.Releaser.from_textfile(file=buf)
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
        releaser = release.Releaser.from_textfile(file=buf)
        releaser.update(mock_model)
        assert list(mock_model.state['X']) == [60, 61]

        # Mark particle 0 as dead and run releaser update
        mock_model.state['alive'][0] = False
        mock_model.solver.time += mock_model.solver.step
        releaser.update(mock_model)

        # Confirm effect on state module
        assert list(mock_model.state['X']) == [61]


class Test_resolve_schedule:
    def test_correct_when_all_events_are_specified(self):
        e = list(release.resolve_schedule(
            times=[0, 1, 1, 6, 9],
            interval=10,
            start_time=0,
            stop_time=10,
        ))
        assert e == [0, 1, 2, 3, 4]

    def test_correct_when_intermediate_points(self):
        e = release.resolve_schedule(
            times=[0, 1, 1, 1, 6, 6, 9],
            interval=2,
            start_time=0,
            stop_time=10,
        ).tolist()
        assert e == [0] + [1, 2, 3] * 3 + [4, 5] * 2 + [6]

    def test_correct_when_before_schedule(self):
        # Repetition should start only after first time
        e = release.resolve_schedule(
            times=[0, 0, 3, 3, 3],
            interval=2,
            start_time=-4,
            stop_time=6,
        ).tolist()
        assert e == [0, 1] * 2 + [2, 3, 4] * 2

    def test_correct_when_no_interval(self):
        e = release.resolve_schedule(
            times=[0, 0, 1, 1, 1, 2, 3, 4, 5],
            interval=0,
            start_time=1,
            stop_time=4,
        ).tolist()
        assert e == [2, 3, 4, 5, 6]

        e = release.resolve_schedule(
            times=[0, 0, 1, 1, 1, 2, 3, 4, 5],
            interval=0,
            start_time=0,
            stop_time=10,
        ).tolist()
        assert e == [0, 1, 2, 3, 4, 5, 6, 7, 8]


class Test_Schedule:
    def test_can_expand(self):
        # Event 0 and 1 are repeated twice when interval = 2
        s1 = release.Schedule(times=np.arange(3) * 3, events=np.arange(3))
        s2 = s1.expand(interval=2, stop=8)
        assert s2.times.tolist() == [0, 2, 3, 5, 6]
        assert s2.events.tolist() == [0, 0, 1, 1, 2]

        # No elements in array; no expansion is possible
        s1 = release.Schedule(times=np.array([]), events=np.array([]))
        s2 = s1.expand(interval=2, stop=8)
        assert s2.times.tolist() == []
        assert s2.events.tolist() == []

    def test_is_invalid_if_nondecreasing(self):
        s = release.Schedule(times=np.array([3, 2, 1]), events=np.arange(3))
        assert not s.valid()
        s2 = release.Schedule(times=np.zeros(3), events=np.zeros(3))
        assert s2.valid()

    def test_can_append(self):
        s1 = release.Schedule(np.arange(3), np.arange(3) * 2)
        s2 = release.Schedule(np.arange(4), np.arange(4) * 2)
        s3 = s1.append(s2)
        assert s3.times.tolist() == [0, 1, 2, 0, 1, 2, 3]
        assert s3.events.tolist() == [0, 2, 4, 0, 2, 4, 6]

    def test_can_extend_backwards(self):
        s1 = release.Schedule(np.ones(3), np.arange(3))
        s2 = s1.extend_backwards(0)
        assert s2.times.tolist() == [0, 0, 0, 1, 1, 1]
        assert s2.events.tolist() == [0, 1, 2, 0, 1, 2]

    def test_can_extend_backwards_using_interval(self):
        s1 = release.Schedule(np.ones(3) * 4, np.arange(3))
        s2 = s1.extend_backwards_using_interval(
            time=1,
            interval=2,
        )
        assert s2.times.tolist() == [0, 0, 0, 4, 4, 4]
        assert s2.events.tolist() == [0, 1, 2, 0, 1, 2]

    def test_can_trim_tail(self):
        s1 = release.Schedule(np.arange(5), np.arange(5))

        assert s1.trim_tail(4.0).times.tolist() == [0, 1, 2, 3]
        assert s1.trim_tail(3.0).times.tolist() == [0, 1, 2]
        assert s1.trim_tail(2.5).times.tolist() == [0, 1, 2]

    def test_can_rightshift_time_values(self):
        s1 = release.Schedule(np.array([0, 10, 10, 20, 30]), np.arange(5))

        # Largest time value smaller than or equal to 17 is 10
        # We add N=4 intervals to this time so that it becomes >= 17
        s2 = s1.rightshift_closest_time_value(time=17, interval=2)
        assert s2.times.tolist() == [0, 18, 18, 20, 30]

        # Largest time value smaller than or equal to 17 is 10
        # We could add N=1 intervals to this time so that it becomes >= 17
        # ..but then the new value would become 25, which is larger than the
        # next value, so this is not done. Instead, no elements are changed.
        s2 = s1.rightshift_closest_time_value(time=17, interval=15)
        assert s2.times.tolist() == [0, 10, 10, 20, 30]

        # Largest time value smaller than or equal to 10 is 10
        # This element does not have to change, it is already <= 10
        s2 = s1.rightshift_closest_time_value(time=10, interval=2)
        assert s2.times.tolist() == [0, 10, 10, 20, 30]

        # No time value is smaller than -5
        # No elements are changed
        s2 = s1.rightshift_closest_time_value(time=-5, interval=2)
        assert s2.times.tolist() == [0, 10, 10, 20, 30]

        # Interval = 0 means nothing should change
        s2 = s1.rightshift_closest_time_value(time=17, interval=0)
        assert s2.times.tolist() == [0, 10, 10, 20, 30]

        # No elements in schedule; nothing should change
        s3 = release.Schedule(times=np.array([]), events=np.array([]))
        s2 = s3.rightshift_closest_time_value(time=17, interval=2)
        assert s2.times.tolist() == []

        # One element in schedule; should be shifted
        s3 = release.Schedule(times=np.array([10]), events=np.array([0]))
        s2 = s3.rightshift_closest_time_value(time=17, interval=2)
        assert s2.times.tolist() == [18]

    def test_can_trim_head(self):
        s = release.Schedule(
            times=np.array([10, 10, 20, 20, 20, 30, 40]),
            events=np.array([0, 1, 2, 3, 4, 5, 6]),
        )

        # Start time equals first time; no trimming
        s2 = s.trim_head(start_time=10)
        assert s2.times.tolist() == [10, 10, 20, 20, 20, 30, 40]
        assert s2.events.tolist() == [0, 1, 2, 3, 4, 5, 6]

        # Start time before first time; no trimming
        s2 = s.trim_head(start_time=5)
        assert s2.times.tolist() == [10, 10, 20, 20, 20, 30, 40]
        assert s2.events.tolist() == [0, 1, 2, 3, 4, 5, 6]

        # Start time after first time; remove first entries
        s2 = s.trim_head(start_time=30)
        assert s2.times.tolist() == [30, 40]
        assert s2.events.tolist() == [5, 6]

        # Start time after last time; remove all entries
        s2 = s.trim_head(start_time=50)
        assert s2.times.tolist() == []
        assert s2.events.tolist() == []


class MockObj:
    pass
