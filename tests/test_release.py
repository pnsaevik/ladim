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

        model.state = ladim.state.DynamicState()

        model.grid = MockObj()
        model.grid.ll2xy = lambda lon, lat: (100*lon, 10*lat)
        model.grid.ingrid = lambda X, Y: np.ones(len(X), bool)

        model.solver = MockObj()
        model.solver.time = np.datetime64('2000-01-01', 's').astype('int64')
        model.solver.stop = np.datetime64('2000-01-02', 's').astype('int64')
        model.solver.step = 60

        return model

    def test_continuous_release_resets_when_new_release_time(self, mock_model):
        # Create mock release file
        buf = io.StringIO(
            'release_time X Y\n'
            '2000-01-01T00:01:00 4 60\n'
            '2000-01-01T00:01:00 5 61\n'
            '2000-01-01T00:05:00 6 62\n'
        )

        # Create continuous releaser
        releaser = release.TextFileReleaser(
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
        releaser = release.TextFileReleaser(file=buf)
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
        releaser = release.TextFileReleaser(
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
        releaser = release.TextFileReleaser(file=buf)
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
        releaser = release.TextFileReleaser(file=buf)
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
        e = release.resolve_schedule(
            times=[0, 0, 1, 1, 1],
            interval=2,
            start_time=-4,
            stop_time=4,
        ).tolist()
        assert e == [0, 1] * 3 + [2, 3, 4] * 2


class Test_Schedule:
    def test_can_expand(self):
        s = release.Schedule(times=np.arange(3) * 3, events=np.arange(3))
        s2 = s.expand(interval=2, stop=8)
        assert s2.times.tolist() == [0, 2, 3, 5, 6]
        assert s2.events.tolist() == [0, 0, 1, 1, 2]

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

    def test_can_trim_head(self):
        s1 = release.Schedule(np.arange(5), np.arange(5))

        assert s1.trim_head(1.0).times.tolist() == [1, 2, 3, 4]
        assert s1.trim_head(0.5).times.tolist() == [0, 1, 2, 3, 4]

class MockObj:
    pass
