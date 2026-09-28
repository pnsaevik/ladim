import numpy as np
import pytest

from ladim.ibms import light


@pytest.mark.parametrize("time", [
    "2026-01-15T00:00", "2026-03-21T06:00", "2026-06-21T12:00",
    "2026-09-01T18:00", "2026-12-21T23:00",
])
def test_matches_numpy_reference(time):
    lon, lat = np.meshgrid(np.linspace(-30, 40, 141), np.linspace(40, 85, 91))
    new = light.surface_light(np.datetime64(time), lon, lat)
    ref = light.surface_light_numpy(np.datetime64(time), lon, lat)
    assert new.shape == ref.shape
    assert np.allclose(new, ref, rtol=1e-12, atol=0)


def test_accepts_scalars_and_lists():
    time = np.datetime64("2026-06-21T12:00")
    assert light.surface_light(time, 5.0, 60.0) == pytest.approx(
        light.surface_light_numpy(time, 5.0, 60.0), rel=1e-12)
    assert np.allclose(light.surface_light(time, [5.0, 6.0], [60.0, 61.0]),
                       light.surface_light_numpy(time, [5.0, 6.0], [60.0, 61.0]))


def test_nan_positions_give_zero_like_reference():
    time = np.datetime64("2026-06-21T12:00")
    lat = np.array([60.0, np.nan])
    lon = np.array([5.0, 5.0])
    assert np.array_equal(light.surface_light(time, lon, lat),
                          light.surface_light_numpy(time, lon, lat))
