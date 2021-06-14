from pathlib import Path
import numpy as np
from netCDF4 import Dataset

import pytest

import ladim2.ROMS as force
from ladim2.timekeeper import TimeKeeper


@pytest.fixture(scope="module")
def nc_files():
    # Setup, make netCDF files with time records
    recnr = 0
    for i in range(3, 10):
        fname = f"file_{i:03d}.nc"
        with Dataset(fname, mode="w") as nc:
            nc.createDimension("time", None)
            v = nc.createVariable("ocean_time", float, ("time",))
            v.units = "seconds since 2000-01-01 00:00:00"
            for n in range(2 + i % 3):
                v[n] = recnr * 3600
                recnr += 1
    yield
    # Remove the files
    for i in range(3, 10):
        Path(f"file_{i:03d}.nc").unlink()


def test_find_files(nc_files):

    # Test single file
    files = force.find_files("file_005.nc")
    assert files == [Path("file_005.nc")]

    files = force.find_files("file_*.nc")
    assert len(files) == 7
    assert files[0] == Path("file_003.nc")

    files = force.find_files("file_*.nc", first_file="file_005.nc")
    assert len(files) == 5
    assert files[0] == Path("file_005.nc")

    files = force.find_files("file_*.nc", last_file="file_005.nc")
    assert len(files) == 3
    assert files[-1] == Path("file_005.nc")

    files = force.find_files(
        "file_*.nc", first_file="file_005.nc", last_file="file_007.nc"
    )
    assert len(files) == 3
    assert files[0] == Path("file_005.nc")
    assert files[-1] == Path("file_007.nc")


def test_scan_file_times(nc_files):

    # Everything is OK
    files = force.find_files("file_*.nc")
    all_frames, num_frames = force.scan_file_times(files)
    assert len(all_frames) == sum([2 + i % 3 for i in range(3, 10)])
    assert all_frames[0] == np.datetime64("2000-01-01")
    assert all_frames[4] == np.datetime64("2000-01-01 04")
    assert all(np.unique(all_frames) == all_frames)
    assert len(num_frames) == len(files)
    assert num_frames[Path("file_005.nc")] == 2 + 5 % 3

    # Time frames not ordered
    files = [Path("file_005.nc"), Path("file_003.nc")]
    with pytest.raises(SystemExit):
        all_frames, time_frames = force.scan_file_times(files)

    # Duplicate time frames
    files = [Path("file_005.nc"), Path("file_005.nc")]
    with pytest.raises(SystemExit):
        all_frames, time_frames = force.scan_file_times(files)


def test_forcing_steps(nc_files):
    files = force.find_files("file_*.nc")
    all_frames, num_frames = force.scan_file_times(files)

    # Forward time
    timer = TimeKeeper(start="2000-01-01 03", stop="2000-01-01 11", dt=1800)
    steps, file_idx, frame_idx = force.forcing_steps(files, timer)
    assert len(steps) == len(all_frames)
    assert steps == [2 * (i - 3) for i in range(len(all_frames))]

    for step in [-4, 0, 4, 16]:
        with Dataset(file_idx[step]) as nc:
            timeval = nc.variables["ocean_time"][frame_idx[step]]
            filetime = np.datetime64("2000-01-01") + np.timedelta64(int(timeval), "s")
            assert filetime == timer.start_time + step * timer.dt

    # Reversed time
    timer = TimeKeeper(
        start="2000-01-01 11", stop="2000-01-01 03", dt=1800, time_reversal=True
    )
    steps, file_idx, frame_idx = force.forcing_steps(files, timer)
    assert steps == [2 * (11 - i) for i in range(len(all_frames))]
    print(steps)

    for step in [-4, 0, 4, 16]:
        with Dataset(file_idx[step]) as nc:
            timeval = nc.variables["ocean_time"][frame_idx[step]]
            filetime = np.datetime64("2000-01-01") + np.timedelta64(int(timeval), "s")
            assert filetime == timer.start_time - step * timer.dt
