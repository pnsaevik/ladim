from pathlib import Path
import subprocess

import numpy as np
from netCDF4 import Dataset
import pytest

from ladim2.state import State
from ladim2.timekeeper import TimeKeeper
from ladim2.out_netcdf import fname_gnrt, Output

NCFILE = Path("output_test.nc")


class Dummy:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


config0 = dict(
    modules=dict(
        time=TimeKeeper(start="2020-01-01 12", stop="2020-01-03 12", dt=1800),
        release=Dummy(total_particle_count=3),
        grid=None,
    ),
    filename=NCFILE,
    output_period=np.timedelta64(12, "h"),
    instance_variables=dict(
        pid=dict(
            encoding=dict(datatype="i", zlib=True),
            attributes=dict(long_name="particle_identifier"),
        ),
        X=dict(
            encoding=dict(datatype="f4"),
            attributes=dict(long_name="particle X-coordinate"),
        ),
    ),
    particle_variables=dict(
        X0=dict(
            encoding=dict(datatype="f4"),
            attributes=dict(long_name="inital X-position"),
        ),
    ),
    global_attributes=dict(institution="Institute of Marine Research", source="LADiM"),
)


def test_filename():
    """Test the filename generator"""
    # No digits
    g = fname_gnrt(Path("kake.nc"))
    assert next(g) == Path("kake_000.nc")
    assert next(g) == Path("kake_001.nc")
    # Trailing underscore, no digits. Gives extra underscore.
    g = fname_gnrt(Path("kake_.nc"))
    assert next(g) == Path("kake__000.nc")
    # Full file path with digits
    g = fname_gnrt(Path("output/kake_0023.nc"))
    assert next(g) == Path("output/kake_0023.nc")
    assert next(g) == Path("output/kake_0024.nc")
    # Number width too small
    # should perhaps raise exception as filenames will not sort properly
    g = fname_gnrt(Path("kake_8.nc"))
    assert next(g) == Path("kake_8.nc")
    assert next(g) == Path("kake_9.nc")
    assert next(g) == Path("kake_10.nc")
    # Number and no underscore
    g = fname_gnrt(Path("kake42.nc"))
    assert next(g) == Path("kake42_000.nc")
    assert next(g) == Path("kake42_001.nc")
    # Several numbers and underscores
    g = fname_gnrt(Path("kake_42_007.nc"))
    assert next(g) == Path("kake_42_007.nc")
    assert next(g) == Path("kake_42_008.nc")


def test_output_init():
    """Test module initialization"""
    out = Output(**config0)

    # Check some attributes
    assert out.filename == NCFILE
    assert set(out.instance_variables) == {"pid", "X"}
    assert out.output_period_step == 24  # 12 h / 0.5 h
    out.close()
    NCFILE.unlink()


def test_file_creation():
    """Test the file creation part of the initialization"""

    out = Output(**config0)
    # Close the file before testing
    out.close()

    # The file exist and is recognized by ncdump as a netcdf file
    assert NCFILE.exists()
    assert (
        subprocess.run(
            ["ncdump", "-h", str(NCFILE)], stdout=subprocess.DEVNULL, shell=False
        ).returncode
        == 0
    )

    # Check some of the content
    with Dataset(NCFILE) as nc:
        assert nc.data_model == "NETCDF4"
        assert set(nc.dimensions) == {"time", "particle_instance", "particle"}
        assert len(nc.dimensions["time"]) == 0  # Unlimited, presently 0
        assert set(nc.variables.keys()) == {"time", "particle_count", "pid", "X", "X0"}
        assert nc.variables["pid"].dimensions == ("particle_instance",)
        assert nc.variables["X0"].dimensions == ("particle",)
        assert set(nc.ncattrs()) == {"institution", "source", "type", "history"}
        assert nc.getncattr("source") == "LADiM"

    NCFILE.unlink()


def test_reference_time():
    """Explicit reference time"""

    timer = TimeKeeper(
        start="2020-01-01 12", stop="2020-01-03 12", reference="2000-01-01", dt=1800,
    )
    state = State()
    config = config0.copy()
    config['modules'] = config['modules'].copy()
    config['modules']['time'] = timer
    out = Output(**config)
    state.append(X=100, Y=10, Z=5)
    out.write(state)
    out.close()
    with Dataset(NCFILE) as nc:
        tvar = nc.variables["time"]
        assert tvar.units == "seconds since 2000-01-01T00:00:00"
        assert (
            timer.reference_time + np.timedelta64(int(tvar[0]), "s") == timer.start_time
        )
    NCFILE.unlink()


def test_write():
    """Write a sequence of states"""
    state = State(particle_variables={"X0": float})
    out = Output(**config0)
    timer = out.timer
    outper = out.output_period // timer.dt

    assert out.record_count == 0
    assert out.instance_count == 0

    # Initially one particle
    state.append(X=100, Y=10, Z=5, X0=100)
    timer.reset()
    out.write(state)
    assert out.record_count == 1
    assert out.instance_count == 1

    # Update position
    state["X"] += 1
    for i in range(outper):
        timer.update()
    out.write(state)
    assert out.record_count == 2
    assert out.instance_count == 2

    # Update first particle and add two new particles
    state["X"] += 1
    state.append(
        X=np.array([200, 300]), Y=np.array([20, 30]), Z=5, X0=np.array([200, 300])
    )
    for i in range(outper):
        timer.update()
    out.write(state)
    assert out.record_count == 3
    assert out.instance_count == 5

    # Update particle positions and kill the first particle, pid=0
    state["X"] = state["X"] + 1.0
    state["alive"][0] = False
    state.compactify()
    for i in range(outper):
        timer.update()
    out.write(state)
    assert out.record_count == 4
    assert out.instance_count == 7

    # Update positions
    state["X"] += 1
    for i in range(outper):
        timer.update()
    out.write(state)
    assert out.record_count == 5
    assert out.instance_count == 9


    # Write particle variable
    # out.write_particle_variables(state)
    # out.close()

    # Check some of the content
    h = 3600
    with Dataset(NCFILE) as nc:
        assert all(nc.variables["time"][:] == [i * 12 * h for i in range(5)])
        assert all(nc.variables["particle_count"][:] == [1, 1, 3, 2, 2])
        # Instance variables
        assert all(nc.variables["pid"][:] == [0, 0, 0, 1, 2, 1, 2, 1, 2])
        assert all(
            nc.variables["X"][:] == [100, 101, 102, 200, 300, 201, 301, 202, 302]
        )
        # The particle variable has been written (by last write)
        assert all(nc.variables["X0"][:] == [100, 200, 300])

    NCFILE.unlink()


def test_multifile():
    """Test the multifile functionality"""
    # Missing: Test with particle variables (when implemented)

    h = 3600

    config = dict(config0, filename="a.nc", numrec=2, particle_variables=dict())
    out = Output(**config)
    state = State()
    timer = out.timer
    outper = out.output_period // timer.dt

    # First file
    state.append(X=100, Y=10, Z=5)
    timer.reset()
    out.write(state)
    state["X"] += 1
    for i in range(outper):
        timer.update()
    out.write(state)
    with Dataset("a_000.nc") as nc:
        assert all(nc.variables["time"][:] == [0, 12 * h])
        assert all(nc.variables["particle_count"][:] == [1, 1])
        assert all(nc.variables["pid"][:] == [0, 0])

    # Second file
    # Update first particle and add two new particles
    state["X"] += 1
    state.append(X=np.array([200, 300]), Y=np.array([20, 30]), Z=5)
    for i in range(outper):
        timer.update()
    out.write(state)
    state["X"] = state["X"] + 1.0
    state["alive"][0] = False
    state.compactify()
    for i in range(outper):
        timer.update()
    out.write(state)
    with Dataset("a_001.nc") as nc:
        assert all(nc.variables["time"][:] == [24 * h, 36 * h])
        assert all(nc.variables["particle_count"][:] == [3, 2])
        assert all(nc.variables["pid"][:] == [0, 1, 2, 1, 2])

    # Third and last file
    state["X"] += 1
    for i in range(outper):
        timer.update()
    out.write(state)
    with Dataset("a_002.nc") as nc:
        assert all(nc.variables["time"][:] == [48 * h])
        assert all(nc.variables["particle_count"][:] == [2])
        assert all(nc.variables["pid"][:] == [1, 2])

    # Clean up, remove the netcdf files
    for file_ in Path(".").glob("a_0??.nc"):
        file_.unlink()
