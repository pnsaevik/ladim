"""Module for testing config file API"""


import xarray as xr
import pandas as pd

import uuid
import shutil
import contextlib
from pathlib import Path
from ladim2.main import main
import yaml
import numpy as np


class Test_output_when_different_scenarios:
    def test_single_stationary_particle(self):
        gridforce_zero = make_gridforce()
        release_single = make_release(t=[0], X=[2], Y=[2], Z=[0])
        conf = make_conf()

        with runladim(conf, gridforce_zero, release_single) as result:
            assert set(result.dims) == {"particle", "particle_instance", "time"}
            assert set(result.coords) == {"time"}
            assert set(result.data_vars) == {
                "release_time",
                "X",
                "Z",
                "Y",
                "particle_count",
                "pid",
            }

            assert result.X.values.tolist() == [2, 2, 2]

    def test_multiple_release_times(self):
        gridforce_zero = make_gridforce()
        release_multiple = make_release(t=[0, 1, 2], X=2, Y=2, Z=0)
        conf = make_conf()

        with runladim(conf, gridforce_zero, release_multiple) as result:
            assert result.particle_count.values.tolist() == [1, 2, 3]
            assert result.pid.values.tolist() == [0, 0, 1, 0, 1, 2]
            assert (
                (result.release_time.values - result.release_time.values[0])
                / np.timedelta64(1, "s")
            ).tolist() == [0, 60, 120]

    def test_multiple_initial_particles(self):
        gridforce_zero = make_gridforce()
        release_multiple = make_release(t=[0] * 5, X=2, Y=2, Z=0)
        conf = make_conf()

        with runladim(conf, gridforce_zero, release_multiple) as result:
            assert result.particle_count.values.tolist() == [5] * 3
            assert result.pid.values.tolist() == [0, 1, 2, 3, 4] * 3
            assert (
                (result.release_time.values - result.release_time.values[0])
                / np.timedelta64(1, "s")
            ).tolist() == [0] * 5

    def test_single_particle_linear_forcing(self):
        gf = make_gridforce(ufunc=lambda *_: 1 / 60, vfunc=lambda *_: 2 / 60)
        rls = make_release(t=[0], X=2, Y=2, Z=0)
        conf = make_conf()

        with runladim(conf, gf, rls) as result:
            assert result.X.values.tolist() == [2, 3, 4]
            assert result.Y.values.tolist() == [2, 4, 6]

    def test_single_particle_nonzero_vertdiff(self):
        gf = make_gridforce()
        rls = make_release(t=[0] * 4, X=2, Y=2, Z=[0, 1, 2, 3])
        conf = make_conf()
        conf["tracker"]["vertdiff"] = 1e-7

        with runladim(conf, gf, rls) as result:
            assert result.pid.values.tolist() == [0, 1, 2, 3] * 3
            assert all(result.Z.values[4:] != [0, 1, 2, 3] * 2)
            assert all(result.Z.values[4:] > 0)
            assert all(result.Z.values[4:] < 3)
            assert all(np.abs(result.Z.values - [0, 1, 2, 3] * 3) < 1)

    def test_single_particle_nonzero_vertical_advection(self):
        gf = make_gridforce(wfunc=lambda *args: 0.125 / 60)
        rls = make_release(t=[0] * 4, X=2, Y=2, Z=[0, 1, 2, 3])
        conf = make_conf()
        conf["tracker"]["vertical_advection"] = "EF"
        conf["forcing"]["ibm_forcing"] = ["w"]

        with runladim(conf, gf, rls) as result:
            assert result.pid.values.tolist() == [0, 1, 2, 3] * 3
            assert result.Z.values.tolist() == [
                0,
                1,
                2,
                3,
                0.125,
                1.125,
                2.125,
                3,
                0.25,
                1.25,
                2.25,
                3,
            ]


@contextlib.contextmanager
def tempfile(num):
    """Creates an arbritary number of temporary files which are deleted upon exit"""
    d = Path(__file__).parent.joinpath("temp")
    d.mkdir(exist_ok=True)
    paths = [d.joinpath(uuid.uuid4().hex + ".tmp") for _ in range(num)]

    try:
        yield [str(p) for p in paths]

    finally:
        for p in paths:
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except IOError:
                pass

        try:
            d.rmdir()
        except IOError:
            pass


@contextlib.contextmanager
def runladim(conf, gridforce, release):
    """Run ladim and return result, creating temporary files as necessary.

    :param conf: Ladim configuration, with certain elements left blank (these are
                 auto-filled by the function): conf['grid']['filename'],
                 conf['forcing']['filename'], conf['release']['release_file'],
                 conf['output']['filename']
    :type conf: dict
    :param gridforce: Grid and forcing dataset
    :type gridforce: xarray.Dataset
    :param release: Particle release table
    :type release: pandas.DataFrame
    :return: Ladim output dataset
    :rtype: xarray.Dataset
    """
    with tempfile(4) as fnames:
        conf_fname, out_fname, gridforce_fname, rls_fname = fnames

        gridforce.to_netcdf(gridforce_fname)
        release.to_csv(rls_fname, sep="\t", index=False)

        conf["grid"]["filename"] = gridforce_fname
        conf["forcing"]["filename"] = gridforce_fname
        conf["release"]["release_file"] = rls_fname
        conf["output"]["filename"] = out_fname

        with open(conf_fname, "w", encoding="utf-8") as conf_file:
            yaml.safe_dump(conf, conf_file)

        main(conf_fname)

        with xr.open_dataset(out_fname, engine="netcdf4") as dset:
            yield dset


def make_gridforce(ufunc=None, vfunc=None, wfunc=None):
    """Create grid and forcing dataset to be used for testing"""

    def zerofunc(tt, ss, yy, xx):
        return tt * 0.0 + ss * 0.0 + yy * 0.0 + xx * 0.0

    ufunc = ufunc or zerofunc
    vfunc = vfunc or zerofunc
    wfunc = wfunc or zerofunc

    x = xr.Variable("xi_rho", np.arange(10))
    y = xr.Variable("eta_rho", np.arange(15))
    x_u = xr.Variable("xi_u", np.arange(len(x) - 1))
    y_u = xr.Variable("eta_u", np.arange(len(y)))
    x_v = xr.Variable("xi_v", np.arange(len(x)))
    y_v = xr.Variable("eta_v", np.arange(len(y) - 1))
    s = xr.Variable("s_rho", np.arange(2))
    s_w = xr.Variable("s_w", np.arange(len(s) + 1))
    ocean_t = xr.Variable(
        "ocean_time",
        np.datetime64("2000-01-02T03") + np.arange(3) * np.timedelta64(1, "m"),
    )
    t = xr.Variable("ocean_time", np.arange(len(ocean_t)))

    return xr.Dataset(
        data_vars=dict(
            ocean_time=ocean_t,
            h=(y * 0 + x * 0) + 3.0,
            mask_rho=(y * 0 + x * 0) + 1,
            pm=(y * 0 + x * 0) + 1.0,
            pn=(y * 0 + x * 0) + 1.0,
            angle=(y * 0 + x * 0) + 0.0,
            hc=0.0,
            Cs_r=-1.0 + (s + 0.5) / len(s),
            Cs_w=-1.0 + s_w / (len(s_w) - 1),
            u=zerofunc(t, s, y_u, x_u) + ufunc(t, s, y_u, x_u),
            v=zerofunc(t, s, y_v, x_v) + vfunc(t, s, y_v, x_v),
            w=zerofunc(t, s_w, y, x) + wfunc(t, s_w, y, x),
        ),
        coords=dict(lon_rho=(x * 1 + y * 0), lat_rho=(x * 0 + y * 1),),
    )


def make_release(t, **params_or_funcs):
    """Create particle release table to be used for testing"""

    t = np.array(t)
    i = np.arange(len(t))

    params = {
        k: (p(i, t) if callable(p) else p) + np.zeros_like(t)
        for k, p in params_or_funcs.items()
    }

    start_date = np.datetime64("2000-01-02T03")
    minute = np.timedelta64(60, "s")
    dates = start_date + np.array(t) * minute

    return pd.DataFrame(data={**dict(release_time=dates.astype(str)), **params,})


def make_conf() -> dict:
    """Create standard ladim configuration to be used for testing"""

    return dict(
        version=2,
        time=dict(dt=[30, "s"], start="2000-01-02T03", stop="2000-01-02T03:02",),
        grid=dict(module="ladim2.ROMS",),
        forcing=dict(module="ladim2.ROMS",),
        release=dict(),
        state=dict(
            particle_variables=dict(release_time="time", weight="float",),
            instance_variables=dict(age="float"),
            default_values=dict(weight=0, age=0),
        ),
        tracker=dict(advection="EF", diffusion=0.0, vertdiff=0.0,),
        output=dict(
            output_period=[60, "s"],
            particle_variables=dict(
                release_time=dict(
                    encoding=dict(datatype="f8"),
                    attributes=dict(
                        long_name="particle release time",
                        units="seconds since reference_time",
                    ),
                ),
            ),
            instance_variables=dict(
                pid=dict(
                    encoding=dict(datatype="i4"),
                    attributes=dict(long_name="particle identifier",),
                ),
                X=dict(
                    encoding=dict(datatype="f4"),
                    attributes=dict(long_name="particle X-coordinate",),
                ),
                Y=dict(
                    encoding=dict(datatype="f4"),
                    attributes=dict(long_name="particle Y-coordinate",),
                ),
                Z=dict(
                    encoding=dict(datatype="f4"),
                    attributes=dict(
                        long_name="particle depth",
                        standard_name="depth_below_surface",
                        units="m",
                        positive="down",
                    ),
                ),
            ),
        ),
    )
