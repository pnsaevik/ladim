import numpy as np
import pytest
from netCDF4 import Dataset

ZLEV = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0])
NY, NX = 9, 11


def _linear(z, y, x, t):
    """Analytic field, linear in all coordinates"""
    return 1.0 + 0.5 * z + 0.25 * y - 0.125 * x + 2.0 * t


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    path = tmp_path_factory.mktemp("zroms")
    grid_file = str(path / "grid.nc")
    forc_file = str(path / "forcing_001.nc")

    with Dataset(grid_file, "w") as nc:
        nc.createDimension("eta_rho", NY)
        nc.createDimension("xi_rho", NX)
        for name in ("h", "mask_rho", "pm", "pn", "lon_rho", "lat_rho"):
            nc.createVariable(name, "f8", ("eta_rho", "xi_rho"))
        nc["h"][:] = 50.0
        nc["mask_rho"][:] = 1.0
        nc["pm"][:] = 1 / 800.0
        nc["pn"][:] = 1 / 800.0
        nc["lon_rho"][:] = np.arange(NX) * 0.01 + np.zeros((NY, 1))
        nc["lat_rho"][:] = np.arange(NY)[:, None] * 0.01 + np.zeros(NX)

    nt = 3
    with Dataset(forc_file, "w") as nc:
        for name, n in [("time", nt), ("depth", len(ZLEV)), ("eta_rho", NY),
                        ("xi_rho", NX), ("eta_v", NY + 1), ("xi_u", NX + 1)]:
            nc.createDimension(name, n)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "hours since 2000-01-01"
        t[:] = np.arange(nt)
        for name, dims, dy, dx in [
            ("u", ("eta_rho", "xi_u"), 0.0, 0.5),
            ("v", ("eta_v", "xi_rho"), 0.5, 0.0),
            ("temp", ("eta_rho", "xi_rho"), 0.0, 0.0),
        ]:
            v = nc.createVariable(name, "f4", ("time", "depth") + dims,
                                  chunksizes=(1, 3, 4, 5))
            ny, nx = NY + (name == "v"), NX + (name == "u")
            Z = ZLEV[None, :, None, None]
            Y = np.arange(ny)[None, None, :, None] + dy
            X = np.arange(nx)[None, None, None, :] + dx
            T = np.arange(nt)[:, None, None, None]
            v[:] = np.broadcast_to(_linear(Z, Y, X, T), v.shape)
    return grid_file, str(path / "forcing_*.nc")


def _config(files, **kwargs):
    grid_file, input_file = files
    return dict(
        gridforce=dict(grid_file=grid_file, input_file=input_file,
                       num_threads=2, **kwargs),
        ibm_forcing=[],
        start_time="2000-01-01T00:00:00",
        stop_time="2000-01-01T02:00:00",
        dt=600,
    )


@pytest.fixture
def forcing(files):
    from ladim.gridforce.zROMS import Forcing

    f = Forcing(_config(files), None)
    yield f
    f.close()


def _particles(n=200, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(2.1, NX - 2.1, n)
    Y = rng.uniform(2.1, NY - 2.1, n)
    Z = rng.uniform(0.0, 20.0, n)
    return X, Y, Z


class TestZCoordinate:
    def test_level_interpolates_in_depth(self):
        from ladim.gridforce.zroms.coords import ZCoordinate

        zc = ZCoordinate(ZLEV, np.zeros((4, 5)))
        Z = np.array([0.0, 0.5, 1.0, 4.0, 5.0, 12.5, 20.0])
        k = zc.level(np.zeros(7), np.zeros(7), Z)
        assert np.allclose(k, [0.0, 0.5, 1.0, 3.5, 4.0, 5.5, 7.0])

    def test_constant_extension_outside_levels(self):
        from ladim.gridforce.zroms.coords import ZCoordinate

        zc = ZCoordinate(ZLEV, np.zeros((4, 5)))
        k = zc.level(np.zeros(2), np.zeros(2), np.array([-3.0, 100.0]))
        assert np.array_equal(k, [0.0, 7.0])

    def test_cell_is_level_above_particle(self):
        from ladim.gridforce.zroms.coords import ZCoordinate

        zc = ZCoordinate(ZLEV, np.zeros((4, 5)), i0=1, j0=1)
        X = np.array([1.2, 3.6, 9.0])
        Y = np.array([1.0, 2.4, 9.0])
        Z = np.array([0.5, 4.0, 99.0])
        K, J, I = zc.cell(X, Y, Z)
        assert np.array_equal(K, [0, 3, 7])
        assert np.array_equal(I, [1, 4, 5])
        assert np.array_equal(J, [1, 2, 4])

    def test_rejects_unsorted_levels(self):
        from ladim.gridforce.zroms.coords import ZCoordinate

        with pytest.raises(ValueError):
            ZCoordinate([0, 5, 2], np.zeros((2, 2)))


def test_grid_z_levels_default(files):
    from ladim.gridforce.zROMS import Grid

    grid = Grid(_config(files))
    assert np.array_equal(grid.z_levels, ZLEV)
    assert grid.angle is None


def test_grid_z_levels_from_config(files):
    from ladim.gridforce.zROMS import Grid

    grid = Grid(_config(files, z_levels=[0, 10, 20]))
    assert np.array_equal(grid.z_levels, [0, 10, 20])


def test_velocity_is_linear_in_all_dimensions(forcing):
    X, Y, Z = _particles()
    forcing.update(0)
    u, v = forcing.velocity(X, Y, Z)
    # u and v are stored at x = i + 1/2 and y = j + 1/2 respectively
    assert np.allclose(u, _linear(Z, Y, X, 0), atol=1e-4)
    assert np.allclose(v, _linear(Z, Y, X, 0), atol=1e-4)


def test_velocity_interpolates_in_time(forcing):
    X, Y, Z = _particles()
    forcing.update(3)  # Half way between frame 0 and frame 1 (dt = 600 s)
    u, _ = forcing.velocity(X, Y, Z)
    assert np.allclose(u, _linear(Z, Y, X, 0.5), atol=1e-4)
    u, _ = forcing.velocity(X, Y, Z, tstep=3)
    assert np.allclose(u, _linear(Z, Y, X, 1.0), atol=1e-4)


def test_field_nearest_uses_level_above(forcing):
    X, Y, Z = _particles()
    forcing.update(0)
    temp = forcing.field(X, Y, Z, "temp")
    K = np.searchsorted(ZLEV, Z, side="right") - 1
    I, J = np.rint(X).astype(int), np.rint(Y).astype(int)
    assert np.allclose(temp, _linear(ZLEV[K], J, I, 0), atol=1e-4)


def test_field_linear_in_depth(forcing):
    X, Y, Z = _particles()
    forcing.update(0)
    temp = forcing.field(X, Y, Z, "temp", linear="z")
    J, I = np.rint(Y).astype(int), np.rint(X).astype(int)
    assert np.allclose(temp, _linear(Z, J, I, 0), atol=1e-4)


def test_depths_outside_levels_use_nearest_level(forcing):
    X, Y, _ = _particles(10)
    forcing.update(0)
    above = forcing.velocity(X, Y, np.full(10, -5.0))[0]
    top = forcing.velocity(X, Y, np.zeros(10))[0]
    below = forcing.velocity(X, Y, np.full(10, 500.0))[0]
    bottom = forcing.velocity(X, Y, np.full(10, 20.0))[0]
    assert np.array_equal(above, top)
    assert np.array_equal(below, bottom)


def test_too_many_z_levels_is_an_error(files):
    from ladim.gridforce.zROMS import Forcing

    with pytest.raises(SystemExit):
        Forcing(_config(files, z_levels=list(range(20))), None)
