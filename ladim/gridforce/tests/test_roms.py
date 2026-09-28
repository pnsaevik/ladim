import pathlib
import warnings

import numpy as np
import pytest

from ladim.gridforce.roms import coords

SAMPLE = pathlib.Path(__file__).parents[3] / "tests" / "sample_data" / "forcing.nc"


@pytest.mark.parametrize("vtransform", [1, 2])
def test_z2s_matches_dense(vtransform):
    rng = np.random.default_rng(0)
    H = rng.uniform(5, 400, size=(12, 17))
    Cs_r = coords.s_stretch(30, 6.0, 0.5, stagger="rho")
    hc = 20.0
    i0, j0 = 3, 5
    n = 2000
    X = rng.uniform(i0 - 2, i0 + 18, n)
    Y = rng.uniform(j0 - 2, j0 + 13, n)
    X[:50] = np.round(X[:50]) + 0.5  # Rounding ties
    Z = rng.uniform(-1, 420, n)

    z_r = coords.sdepth(H, hc, Cs_r, stagger="rho", Vtransform=vtransform)
    K0, A0 = coords.z2s(z_r, X - i0, Y - j0, Z)
    sc = coords.SCoordinate(H, Cs_r, hc, vtransform, i0, j0)
    K, A = sc.z2s(X, Y, Z)
    assert np.array_equal(K, K0)
    assert np.allclose(A, A0, rtol=0, atol=1e-12)
    assert np.allclose(sc.level(X, Y, Z), K0 - A0, rtol=0, atol=1e-12)

    Kc, J, I = sc.cell(X, Y, Z)
    assert np.array_equal(Kc, K0)
    assert np.array_equal(I - i0, np.clip(np.around(X - i0), 0, 16).astype(int))
    assert np.array_equal(J - j0, np.clip(np.around(Y - j0), 0, 11).astype(int))


def test_lookup_rounds_global_coordinates():
    F = np.arange(20.0).reshape(4, 5)
    X = np.array([0.0, 2.5, 3.5, 4.4, 9.0, -3.0, np.nan])
    Y = np.array([1.5, 2.5, 2.0, 2.6, 1.0, 0.0, 1.0])
    i0, j0 = 1, 1
    I = np.clip(np.nan_to_num(X, nan=-1e9).round().astype(int) - i0, 0, 4)
    J = np.clip(Y.round().astype(int) - j0, 0, 3)
    assert np.array_equal(coords.lookup(F, X, Y, i0, j0), F[J, I])


def test_bilinear():
    F = np.arange(20.0).reshape(4, 5) ** 1.5
    x = np.array([0.0, 1.25, 3.99])
    y = np.array([0.0, 2.5, 2.2])
    assert np.allclose(coords.bilinear(F, x, y), legacy_sample2D(F, x, y))
    with pytest.raises(ValueError):
        coords.bilinear(F, np.array([4.0]), np.array([1.0]))


def legacy_sample2D(F, X, Y):
    from ladim.sample import sample2D
    return sample2D(F, X, Y)


@pytest.fixture(scope="module")
def forcing():
    if not SAMPLE.exists():
        pytest.skip("Sample data not available")
    from ladim.gridforce.ROMS import Forcing

    config = dict(
        gridforce=dict(input_file=str(SAMPLE), num_threads=2),
        ibm_forcing=["temp", "salt"],
        start_time="2015-09-07T01:00:00",
        stop_time="2015-09-07T04:00:00",
        dt=600,
    )
    f = Forcing(config, None)
    yield f
    f.close()


def test_forcing_steps(forcing):
    assert forcing.steps == [0, 6, 12, 18]
    assert list(forcing.stepdiff) == [6, 6, 6]


@pytest.mark.parametrize("first_step", [0, 3, 8])
def test_scalar_field_frames(forcing, first_step):
    from ladim.gridforce.roms.forcing import _ScalarTiming

    timing = _ScalarTiming(forcing._steps)
    timing.update(-1)  # Initialization
    frames = []
    for t in range(first_step, 20):
        timing.update(t)
        n0, n1, w1 = timing.frames
        frames.append(n0 if w1 == 0 else None)
    if first_step == 0:
        # Start at a forcing time: next frame in the first interval (as in
        # earlier LADiM versions)
        expected = [1] * 12 + [2] * 6 + [3] * 2
    else:
        # Latest frame at or before the step
        expected = [t // 6 for t in range(first_step, 20)]
    assert frames == expected


@pytest.fixture(scope="module")
def dense_fields(tmp_path_factory):
    """Random fields F (rho points), U and V (u and v points) in a netCDF file"""
    from netCDF4 import Dataset as NCDataset

    from ladim.gridforce.chunkloader import Dataset

    rng = np.random.default_rng(5)
    nt, nz, ny, nx = 2, 8, 11, 13
    shapes = dict(F=(ny, nx), U=(ny, nx + 1), V=(ny + 1, nx))
    fields = {k: rng.normal(size=(nt, nz) + s) for k, s in shapes.items()}
    path = str(tmp_path_factory.mktemp("dense") / "fields.nc")
    with warnings.catch_warnings(), NCDataset(path, "w") as nc:
        warnings.simplefilter("ignore")
        for name, n in [("time", nt), ("z", nz), ("y", ny), ("x", nx),
                        ("yv", ny + 1), ("xu", nx + 1)]:
            nc.createDimension(name, n)
        v = nc.createVariable("time", "f8", ("time",))
        v.units = "hours since 2000-01-01"
        v[:] = np.arange(nt)
        dims = dict(F=("y", "x"), U=("y", "xu"), V=("yv", "x"))
        for name, values in fields.items():
            nc.createVariable(name, "f8", ("time", "z") + dims[name],
                              chunksizes=(1, 3, 4, 5))[:] = values
    dset = Dataset(path, time_name="time", num_threads=2)
    yield dset, fields
    dset.close()


def test_sample3D_matches_interp(dense_fields):
    dset, fields = dense_fields
    F = fields["F"][1]
    nz, ny, nx = F.shape
    rng = np.random.default_rng(6)
    n = 400
    X = rng.uniform(0, nx - 1.001, n)
    Y = rng.uniform(0, ny - 1.001, n)
    K = rng.integers(1, nz, n)
    A = rng.uniform(0, 1, n)

    ref = coords.sample3D(F, X, Y, K, A, method="bilinear")
    new = dset["F"].interp([K - A, Y, X], frame=1, linear="zyx")
    assert np.allclose(new, ref, rtol=0, atol=1e-12)

    ref = coords.sample3D(F, X, Y, K, A, method="nearest")
    J, I = np.round(Y).astype(int), np.round(X).astype(int)
    new = dset["F"].interp([K, J, I], frame=1)
    assert np.array_equal(new, ref)


def test_sample3DUV_matches_interp(dense_fields):
    dset, fields = dense_fields
    U, V = fields["U"][0], fields["V"][0]
    nz, ny, nx = fields["F"][0].shape
    rng = np.random.default_rng(7)
    n = 400
    # Inside the rho grid (outside, sample3D extrapolates linearly while
    # interp extends the values at the boundary)
    X = rng.uniform(0, nx - 1.001, n)
    Y = rng.uniform(0, ny - 1.001, n)
    K = rng.integers(1, nz, n)
    A = rng.uniform(0, 1, n)

    u0, v0 = coords.sample3DUV(U, V, X, Y, K, A)
    # u point i is at x = i - 0.5, v point j at y = j - 0.5 (local indices)
    u = dset["U"].interp([K - A, Y, X + 0.5], frame=0)
    v = dset["V"].interp([K - A, Y + 0.5, X], frame=0)
    assert np.allclose(u, u0, rtol=0, atol=1e-12)
    assert np.allclose(v, v0, rtol=0, atol=1e-12)
