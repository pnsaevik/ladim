import pathlib
import warnings

import numpy as np
import pytest

from ladim.gridforce.roms import coords

SAMPLE = pathlib.Path(__file__).parent / "sample_data" / "forcing.nc"


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
    # The level just above, or the lowest level if below all levels
    col = z_r[:, J - j0, I - i0]
    count = np.sum(col < -Z, axis=0)
    assert np.array_equal(Kc, np.minimum(count, len(Cs_r) - 1))
    assert np.any(Kc == 0)
    # Same as the legacy rule (K from z2s, lowest level if A == 1) where the
    # levels are monotonic (not for Vtransform 1 with bottom depth < hc)
    mono = np.all(np.diff(col, axis=0) > 0, axis=0)
    assert np.array_equal(Kc[mono], (K0 - (A0 == 1))[mono])
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
    assert forcing._steps.tolist() == [0, 6, 12, 18]


def test_field_uses_latest_forcing_frame(forcing):
    from netCDF4 import Dataset as NCDataset

    grid = forcing._grid
    rng = np.random.default_rng(8)
    X = rng.uniform(grid.xmin + 0.5, grid.xmax - 0.5, 200)
    Y = rng.uniform(grid.ymin + 0.5, grid.ymax - 0.5, 200)
    Z = rng.uniform(0, 60, 200)
    Z[:20] = grid.sample_depth(X[:20], Y[:20])  # At the bottom
    Z[20:40] = 1e4  # Far below the bottom
    K, A = coords.z2s(grid.z_r, X - grid.i0, Y - grid.j0, Z)
    with NCDataset(str(SAMPLE)) as nc:
        nc.set_auto_maskandscale(False)
        v = nc.variables["temp"]
        frames = v.add_offset + v.scale_factor * v[:, :, grid.J, grid.I]

    # Forcing frames at steps 0, 6, 12, 18, also after jumps in time
    for t in [-1, 0, 3, 5, 6, 8, 12, 17, 18, 19, 2]:
        forcing.update(t)
        n = min(max(t // 6, 0), 3)
        ref = coords.sample3D(frames[n], X - grid.i0, Y - grid.j0, K, A, method="nearest")
        field = forcing.field(X, Y, Z, "temp")
        assert np.allclose(field, ref, atol=1e-5), t
        # Lowest level at and below the bottom
        J, I = np.round(Y[:40] - grid.j0).astype(int), np.round(X[:40] - grid.i0).astype(int)
        assert np.allclose(field[:40], frames[n][0, J, I], atol=1e-5), t


@pytest.mark.parametrize("vtransform", [1, 2])
def test_w_levels_match_dense(vtransform):
    rng = np.random.default_rng(9)
    H = rng.uniform(25, 400, size=(12, 17))  # Monotonic levels (H > hc)
    Cs_w = coords.s_stretch(30, 6.0, 0.5, stagger="w")
    hc = 20.0
    i0, j0 = 3, 5
    n = 2000
    X = rng.uniform(i0 - 2, i0 + 18, n)
    Y = rng.uniform(j0 - 2, j0 + 13, n)
    Z = rng.uniform(-1, 420, n)

    z_w = coords.sdepth(H, hc, Cs_w, stagger="w", Vtransform=vtransform)
    sc = coords.SCoordinate(H, Cs_w, hc, vtransform, i0, j0, stagger="w")
    K0, A0 = coords.z2s(z_w, X - i0, Y - j0, Z)
    K, A = sc.z2s(X, Y, Z)
    assert np.array_equal(K, K0)
    assert np.allclose(A, A0, rtol=0, atol=1e-12)

    Kc, J, I = sc.cell(X, Y, Z)
    count = np.sum(z_w[:, J - j0, I - i0] < -Z, axis=0)
    assert np.array_equal(Kc, np.minimum(count, len(Cs_w) - 1))


def _particles(grid, n=300, seed=10):
    rng = np.random.default_rng(seed)
    X = rng.uniform(grid.xmin + 0.5, grid.xmax - 0.5, n)
    Y = rng.uniform(grid.ymin + 0.5, grid.ymax - 0.5, n)
    Z = rng.uniform(0, 60, n)
    Z[:20] = grid.sample_depth(X[:20], Y[:20])  # At the bottom
    Z[20:40] = 1e4  # Far below the bottom
    return X, Y, Z


def _frames(name, J, I):
    """Decoded values of a forcing variable at grid points (J, I)"""
    from netCDF4 import Dataset as NCDataset

    with NCDataset(str(SAMPLE)) as nc:
        nc.set_auto_maskandscale(False)
        v = nc.variables[name]
        raw = v[:][..., J, I]
        return getattr(v, "add_offset", 0) + getattr(v, "scale_factor", 1) * raw


def test_field_on_w_levels(forcing):
    grid = forcing._grid
    X, Y, Z = _particles(grid)
    J = np.round(Y - grid.j0).astype(int)
    I = np.round(X - grid.i0).astype(int)
    z_w = grid.z_w[:, J, I]
    W = _frames("w", J + grid.j0, I + grid.i0)  # (time, s_w, particle)
    K, A = coords.z2s(grid.z_w, X - grid.i0, Y - grid.j0, Z)
    level = np.minimum(np.sum(z_w < -Z, axis=0), grid.N)
    p = np.arange(len(X))

    for t in range(14):
        forcing.update(t)
        # Default for w: linear in time and depth, nearest horizontally
        n0 = min(t // 6, 2)
        wt = (t - 6 * n0) / 6
        F = (1 - wt) * W[n0] + wt * W[n0 + 1]
        ref = A * F[K - 1, p] + (1 - A) * F[K, p]
        assert np.allclose(forcing.field(X, Y, Z, "w"), ref, atol=1e-6), t

        # No interpolation: latest frame, level just above (or the lowest)
        ref = W[min(t // 6, 3)][level, p]
        assert np.allclose(forcing.field(X, Y, Z, "w", linear=""), ref, atol=1e-6), t


def test_field_at_u_points(forcing):
    grid = forcing._grid
    X, Y, Z = _particles(grid)
    forcing.update(0)
    K, _, _ = grid.scoord.cell(X, Y, Z)
    J = np.rint(Y).astype(int)
    I = np.rint(X - 0.5).astype(int)  # u point i is at x = i + 1/2
    U = _frames("u", J, I)
    ref = U[0][K, np.arange(len(X))]
    assert np.allclose(forcing.field(X, Y, Z, "u"), ref, atol=1e-6)


def test_interpolation_setting():
    from ladim.gridforce.ROMS import Forcing

    config = dict(
        gridforce=dict(input_file=str(SAMPLE), num_threads=2,
                       interpolation=dict(temp="tz", w="")),
        ibm_forcing=[],
        start_time="2015-09-07T01:00:00",
        stop_time="2015-09-07T04:00:00",
        dt=600,
    )
    f = Forcing(config, None)
    X, Y, Z = _particles(f._grid)
    f.update(3)
    assert np.array_equal(f.field(X, Y, Z, "temp"), f.field(X, Y, Z, "temp", linear="tz"))
    assert np.array_equal(f.field(X, Y, Z, "w"), f.field(X, Y, Z, "w", linear=""))
    assert not np.allclose(f.field(X, Y, Z, "w"), f.field(X, Y, Z, "w", linear="tz"))
    with pytest.raises(ValueError):
        f.field(X, Y, Z, "w", linear="q")
    f.close()


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


# --------------------------------------------------------------------------
# s-coordinate (s_rho, s_w)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("vstretching", [1, 2, 3, 4])
def test_s_coordinate_is_even_for_vstretching_1_to_4(vstretching):
    N = 12
    assert np.allclose(coords.s_coordinate(N, "rho", vstretching),
                       -1 + (0.5 + np.arange(N)) / N)
    assert np.allclose(coords.s_coordinate(N, "w", vstretching),
                       np.linspace(-1, 0, N + 1))


def test_s_coordinate_vstretching_5_matches_norkyst():
    # s_rho in the Norkyst files (40 levels, Vstretching 5)
    s_rho = coords.s_coordinate(40, "rho", 5)
    assert np.allclose(s_rho[[0, 1, 2, -2, -1]],
                       [-0.979904, -0.940288, -0.901442, -0.015288, -0.004904],
                       rtol=0, atol=1e-6)
    s_w = coords.s_coordinate(40, "w", 5)
    assert s_w[0] == -1 and s_w[-1] == 0
    assert np.all(np.diff(s_w) > 0)
    # The levels are in the middle of the layers
    assert not np.allclose(s_rho, -1 + (0.5 + np.arange(40)) / 40, atol=0.01)


def test_s_stretch_unchanged_by_s_coordinate():
    # Vstretching 5 curve of Norkyst: bottom -1, surface 0
    Cs_w = coords.s_stretch(40, 5.0, 4.0, stagger="w", Vstretching=5)
    assert Cs_w[0] == pytest.approx(-1.0) and Cs_w[-1] == pytest.approx(0.0)


def test_sdepth_uses_given_s_coordinate():
    N, hc, H = 40, 100.0, np.array([200.0])
    Cs_w = coords.s_stretch(N, 5.0, 4.0, stagger="w", Vstretching=5)
    S_w = coords.s_coordinate(N, "w", 5)
    z_w = coords.sdepth(H, hc, Cs_w, stagger="w", Vtransform=2, S=S_w)[:, 0]
    # Surface and bottom
    assert z_w[0] == pytest.approx(-200.0) and z_w[-1] == pytest.approx(0.0)
    # Thickness of the uppermost layer (0.676 m), not 1.676 m as with evenly
    # spaced s-coordinates
    assert z_w[-1] - z_w[-2] == pytest.approx(0.676, abs=0.005)
    z_even = coords.sdepth(H, hc, Cs_w, stagger="w", Vtransform=2)[:, 0]
    assert z_even[-1] - z_even[-2] == pytest.approx(1.676, abs=0.005)
    # Vtransform 2: z = H * (hc * S + H * C) / (hc + H)
    assert np.allclose(z_w, 200 * (hc * S_w + 200 * Cs_w) / (hc + 200))


@pytest.mark.parametrize("vtransform", [1, 2])
@pytest.mark.parametrize("stagger", ["rho", "w"])
def test_scoordinate_with_s_matches_dense(vtransform, stagger):
    rng = np.random.default_rng(1)
    H = rng.uniform(120, 400, size=(8, 9))
    N, hc, i0, j0 = 30, 100.0, 2, 3
    C = coords.s_stretch(N, 5.0, 4.0, stagger=stagger, Vstretching=5)
    S = coords.s_coordinate(N, stagger, 5)
    n = 1000
    X = rng.uniform(i0, i0 + 8, n)
    Y = rng.uniform(j0, j0 + 7, n)
    Z = rng.uniform(0, 380, n)

    z = coords.sdepth(H, hc, C, stagger=stagger, Vtransform=vtransform, S=S)
    K0, A0 = coords.z2s(z, X - i0, Y - j0, Z)
    sc = coords.SCoordinate(H, C, hc, vtransform, i0, j0, stagger=stagger, S=S)
    K, A = sc.z2s(X, Y, Z)
    assert np.array_equal(K, K0)
    assert np.allclose(A, A0, rtol=0, atol=1e-12)
    assert np.allclose(sc.level(X, Y, Z), K0 - A0, rtol=0, atol=1e-12)

    # Not the same as with evenly spaced s-coordinates
    sc_even = coords.SCoordinate(H, C, hc, vtransform, i0, j0, stagger=stagger)
    assert not np.allclose(sc_even.level(X, Y, Z), sc.level(X, Y, Z), atol=0.01)


def test_sdepth_rejects_wrong_length_of_s():
    with pytest.raises(ValueError, match="same length"):
        coords.sdepth(np.array([100.0]), 10.0, np.linspace(-1, 0, 5), S=np.zeros(4))


def _write_vertical_grid_file(path, *, N=20, vstretching=5, vtransform=2):
    from netCDF4 import Dataset as NC

    jmax, imax = 6, 7
    Cs_r = coords.s_stretch(N, 5.0, 4.0, stagger="rho", Vstretching=5)
    Cs_w = coords.s_stretch(N, 5.0, 4.0, stagger="w", Vstretching=5)
    with NC(path, "w") as nc:
        nc.createDimension("eta_rho", jmax)
        nc.createDimension("xi_rho", imax)
        nc.createDimension("s_rho", N)
        nc.createDimension("s_w", N + 1)
        dims = ("eta_rho", "xi_rho")
        for name, value in [("h", 300.0), ("mask_rho", 1.0), ("pm", 1 / 800.0),
                            ("pn", 1 / 800.0), ("lon_rho", 5.0),
                            ("lat_rho", 60.0), ("angle", 0.0)]:
            nc.createVariable(name, "f8", dims)[:] = value
        nc.createVariable("hc", "f8", ())[:] = 100.0
        nc.createVariable("Vtransform", "i4", ())[:] = vtransform
        nc.createVariable("Cs_r", "f8", ("s_rho",))[:] = Cs_r
        nc.createVariable("Cs_w", "f8", ("s_w",))[:] = Cs_w
        if vstretching is not None:
            nc.createVariable("Vstretching", "i4", ())[:] = vstretching


def _grid(path):
    from ladim.gridforce.ROMS import Grid

    return Grid(dict(gridforce=dict(grid_file=str(path), num_threads=1)))


def test_grid_computes_s_coordinate_from_vstretching(tmp_path):
    N = 20
    f = tmp_path / "grid.nc"
    _write_vertical_grid_file(f, N=N, vstretching=5)
    grid = _grid(f)
    assert np.allclose(grid.S_r, coords.s_coordinate(N, "rho", 5))
    assert np.allclose(grid.S_w, coords.s_coordinate(N, "w", 5))
    assert np.allclose(grid.scoord.S, grid.S_r)
    assert np.allclose(grid.scoord_w.S, grid.S_w)
    # The depths use the s-coordinate
    z_w = grid.z_w[:, 0, 0]
    S, C = grid.S_w, grid.Cs_w
    assert np.allclose(z_w, 300 * (100 * S + 300 * C) / (100 + 300))


def test_grid_with_vstretching_1_has_even_s_coordinate(tmp_path):
    N = 20
    f = tmp_path / "grid.nc"
    _write_vertical_grid_file(f, N=N, vstretching=1)
    grid = _grid(f)
    assert np.allclose(grid.S_r, -1 + (0.5 + np.arange(N)) / N)
    assert np.allclose(grid.S_w, np.linspace(-1, 0, N + 1))


def test_grid_requires_vstretching(tmp_path):
    f = tmp_path / "grid.nc"
    _write_vertical_grid_file(f, vstretching=None)
    with pytest.raises(ValueError, match="Vstretching is missing"):
        _grid(f)


def test_grid_with_vinfo_uses_vstretching(tmp_path):
    f = tmp_path / "grid.nc"
    _write_vertical_grid_file(f, N=20)
    from ladim.gridforce.ROMS import Grid

    vinfo = dict(N=20, hc=100.0, theta_s=5.0, theta_b=4.0, Vstretching=5, Vtransform=2)
    grid = Grid(dict(gridforce=dict(grid_file=str(f), num_threads=1, Vinfo=vinfo)))
    assert np.allclose(grid.S_r, coords.s_coordinate(20, "rho", 5))
    assert np.allclose(grid.S_w, coords.s_coordinate(20, "w", 5))


def test_vinfo_overrides_the_vertical_grid_of_the_file(tmp_path):
    f = tmp_path / "grid.nc"
    _write_vertical_grid_file(f, N=20, vstretching=5, vtransform=2)
    from ladim.gridforce.ROMS import Grid

    vinfo = dict(N=10, hc=50.0, theta_s=3.0, theta_b=0.4, Vstretching=1, Vtransform=1)
    grid = Grid(dict(gridforce=dict(grid_file=str(f), num_threads=1, Vinfo=vinfo)))
    assert grid.N == 10 and grid.hc == 50.0
    assert grid.Vtransform == 1 and grid.Vstretching == 1
    assert np.allclose(grid.Cs_r, coords.s_stretch(10, 3.0, 0.4, "rho", 1))
    assert np.allclose(grid.S_r, -1 + (0.5 + np.arange(10)) / 10)
    assert grid.z_r.shape[0] == 10
