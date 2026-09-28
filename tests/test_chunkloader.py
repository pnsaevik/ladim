import warnings

import numpy as np
import pytest
from netCDF4 import Dataset as NCDataset
from scipy.interpolate import RegularGridInterpolator

from ladim.gridforce.chunkloader import Dataset

SHAPE = (5, 6, 7, 9)  # time, z, y, x (per file)

# name: createVariable keyword arguments
STORAGE = {
    "packed": dict(datatype="i2", zlib=True, shuffle=True, chunksizes=(1, 2, 3, 4)),
    "float": dict(datatype="f4", chunksizes=(2, 3, 5, 7)),
    "contiguous": dict(datatype="f8", contiguous=True),
    "bigendian": dict(datatype="f4", zlib=True, shuffle=True, endian="big",
                      chunksizes=(1, 6, 7, 9)),
    "multitime": dict(datatype="i2", zlib=True, shuffle=True, chunksizes=(5, 4, 4, 4)),
}


def truth(fnum):
    """Decoded values of the test variables in file fnum"""
    t, z, y, x = np.meshgrid(*[np.arange(n, dtype=float) for n in SHAPE], indexing="ij")
    t = t + fnum * SHAPE[0]
    return 1.5 * t + 0.3 * z * z - 0.7 * y + 0.11 * x * y + np.sin(x)


def write_file(path, fnum, fmt="NETCDF4"):
    with NCDataset(path, "w", format=fmt) as nc:
        for name, n in zip(("time", "z", "y", "x"), SHAPE):
            nc.createDimension(name, n)
        v = nc.createVariable("time", "f8", ("time",))
        v.units = "hours since 2000-01-01"
        v[:] = np.arange(SHAPE[0]) + fnum * SHAPE[0]
        values = truth(fnum)
        storages = STORAGE if fmt == "NETCDF4" else {"float": dict(datatype="f4")}
        for name, kw in storages.items():
            kw = dict(kw)
            dtype = kw.pop("datatype")
            v = nc.createVariable(name, dtype, ("time", "z", "y", "x"), **kw)
            if dtype == "i2":
                v.scale_factor = 0.01
                v.add_offset = 20.0
            v[:] = values
        s = nc.createVariable("static", "f8", ("y", "x"))
        s[:] = values[0, 0]


@pytest.fixture(scope="module", params=["NETCDF4", "NETCDF3_CLASSIC"])
def dset(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(request.param)
    files = [str(tmp / f"file_{i}.nc") for i in range(2)]
    with warnings.catch_warnings():
        # Deprecation and endianness warnings from netCDF4 when writing
        warnings.simplefilter("ignore")
        for i, f in enumerate(files):
            write_file(f, i, request.param)
    with Dataset(files, time_name="time", num_threads=4, max_frames=3) as ds:
        yield ds


def names(ds):
    return [n for n in STORAGE if n in ds]


def full_truth():
    return np.concatenate([truth(0), truth(1)])


def tolerance(name):
    return 0.006 if name in ("packed", "multitime") else 1e-5


def test_times(dset):
    assert len(dset.times) == 2 * SHAPE[0]
    assert dset.times[0] == np.datetime64("2000-01-01T00")
    assert dset.times[-1] == np.datetime64("2000-01-01T09")


def test_read_matches_truth(dset):
    T = full_truth()
    for name in names(dset):
        var = dset[name]
        for n in (0, 4, 5, 9, 3):
            assert np.abs(var.read(n) - T[n]).max() < tolerance(name), name
        region = (slice(1, 4), slice(2, 7), slice(0, 3))
        assert np.abs(var.read(7, region) - T[7][region]).max() < tolerance(name)


def test_read_static(dset):
    assert np.array_equal(dset["static"].read(), truth(0)[0, 0])


def random_coords(rng, n=500, margin=0.3):
    return [rng.uniform(-margin, s - 1 + margin, n) for s in SHAPE[1:]]


def test_interp_linear(dset):
    T = full_truth()
    rng = np.random.default_rng(1)
    k, j, i = random_coords(rng)
    grid = [np.arange(n, dtype=float) for n in T.shape]
    ref = RegularGridInterpolator(grid, T)
    # Constant extension outside the grid
    clipped = [np.clip(c, 0, s - 1) for c, s in zip((k, j, i), SHAPE[1:])]
    for frame in (3, 6.25, (2, 8, 0.4)):
        if isinstance(frame, tuple):
            n0, n1, w = frame
            expected = ((1 - w) * ref(np.stack([np.full_like(k, n0)] + clipped, 1))
                        + w * ref(np.stack([np.full_like(k, n1)] + clipped, 1)))
        else:
            expected = ref(np.stack([np.full_like(k, frame)] + clipped, 1))
        for name in names(dset):
            out = dset[name].interp([k, j, i], frame)
            assert np.abs(out - expected).max() < tolerance(name), (name, frame)


def test_interp_nearest_and_integer(dset):
    T = full_truth()
    rng = np.random.default_rng(2)
    k, j, i = random_coords(rng)
    K, J, I = (np.clip(np.rint(c), 0, s - 1).astype(int)
               for c, s in zip((k, j, i), SHAPE[1:]))
    for name in names(dset):
        var = dset[name]
        expected = T[6, K, J, I]
        tol = tolerance(name)
        assert np.abs(var.interp([k, j, i], 6, linear="") - expected).max() < tol
        # Integer coordinates are never interpolated
        assert np.abs(var.interp([K, J, I], 6, linear="tzyx") - expected).max() < tol
        # Fractional frame: nearest unless "t" is given
        assert np.abs(var.interp([K, J, I], 5.8, linear="zyx") - expected).max() < tol


def test_interp_partial_linear(dset):
    T = full_truth()
    rng = np.random.default_rng(3)
    k, j, i = random_coords(rng, margin=0)
    K = np.rint(k).astype(int)
    J = np.rint(j).astype(int)
    i0 = np.minimum(np.floor(i).astype(int), SHAPE[3] - 2)
    w = i - i0
    expected = (1 - w) * T[2, K, J, i0] + w * T[2, K, J, i0 + 1]
    out = dset["float"].interp([k, j, i], 2, linear="x")
    assert np.abs(out - expected).max() < 1e-5


def test_interp_mask(dset):
    T = full_truth()
    mask = np.ones(SHAPE[2:], dtype=np.uint8)
    mask[3, 4] = 0
    out = dset["float"].interp([[2.0, 2.0], [3.0, 3.0], [4.0, 4.5]], 1, mask=mask)
    assert out[0] == 0
    assert abs(out[1] - 0.5 * T[1, 2, 3, 5]) < 1e-5


def test_prefetch_and_loaded(dset):
    var = dset["float"]
    need = np.zeros(var.grid3, dtype=np.uint8)
    c = tuple(n - 1 for n in var.grid3)  # Last chunk
    need[c] = 1
    var.prefetch(8, need)
    fr = var.ensure(8, need)
    assert fr.loaded[c] == 1
    assert var.loaded(8).sum() >= 1


def test_frame_eviction(dset):
    T = full_truth()
    name = names(dset)[0]
    var = dset[name]
    for n in range(10):  # More frames than max_frames
        assert np.abs(var.read(n) - T[n]).max() < tolerance(name)
    assert sum(1 for key in dset._frames if key[0] == name) <= dset.max_frames
