"""
Chunked, lazily loaded access to gridded variables in a series of netCDF files.

A :class:`Dataset` joins the time frames of a sorted list of files. A
:class:`Variable` gives access to one variable, frame by frame. Each frame is
held in memory as a full-size array of raw (packed) values, but only the
chunks that have been requested are filled in. Chunks are decoded by a pool
of threads, either on demand (:meth:`Variable.interp`, :meth:`Variable.read`)
or ahead of time (:meth:`Variable.prefetch`).

Example::

    with Dataset(sorted(glob.glob("roms_his_*.nc"))) as dset:
        temp = dset["temp"]
        # Temperature at time frame 12.5, s-level 39.0, eta 400.2, xi 1200.7
        values = temp.interp([[39.0], [400.2], [1200.7]], frame=12.5)

Spatial arrays are internally padded to 3-D (leading axes of length 1), and
the time axis, if any, must be the first dimension of the variable.
"""

import collections
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import parallel
from . import codec, kernels

logger = logging.getLogger(__name__)


class Dataset:
    """Time frames of a sorted series of netCDF files

    :param files: File names (a single name or a list), sorted in time
    :param time_name: Name of the time variable, or None if only
        time-independent variables are accessed
    :param num_threads: Number of decoding threads (see
        :mod:`ladim.gridforce.parallel`)
    :param max_frames: Maximum number of time frames kept in memory for each
        variable. The least recently used frame is released first.
    :param times: Optional precomputed times of all frames (skips scanning)
    """

    def __init__(self, files, time_name="ocean_time", num_threads=None,
                 max_frames=4, times=None):
        if isinstance(files, (str, os.PathLike)):
            files = [files]
        self.files = [str(f) for f in files]
        if not self.files:
            raise ValueError("No files given")
        self.time_name = time_name
        self.max_frames = max(2, int(max_frames))
        self.num_threads = parallel.num_threads(num_threads)

        if time_name is None:
            frames_per_file = [0] * len(self.files)
            self.times = np.zeros(0, dtype="datetime64[us]")
        else:
            if times is None:
                times = [read_times(f, time_name) for f in self.files]
            frames_per_file = [len(t) for t in times]
            self.times = np.concatenate(times).astype("datetime64[us]")

        self.file_of_frame = np.repeat(np.arange(len(self.files)), frames_per_file)
        self.index_in_file = np.concatenate(
            [np.arange(n) for n in frames_per_file] + [np.zeros(0, int)]
        ).astype(np.int64)

        self._variables = {}
        self._frames = {}  # (varname, frame) -> _Frame
        self._free = collections.defaultdict(list)  # (shape, dtype) -> arrays
        self._stamp = 0
        self._readers = {}  # (file number, varname) -> ChunkLayout or NetCDFReader
        self._reader_lock = threading.Lock()
        self._fds = {}
        self._chunk_cache = _ChunkCache(max_bytes=512 * 2**20)
        self._pool = ThreadPoolExecutor(
            max_workers=self.num_threads, thread_name_prefix="chunkloader"
        )
        logger.info(
            f"Chunk loader: {len(self.files)} files, {len(self.times)} frames, "
            f"{self.num_threads} threads, inflate = {codec.INFLATE_BACKEND}"
        )

    # ---------- Public interface ----------

    def __getitem__(self, name) -> "Variable":
        var = self._variables.get(name)
        if var is None:
            var = self._variables[name] = Variable(self, name)
        return var

    def __contains__(self, name):
        try:
            self[name]
            return True
        except KeyError:
            return False

    @property
    def num_frames(self):
        return len(self.times)

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        for reader in self._readers.values():
            if isinstance(reader, codec.NetCDFReader):
                reader.close()
        self._frames.clear()
        self._free.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ---------- Frame buffers ----------

    def _frame(self, var, n):
        """Frame buffer of variable var at frame n, created if necessary"""
        self._stamp += 1
        key = (var.name, n)
        fr = self._frames.get(key)
        if fr is not None:
            fr.stamp = self._stamp
            return fr

        own = [k for k in self._frames if k[0] == var.name]
        while len(own) >= self.max_frames:
            oldest = min(own, key=lambda k: self._frames[k].stamp)
            own.remove(oldest)
            self._release(oldest)

        fkey = (var.shape3, var.dtype.str)
        data = self._free[fkey].pop() if self._free[fkey] else np.empty(var.shape3, var.dtype)
        fr = self._frames[key] = _Frame(data, var.grid3, self._stamp)
        return fr

    def _release(self, key):
        fr = self._frames.pop(key)
        for fut in fr.pending.values():
            fut.exception()  # Wait for completion, ignore errors
        self._free[(fr.data.shape, fr.data.dtype.str)].append(fr.data)

    def _request(self, var, n, need, wait):
        """Decode the chunks flagged in need (and not already loaded)"""
        fr = self._frame(var, n)
        for c, fut in list(fr.pending.items()):
            if fut.done():
                del fr.pending[c]
        todo = np.argwhere((need != 0) & (fr.loaded == 0))
        futures = []
        for c in map(tuple, todo.tolist()):
            fut = fr.pending.get(c)
            if fut is None:
                fut = self._pool.submit(self._load_chunk, var, n, fr, c)
                fr.pending[c] = fut
            futures.append(fut)
        if wait:
            for fut in futures:
                fut.result()
        return fr

    # ---------- Decoding (runs in the worker threads) ----------

    def _reader(self, fnum, var):
        key = (fnum, var.name)
        reader = self._readers.get(key)
        if reader is None:
            with self._reader_lock:
                reader = self._readers.get(key)
                if reader is None:
                    reader = self._make_reader(fnum, var)
                    self._readers[key] = reader
        return reader

    def _make_reader(self, fnum, var):
        path = self.files[fnum]
        layout = codec.open_layout(path, var.name)
        if layout is not None:
            ok_shape = layout.shape[var.ntime:] == var.shape
            ok_chunks = (not var.ntime or layout.chunks[0] >= 1) and (
                layout.chunks[var.ntime:] == var.chunks
            )
            if ok_shape and ok_chunks:
                if fnum not in self._fds:
                    self._fds[fnum] = os.open(path, os.O_RDONLY)
                return layout
        return codec.NetCDFReader(path, var.name, var.chunks)

    def _load_chunk(self, var, n, fr, c3):
        """Decode spatial chunk c3 of frame n into the frame buffer fr"""
        region = tuple(
            slice(ci * cs, min((ci + 1) * cs, s))
            for ci, cs, s in zip(c3, var.chunks3, var.shape3)
        )
        pad = 3 - len(var.shape)  # Number of padding axes
        if var.ntime:
            fnum = int(self.file_of_frame[n])
            tidx = int(self.index_in_file[n])
        else:
            fnum, tidx = 0, None

        reader = self._reader(fnum, var)
        if isinstance(reader, codec.ChunkLayout):
            cidx = tuple(c3[pad:])
            if var.ntime:
                ct = reader.chunks[0]
                cidx = (tidx // ct,) + cidx
                if ct == 1:
                    src = reader.decode(self._fds[fnum], cidx)[0]
                else:
                    src = self._chunk_cache.get(
                        (fnum, var.name, cidx),
                        lambda: reader.decode(self._fds[fnum], cidx),
                    )[tidx % ct]
            else:
                src = reader.decode(self._fds[fnum], cidx)
            src = src.reshape(var.chunks3)
            src = src[tuple(slice(0, r.stop - r.start) for r in region)]
        else:
            nc_region = tuple(region[pad:])
            if var.ntime:
                nc_region = (tidx,) + nc_region
            src = reader.read(nc_region)

        fr.data[region] = src.reshape(fr.data[region].shape)
        fr.loaded[c3] = 1


class Variable:
    """A variable of a :class:`Dataset`, with lazily loaded chunks

    :ivar shape: Shape of one time frame
    :ivar chunks: Chunk shape of one time frame
    :ivar ntime: 1 if the variable depends on time, else 0
    :ivar scale, offset: Decoded values are ``offset + scale * raw``
    """

    def __init__(self, dataset: Dataset, name: str):
        self.dataset = dataset
        self.name = name
        meta = read_metadata(dataset.files[0], name, dataset.time_name)
        self.dims = meta["dims"]
        self.ntime = int(bool(meta["time_dependent"]))
        self.shape = tuple(meta["shape"][self.ntime:])
        self.chunks = tuple(meta["chunks"][self.ntime:])
        if len(self.shape) > 3:
            raise NotImplementedError("At most 3 spatial dimensions are supported")
        self.dtype = np.dtype(meta["dtype"]).newbyteorder("=")
        self.scale = float(meta.get("scale_factor", 1.0))
        self.offset = float(meta.get("add_offset", 0.0))
        self.fill_value = meta.get("_FillValue", None)

        pad = 3 - len(self.shape)
        self.shape3 = (1,) * pad + self.shape
        self.chunks3 = (1,) * pad + self.chunks
        self.grid3 = tuple(-(-s // c) for s, c in zip(self.shape3, self.chunks3))
        self._chunks_arr = np.array(self.chunks3, dtype=np.int64)
        self._shape_arr = np.array(self.shape3, dtype=np.int64)

    def __repr__(self):
        return f"<Variable {self.name}{self.dims} {self.dtype} chunks={self.chunks}>"

    # ---------- Frames ----------

    def _frames(self, frame, linear=True):
        """(n0, n1, w1) from a frame specification

        :param frame: An integer (exact frame), a float (fractional frame,
            interpolated if linear else nearest) or a tuple (n0, n1, w1)
        :param linear: Linear interpolation in time for fractional frames
        """
        if not self.ntime:
            return 0, 0, 0.0
        nmax = self.dataset.num_frames - 1
        if isinstance(frame, tuple):
            n0, n1, w1 = frame
            n0, n1 = int(n0), int(n1)
        elif isinstance(frame, (int, np.integer)):
            n0, n1, w1 = int(frame), int(frame), 0.0
        elif linear:
            f = min(max(float(frame), 0.0), float(nmax))
            n0 = int(np.floor(f))
            n1 = min(n0 + 1, nmax)
            w1 = f - n0
        else:
            n0 = n1 = min(max(int(np.rint(float(frame))), 0), nmax)
            w1 = 0.0
        for n in (n0, n1):
            if not 0 <= n <= nmax:
                raise IndexError(f"Frame {n} out of range [0, {nmax}]")
        if n1 == n0:
            w1 = 0.0
        return n0, n1, float(w1)

    def loaded(self, frame):
        """Loaded-flags of the chunks of a frame (uint8, shape grid3)"""
        n0, _, _ = self._frames(frame)
        fr = self.dataset._frames.get((self.name, n0))
        if fr is None:
            return np.zeros(self.grid3, dtype=np.uint8)
        return fr.loaded

    def prefetch(self, frame, chunks=None):
        """Start decoding chunks of a frame in the background.

        :param frame: Frame number
        :param chunks: uint8 array of shape grid3 flagging the chunks to load,
            or None for all chunks
        """
        n0, _, _ = self._frames(int(frame))
        if chunks is None:
            chunks = np.ones(self.grid3, dtype=np.uint8)
        self.dataset._request(self, n0, chunks, wait=False)

    def ensure(self, frame, chunks=None):
        """Decode chunks of a frame and wait for them (see prefetch)"""
        n0, _, _ = self._frames(int(frame))
        if chunks is None:
            chunks = np.ones(self.grid3, dtype=np.uint8)
        return self.dataset._request(self, n0, chunks, wait=True)

    # ---------- Data access ----------

    def read(self, frame=0, region=None, dtype=None):
        """Decoded values of one frame, or of a region of it

        :param frame: Frame number (int), fractional frame (linear
            interpolation in time) or a tuple (n0, n1, w1)
        :param region: Tuple of slices over the spatial dimensions
        :param dtype: Output data type, default float32 for packed data
        :return: Array of decoded values, shape of the region
        """
        if dtype is None:
            dtype = np.float32 if self.dtype.itemsize <= 4 else np.float64
        if region is None:
            region = (slice(None),) * len(self.shape)
        pad = 3 - len(self.shape)
        region3 = (slice(None),) * pad + tuple(region)
        need = np.zeros(self.grid3, dtype=np.uint8)
        idx = [np.arange(s)[r] for s, r in zip(self.shape3, region3)]
        csel = tuple(np.unique(i // c) for i, c in zip(idx, self.chunks3))
        need[np.ix_(*csel)] = 1

        n0, n1, w1 = self._frames(frame)
        values = self._decode(self.dataset._request(self, n0, need, wait=True), region3, dtype)
        if w1:
            v1 = self._decode(self.dataset._request(self, n1, need, wait=True), region3, dtype)
            values = values + dtype(w1) * (v1 - values)
        return values.reshape(values.shape[pad:])

    def _decode(self, fr, region3, dtype):
        raw = fr.data[region3]
        if self.scale == 1.0 and self.offset == 0.0:
            return raw.astype(dtype)
        values = dtype(self.scale) * raw.astype(dtype)
        if self.offset:
            values += dtype(self.offset)
        return values

    def interp(self, coords, frame=0, linear="tzyx", mask=None, out=None):
        """Sample the variable at grid-native (fractional) indices.

        Missing chunks are loaded (and waited for) as needed.

        :param coords: Sequence with one coordinate array per spatial
            dimension, e.g. ``[k, j, i]``. Arrays of length 1 are broadcast.
            Integer arrays are exact indices and are never interpolated.
        :param frame: Time frame: an int (exact frame), a float (fractional
            frame) or a tuple ``(n0, n1, w1)``, giving
            ``(1 - w1) * F[n0] + w1 * F[n1]``.
        :param linear: The dimensions to interpolate linearly, any combination
            of the letters "tzyx". Here "x" is the last spatial dimension, "y"
            the one before, etc, and "t" is time. Fractional coordinates along
            the other dimensions are rounded to the nearest grid point.
        :param mask: Optional array over the last two spatial dimensions.
            Grid points where mask == 0 contribute zero to the result.
        :param out: Optional float64 output array
        :return: float64 array with the sampled values
        """
        nd = len(self.shape)
        if len(coords) != nd:
            raise ValueError(f"Expected {nd} coordinate arrays, got {len(coords)}")
        if set(linear) - set("tzyx"):
            raise ValueError(f"Invalid dimension letters: {linear}")
        pad = 3 - nd
        coords = [np.asarray(c) for c in coords]
        time_linear = "t" in linear
        linear = np.array([False] * pad + [
            letter in linear and c.dtype.kind == "f"
            for letter, c in zip("zyx"[pad:], coords)
        ], dtype=np.bool_)
        cc = [np.zeros(1)] * pad + [
            np.ascontiguousarray(c, dtype=np.float64).reshape(-1) for c in coords
        ]
        npart = max(c.size for c in cc)
        if out is None:
            out = np.empty(npart, dtype=np.float64)
        if npart == 0:
            return out

        if mask is None:
            use_mask, mask = False, np.ones((1, 1), dtype=np.uint8)
        else:
            use_mask = True
            mask = np.asarray(mask)
            if mask.shape != self.shape3[1:]:
                raise ValueError("Mask shape must equal the last two dimensions")

        n0, n1, w1 = self._frames(frame, time_linear)
        ds = self.dataset
        fr0 = ds._frame(self, n0)
        fr1 = ds._frame(self, n1) if w1 else fr0
        miss = np.empty(npart, dtype=np.uint8)

        def run(k, j, i, o, m):
            return kernels.interp(
                fr0.data, fr0.loaded, fr1.data, fr1.loaded, w1, k, j, i, linear,
                self._chunks_arr, self.scale, self.offset, mask, use_mask, o, m,
            )

        nmiss = run(cc[0], cc[1], cc[2], out, miss)
        if nmiss:
            idx = np.flatnonzero(miss)
            need = np.zeros(self.grid3, dtype=np.uint8)
            kernels.mark_chunks(idx, cc[0], cc[1], cc[2], linear, self._shape_arr,
                                self._chunks_arr, need)
            ds._request(self, n0, need, wait=True)
            if w1:
                ds._request(self, n1, need, wait=True)
            sub = [c[idx] if c.size > 1 else c for c in cc]
            sub_out = np.empty(idx.size, dtype=np.float64)
            if run(sub[0], sub[1], sub[2], sub_out, miss[: idx.size]):
                raise RuntimeError("Chunks still missing after loading")
            out[idx] = sub_out
        return out


class _Frame:
    """Raw data of one time frame, with loaded-flags for the chunks"""

    __slots__ = ("data", "loaded", "pending", "stamp")

    def __init__(self, data, grid, stamp):
        self.data = data
        self.loaded = np.zeros(grid, dtype=np.uint8)
        self.pending = {}  # chunk index -> Future
        self.stamp = stamp


class _ChunkCache:
    """Small thread-safe LRU cache of decoded chunks spanning several frames"""

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self._items = collections.OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key, make):
        with self._lock:
            arr = self._items.get(key)
            if arr is not None:
                self._items.move_to_end(key)
                return arr
        arr = make()
        with self._lock:
            if key not in self._items:
                self._items[key] = arr
                self._bytes += arr.nbytes
                while self._bytes > self.max_bytes and len(self._items) > 1:
                    _, old = self._items.popitem(last=False)
                    self._bytes -= old.nbytes
        return arr


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

def _attr(value):
    """netCDF attribute value from h5py (arrays of length 1, bytes)"""
    if isinstance(value, bytes):
        return value.decode()
    value = np.asarray(value)
    if value.dtype.kind in "SO":
        v = value.ravel()[0]
        return v.decode() if isinstance(v, bytes) else str(v)
    return value.ravel()[0].item() if value.size == 1 else value


def read_metadata(path, name, time_name):
    """Dimensions, shape, chunking, dtype and scaling attributes of a variable

    Raises KeyError if the variable does not exist.
    """
    meta = None
    if codec.h5py is not None:
        try:
            with codec.h5py.File(path, "r") as f:
                meta = _metadata_h5(f, name, time_name)
        except OSError:
            meta = None  # Not an HDF5 file
    if meta is None:
        from netCDF4 import Dataset

        with Dataset(path) as nc:
            if name not in nc.variables:
                raise KeyError(name)
            v = nc.variables[name]
            chunking = v.chunking()
            tdims = nc.variables[time_name].dimensions if time_name else ()
            meta = dict(
                dims=v.dimensions,
                shape=v.shape,
                chunks=tuple(chunking) if isinstance(chunking, list) else v.shape,
                dtype=v.dtype,
                time_dependent=bool(tdims) and v.dimensions[:1] == tdims[:1],
            )
            for a in ("scale_factor", "add_offset", "_FillValue"):
                if a in v.ncattrs():
                    meta[a] = _attr(v.getncattr(a))
    # Time dependent variables: chunk shape of one frame
    meta["chunks"] = tuple(meta["chunks"])
    if meta["time_dependent"]:
        meta["chunks"] = (1,) + meta["chunks"][1:]
    return meta


def _metadata_h5(f, name, time_name):
    if name not in f:
        raise KeyError(name)
    d = f[name]
    dims = []
    for dim in d.dims:
        try:
            dims.append(dim[0].name.rsplit("/", 1)[-1])
        except Exception:
            dims.append(None)
    tdim = None
    if time_name and time_name in f:
        try:
            tdim = f[time_name].dims[0][0].name.rsplit("/", 1)[-1]
        except Exception:
            tdim = time_name
    meta = dict(
        dims=tuple(dims),
        shape=d.shape,
        chunks=d.chunks if d.chunks is not None else d.shape,
        dtype=d.dtype,
        time_dependent=tdim is not None and bool(dims) and dims[0] == tdim,
    )
    for a in ("scale_factor", "add_offset", "_FillValue"):
        if a in d.attrs:
            meta[a] = _attr(d.attrs[a])
    return meta


def read_times(path, time_name):
    """Times of all frames in a file, as datetime64[us]"""
    values = units = None
    calendar = "standard"
    if codec.h5py is not None:
        try:
            with codec.h5py.File(path, "r") as f:
                v = f[time_name]
                values = np.asarray(v[()]).ravel()
                units = _attr(v.attrs["units"])
                if "calendar" in v.attrs:
                    calendar = _attr(v.attrs["calendar"])
        except OSError:
            values = None
    if values is None:
        from netCDF4 import Dataset

        with Dataset(path) as nc:
            v = nc.variables[time_name]
            v.set_auto_mask(False)
            values = np.asarray(v[:]).ravel()
            units = v.units
            calendar = getattr(v, "calendar", "standard")
    return to_datetime64(values, units, calendar)


def to_datetime64(values, units, calendar="standard"):
    """Convert CF time values to datetime64[us]"""
    import cftime

    dates = cftime.num2date(
        values, units, calendar, only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return np.array([np.datetime64(d, "us") for d in np.ravel(dates)],
                    dtype="datetime64[us]")
