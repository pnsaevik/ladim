"""
Direct access to HDF5 chunks, bypassing the netCDF4/HDF5 libraries.

The byte offsets of the chunks are found once per file and variable using
h5py. The chunks are then read with ``os.pread``, inflated with libdeflate
(``deflate`` package), ISA-L (``isal`` package) or zlib, whichever is
available, and un-shuffled by a numba kernel. All of these release the GIL,
so chunks can be decoded in parallel by a thread pool.

Only the filters shuffle and deflate are supported. Variables with other
filters, or files that are not HDF5, are read through netCDF4 instead
(:class:`NetCDFReader`).
"""

import logging
import os
import sys
import threading
import zlib

import numpy as np

from . import kernels

logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

_FILTER_DEFLATE = 1
_FILTER_SHUFFLE = 2


# --------------------------------------------------------------------------
# Inflate
# --------------------------------------------------------------------------

def _select_inflate():
    try:
        import deflate

        return "libdeflate", lambda raw, n: deflate.zlib_decompress(raw, n)
    except ImportError:
        pass
    try:
        from isal import isal_zlib

        return "isal", lambda raw, n: isal_zlib.decompress(raw, bufsize=n)
    except ImportError:
        pass
    return "zlib", lambda raw, n: zlib.decompress(raw, bufsize=n)


INFLATE_BACKEND, inflate = _select_inflate()


# --------------------------------------------------------------------------
# HDF5 chunk layout
# --------------------------------------------------------------------------

class ChunkLayout:
    """Byte offsets and filters of the chunks of one variable in one file

    :ivar shape: Shape of the variable
    :ivar chunks: Chunk shape
    :ivar grid: Number of chunks along each axis
    :ivar offset: Byte offset of each chunk (-1 if not allocated)
    :ivar size: Stored size in bytes of each chunk
    """

    def __init__(self, dset):
        self.shape = tuple(dset.shape)
        self.dtype = dset.dtype
        self.fillvalue = dset.fillvalue
        plist = dset.id.get_create_plist()
        layout = plist.get_layout()

        if layout == h5py.h5d.CHUNKED:
            self.chunks = tuple(dset.chunks)
            filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
        elif layout == h5py.h5d.CONTIGUOUS:
            self.chunks = self.shape
            filters = []
        else:
            raise NotImplementedError("Unsupported storage layout")

        if filters not in ([], [_FILTER_DEFLATE], [_FILTER_SHUFFLE],
                           [_FILTER_SHUFFLE, _FILTER_DEFLATE]):
            raise NotImplementedError(f"Unsupported HDF5 filters {filters}")
        self.shuffle = _FILTER_SHUFFLE in filters and self.dtype.itemsize > 1
        self.deflate = _FILTER_DEFLATE in filters
        if self.shuffle and self.dtype.itemsize not in kernels.UNSHUFFLE:
            raise NotImplementedError("Unsupported item size")

        self.grid = tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))
        self.offset = np.full(self.grid, -1, dtype=np.int64)
        self.size = np.zeros(self.grid, dtype=np.int64)

        if layout == h5py.h5d.CONTIGUOUS:
            off = dset.id.get_offset()
            if off is not None:
                self.offset[...] = off
                self.size[...] = dset.id.get_storage_size()
            return

        def visit(info):
            if info.filter_mask != 0:
                raise NotImplementedError("Chunk with skipped filters")
            idx = tuple(o // c for o, c in zip(info.chunk_offset, self.chunks))
            self.offset[idx] = info.byte_offset
            self.size[idx] = info.size

        dset.id.chunk_iter(visit)

    def decode(self, fd, cidx):
        """Read and decode one chunk.

        :param fd: File descriptor of the HDF5 file
        :param cidx: Chunk index (tuple)
        :return: Array with the full chunk shape and native byte order
        """
        off = int(self.offset[cidx])
        dtype = self.dtype
        if off < 0:
            return np.full(self.chunks, self.fillvalue, dtype=dtype.newbyteorder("="))

        size = int(self.size[cidx])
        raw = os.pread(fd, size, off)
        if len(raw) != size:
            raise IOError(f"Short read ({len(raw)} of {size} bytes)")

        n = int(np.prod(self.chunks))
        nbytes = n * dtype.itemsize
        data = inflate(raw, nbytes) if self.deflate else raw

        if self.shuffle and sys.byteorder == "little":
            # The kernel assembles the bytes as little-endian integers, i.e.
            # in the stored byte order in memory
            unshuffle, utype = kernels.UNSHUFFLE[dtype.itemsize]
            u = np.empty(n, dtype=utype)
            unshuffle(np.frombuffer(data, dtype=np.uint8, count=nbytes), u)
            arr = u.view(dtype)
        elif self.shuffle:
            planes = np.frombuffer(data, dtype=np.uint8, count=nbytes)
            arr = planes.reshape(dtype.itemsize, n).T.copy().view(dtype).reshape(n)
        else:
            arr = np.frombuffer(data, dtype=dtype, count=n)

        if not arr.dtype.isnative:
            arr = arr.astype(arr.dtype.newbyteorder("="))
        return arr.reshape(self.chunks)


def open_layout(path, name):
    """ChunkLayout of variable name in file path, or None if not supported"""
    if h5py is None:
        return None
    try:
        with h5py.File(path, "r") as f:
            return ChunkLayout(f[name])
    except (OSError, NotImplementedError, KeyError, TypeError) as e:
        logger.info(f"Direct chunk access not possible for {name} in {path}: {e}")
        return None


# --------------------------------------------------------------------------
# Fallback: netCDF4
# --------------------------------------------------------------------------

class NetCDFReader:
    """Reads chunk-sized regions through the netCDF4 library.

    The netCDF4 library is not thread safe, so all reads are serialized.
    """

    _lock = threading.Lock()

    def __init__(self, path, name, chunks):
        self.path = path
        self.name = name
        self.chunks = chunks
        self._nc = None

    def read(self, region):
        from netCDF4 import Dataset

        with self._lock:
            if self._nc is None:
                self._nc = Dataset(self.path)
                self._nc.set_auto_maskandscale(False)
            var = self._nc.variables[self.name]
            return np.asarray(var[region])

    def close(self):
        with self._lock:
            if self._nc is not None:
                self._nc.close()
                self._nc = None
