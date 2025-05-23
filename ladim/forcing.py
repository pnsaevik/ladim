import typing
if typing.TYPE_CHECKING:
    from ladim.model import Model
import numexpr
import string
import numpy as np
from numba import njit


class Forcing:
    @staticmethod
    def from_roms(**conf):
        return RomsForcing(**conf)

    def velocity(self, X, Y, Z, tstep=0.0):
        raise NotImplementedError

    def update(self, model: "Model"):
        raise NotImplementedError


class RomsForcing(Forcing):
    def __init__(self, file, variables=None, **conf):
        """
        Forcing module which uses output data from the ROMS ocean model

        :param file: Glob pattern for the input files
        :param variables: A mapping of variable names to interpolation
        specifications. Each interpolaction specification consists of 0-4
        of the letters "xyzt". Coordinates that are listed in the string are
        interpolated linearly, while the remaining ones use nearest-neighbor
        interpolation. Some default configurations are defined:

        .. code-block:: json
            {
                "temp": "xyzt",
                "salt": "xyzt",
                "u": "xt",
                "v": "yt",
                "w": "zt",
            }


        :param conf: Legacy config dict
        """
        # Apply default interpolation configs
        variables = variables or dict()
        default_vars = dict(u="xt", v="yt", w="zt", temp="xyzt", salt="xyzt")
        self.variables = {**default_vars, **variables}

        grid_ref = GridReference()
        legacy_conf = dict(
            gridforce=dict(input_file=file, **conf),
            ibm_forcing=conf.get('ibm_forcing', []),
            start_time=conf.get('start_time', None),
            stop_time=conf.get('stop_time', None),
            dt=conf.get('dt', None),
        )
        if conf.get('subgrid', None) is not None:
            legacy_conf['gridforce']['subgrid'] = conf['subgrid']

        from .utilities import load_class
        LegacyForcing = load_class(conf.get('legacy_module', 'ladim.gridforce.ROMS.Forcing'))

        # Allow gridforce module in current directory
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        # Import correct gridforce_module
        self.forcing = LegacyForcing(legacy_conf, grid_ref)
        # self.steps = self.forcing.steps
        # self.U = self.forcing.U
        # self.V = self.forcing.V

    def update(self, model: "Model"):
        elapsed = model.solver.time - model.solver.start
        t = elapsed // model.solver.step

        # noinspection PyProtectedMember
        self.forcing._grid.modules = model
        self.forcing.update(t)

        # Update state variables by sampling the field
        x, y, z = model.state['X'], model.state['Y'], model.state['Z']
        for v in self.variables:
            if v in model.state:
                model.state[v] = self.field(x, y, z, v)

    def velocity(self, X, Y, Z, tstep=0.0):
        return self.forcing.velocity(X, Y, Z, tstep=tstep)

    def field(self, X, Y, Z, name):
        return self.forcing.field(X, Y, Z, name)

    def close(self):
        return self.forcing.close()


class GridReference:
    def __init__(self):
        self.modules = None

    def __getattr__(self, item):
        return getattr(self.modules.grid.grid, item)


def load_netcdf_chunk(url, varname, subset):
    """
    Download, unzip and decode a netcdf chunk from file or url
    """
    import xarray as xr
    with xr.open_dataset(url) as dset:
        values = dset.variables[varname][subset].values
    if varname in ['u', 'v', 'w']:
        values = np.nan_to_num(values)
    return values


class ChunkCache:
    """
    A cache for storing and sharing chunks of data using shared memory.

    This class manages a memory block divided into a header, index, and data section.
    It is designed for efficient inter-process communication of chunked data arrays.

    :ivar mem: SharedMemory object representing the memory block.
    :ivar num_chunks: Number of slots/chunks in the cache (read-only).
    :ivar chunksize: Size of each chunk (read-only).
    :ivar datatype: Data type of the stored chunks (read-only).
    :ivar itemsize: Size in bytes of each data item (read-only).
    :ivar chunk_id: Array of chunk IDs for tracking which data is stored in each slot.
    :ivar data: 2D array holding the actual chunked data.
    """
    def __init__(self, name: str):
        """
        Attach to an existing shared memory block and map the cache structure.

        :param name: The name of the shared memory block to attach to.
        """
        from multiprocessing.shared_memory import SharedMemory
        mem = SharedMemory(name=name, create=False)
        self.mem = mem

        # Header block
        self.num_chunks = np.ndarray(shape=(), dtype=np.int64, buffer=mem.buf[0:8])
        self.chunksize = np.ndarray(shape=(), dtype=np.int64, buffer=mem.buf[8:16])
        self.datatype = np.ndarray(shape=(), dtype='S8', buffer=mem.buf[16:24])
        self.itemsize = np.ndarray(shape=(), dtype=np.int64, buffer=mem.buf[24:32])
        self.num_chunks.setflags(write=False)
        self.chunksize.setflags(write=False)
        self.datatype.setflags(write=False)
        self.itemsize.setflags(write=False)

        # LRU block
        lru_start = 32
        lru_stop = lru_start + 2*self.num_chunks
        self.lru = np.ndarray(
            shape=(self.num_chunks,),
            dtype=np.int16,
            buffer=mem.buf[lru_start:lru_stop])

        # Index block
        idx_start = lru_stop
        idx_stop = idx_start + 8*self.num_chunks
        self.chunk_id = np.ndarray(
            shape=(self.num_chunks, ),
            dtype=np.int64,
            buffer=mem.buf[idx_start:idx_stop])
        
        # Data block
        dat_start = idx_stop
        dat_stop = dat_start + self.num_chunks * self.chunksize * self.itemsize
        self.data = np.ndarray(
            shape=(self.num_chunks, self.chunksize),
            dtype=self.datatype.item().decode('ascii'),
            buffer=mem.buf[dat_start:dat_stop])

    def _update_lru(self, slot: int) -> None:
        """
        Move the given slot to the front (most recently used) in the LRU table.
        """
        update_lru(self.lru, slot)

    def read(self, slot: int) -> np.ndarray:
        """
        Read data from the given slot and update the LRU table.

        :param slot: The slot index to read
        :return: The data in the slot
        """
        self._update_lru(slot)
        return self.data[slot, :]

    def write(self, data: np.ndarray, slot: int) -> None:
        """
        Overwrite the data in the given slot and update the LRU table.

        :param data: 1D numpy array of length self.chunksize and dtype self.datatype
        :param slot: The slot index to overwrite
        """
        self._update_lru(slot)
        self.data[slot, :] = data

    def __enter__(self) -> "ChunkCache":
        """
        Enter the runtime context related to this object.
        Returns self for use in 'with' statements.

        :return: self
        """
        return self

    def __exit__(self, type: type, value: Exception, tb: object) -> None:
        """
        Exit the runtime context and close the shared memory.

        :param type: Exception type
        :param value: Exception value
        :param tb: Traceback object
        """
        self.close()

    def __setattr__(self, name: str, value: object) -> None:
        """
        Prevent reassignment of attributes after initialization.
        Raises AttributeError if an attribute is already set.

        :param name: Attribute name
        :param value: Attribute value
        :raises AttributeError: If attribute is already set
        """
        if hasattr(self, name):
            raise AttributeError(f"Cannot reassign attribute '{name}'")
        super().__setattr__(name, value)

    @staticmethod
    def create(slots: int, chunksize: int, datatype: str = 'f4') -> "ChunkCache":
        """
        Create a new shared memory block and initialize a ChunkCache.

        :param slots: Number of slots/chunks in the cache.
        :param chunksize: Size of each chunk.
        :param datatype: Numpy dtype string for the data (default 'f4').
        :return: An instance attached to the new shared memory block.
        """
        from multiprocessing.shared_memory import SharedMemory

        test_item = np.empty((), dtype=datatype)
        str_dtype = str(test_item.dtype)
        if len(str_dtype) > 8:
            raise ValueError('Unsupported data type: {str_dtype}')
        
        # Reserve memory space for the cache block
        size_header_block = 32
        size_lru_block = 2 * slots  # int16
        size_index_block = 8 * slots
        size_data_block = slots * chunksize * test_item.itemsize
        bytes = size_header_block + size_lru_block + size_index_block + size_data_block
        mem = SharedMemory(create=True, size=bytes)

        # Write some header information
        mem_slots = np.ndarray(shape=(), dtype=np.int64, buffer=mem.buf[0:8])
        mem_slots[...] = slots
        mem_chunksize = np.ndarray(shape=(), dtype=np.int64, buffer=mem.buf[8:16])
        mem_chunksize[...] = chunksize
        mem_datatype = np.ndarray(shape=(), dtype='S8', buffer=mem.buf[16:24])
        mem_datatype[...] = str_dtype
        mem_itemsize = np.ndarray(shape=(), dtype=np.int64, buffer=mem.buf[24:32])
        mem_itemsize[...] = test_item.itemsize

        # LRU block
        lru_start = size_header_block
        mem_lru = np.ndarray(
            shape=(slots,),
            dtype=np.int16,
            buffer=mem.buf[lru_start:lru_start + size_lru_block])
        mem_lru[...] = np.arange(slots, dtype=np.int16)

        # Index block
        index_start = lru_start + size_lru_block
        mem_chunkid = np.ndarray(
            shape=(slots, ),
            dtype=np.int64,
            buffer=mem.buf[index_start:index_start + size_index_block])
        mem_chunkid[...] = -1
        
        # Data block
        # (no need to initialize, will be written on use)
        return ChunkCache(mem.name)

    
    def close(self) -> None:
        """
        Close the shared memory block.
        """
        self.mem.close()

    def contains(self, id: int) -> bool:
        """
        Check if the cache contains a chunk with the given id.

        :param id: The chunk id to check
        :return: True if the chunk is in the cache, False otherwise
        """
        return indexof(self.chunk_id, id) >= 0

    def push(self, data: np.ndarray, id: int) -> None:
        """
        Push a chunk of data into the cache with the given id.

        :param data: 1D numpy array of length self.chunksize and dtype self.datatype
        :param id: The chunk id to associate with this data
        :note: If no free slots are available, evict the least recently used slot.
        """
        free_slots = np.where(self.chunk_id == -1)[0]
        if len(free_slots) > 0:
            slot = free_slots[0]
        else:
            # Evict the least recently used slot (last in lru)
            slot = self.lru[-1]
        self.write(data, slot)
        self.chunk_id[slot] = id

    def pull(self, id: int) -> np.ndarray:
        """
        Retrieve the data for the given chunk id and update the LRU table.

        :param id: The chunk id to retrieve
        :return: The data array for the chunk
        :raises KeyError: If the chunk id is not found in the cache
        """
        slot = indexof(self.chunk_id, id)
        if slot < 0:
            raise KeyError(f"Chunk id {id} not found in cache")
        return self.read(slot)


def timestring_formatter(pattern, time):
    """
    Format a time string

    :param pattern: f-string style formatting pattern
    :param time: Numpy convertible time specification
    :returns: A formatted time string
    """
    posix_time = np.datetime64(time, 's').astype(int)

    class PosixFormatter(string.Formatter):
        def get_value(self, key: int | str, args: typing.Sequence[typing.Any], kwargs: typing.Mapping[str, typing.Any]) -> typing.Any:
            return numexpr.evaluate(
                key, local_dict=kwargs, global_dict=dict())
        
        def format_field(self, value: typing.Any, format_spec: str) -> typing.Any:
            dt = np.int64(value).astype('datetime64[s]').astype(object)
            return dt.strftime(format_spec)
        
    fmt = PosixFormatter()
    return fmt.format(pattern, time=posix_time)


@njit
def update_lru(lru: np.ndarray, slot: int) -> None:
    """
    Update the LRU (Least Recently Used) list by moving the specified slot to the front.

    :param lru: The LRU array
    :param slot: The slot index to move to the front
    """
    v = slot
    for i in range(len(lru)):
        u = lru[i]
        lru[i] = v
        if u == slot:
            break
        v = u


@njit
def indexof(array: np.ndarray, value: int) -> int:
    """
    Find the index of the first occurrence of a value in an array.

    :param array: The input array
    :param value: The value to find
    :return: The index of the first occurrence, or -1 if not found
    """
    for i in range(len(array)):
        if array[i] == value:
            return i
    return -1


@njit(inline="always", fastmath=True)
def get_chunk_id(i, j, l, size_x, size_y, num_x, num_y):
    """
    Calculate the chunk ID based on the indices and sizes.

    We assume that the array is chunked in the x and y dimensions,
    but not in the z dimension. The t dimension is assumed to be
    chunked with size_t = 1.

    For instance, if the chunk size is x:10, y:5 and the number of chunks
    in the x and y dimensions is 6 and 7 respectively, then the chunk
    ID for the coordinates (31, 14, 0) would be calculated as follows:
        
        chunk id in x direction: 31 // 10 = 3
        chunk id in y direction: 14 // 5 = 2
        chunk id in t direction: 0 // 1 = 0
        chunk id = 3 + 6*2 + 6*7*0 = 15
    
    This means that the chunk ID is a unique identifier for the chunk
    containing the coordinates (31, 14, 0) in the array.

    :param i: Integer x coordinate (global index)
    :param j: Integer y coordinate (global index)
    :param l: Integer t coordinate (global index)
    :param size_x: Chunk size in x dimension
    :param size_y: Chunk size in y dimension
    :param num_x: Number of chunks in x dimension
    :param num_y: Number of chunks in y dimension
    :return: Chunk ID
    """

    # The global index is divided by the chunk size to get the chunk ID
    # for each dimension. The chunk ID is then combined into a single
    # integer value.
    return (i // size_x) + num_x * ((j // size_y) + num_y * l)


@njit(inline="always", fastmath=True)
def interp_xyzt(x, y, z, t, v, ncx, ncy, ncz, nct, cache, ids):
    """
    Interpolate the data in the x, y, z, and t dimensions.
    
    :param x: x coordinate (global index)
    :param y: y coordinate (global index)
    :param z: z coordinate (global index)
    :param t: t coordinate (global index)
    :param v: v coordinate (global index)
    :param ncx: Number of chunks in x dimension
    :param ncy: Number of chunks in y dimension
    :param ncz: Number of chunks in z dimension
    :param nct: Number of chunks in t dimension
    :param cache: Chunk cache array
    :param ids: Array of chunk ids for each slot in the cache
    :return: Interpolated value
    """
    _, _, st, sz, sy, sx = cache.shape
    
    max_x = ncx * sx
    max_y = ncy * sy
    max_z = ncz * sz
    max_t = nct * st    

    i0 = max(0, min(max_x - 2, np.int32(x)))
    j0 = max(0, min(max_y - 2, np.int32(y)))
    k0 = max(0, min(max_z - 2, np.int32(z)))
    l0 = max(0, min(max_t - 2, np.int32(t)))

    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1
    l1 = l0 + 1

    # Chunk ID (chid) for the surrounding points
    chid0000 = get_chunk_id(i0, j0, l0, sx, sy, ncx, ncy)
    chid0001 = get_chunk_id(i1, j0, l0, sx, sy, ncx, ncy)
    chid0010 = get_chunk_id(i0, j1, l0, sx, sy, ncx, ncy)
    chid0011 = get_chunk_id(i1, j1, l0, sx, sy, ncx, ncy)
    chid1000 = get_chunk_id(i0, j0, l1, sx, sy, ncx, ncy)
    chid1001 = get_chunk_id(i1, j0, l1, sx, sy, ncx, ncy)
    chid1010 = get_chunk_id(i0, j1, l1, sx, sy, ncx, ncy)
    chid1011 = get_chunk_id(i1, j1, l1, sx, sy, ncx, ncy)

    # Memory offset into cache for each chunk
    slot0000 = indexof(ids, chid0000)
    slot0001 = indexof(ids, chid0001)
    slot0010 = indexof(ids, chid0010)
    slot0011 = indexof(ids, chid0011)
    slot1000 = indexof(ids, chid1000)
    slot1001 = indexof(ids, chid1001)
    slot1010 = indexof(ids, chid1010)
    slot1011 = indexof(ids, chid1011)

    # Return nan if any of the slots are not found
    if (slot0000 < 0 or slot0001 < 0 or slot0010 < 0 or slot0011 < 0 or
        slot1000 < 0 or slot1001 < 0 or slot1010 < 0 or slot1011 < 0):
        return np.nan

    # Within-chunk indices
    ii0 = i0 % sx
    ii1 = i1 % sx
    jj0 = j0 % sy
    jj1 = j1 % sy
    kk0 = k0 % sz
    kk1 = k1 % sz
    ll0 = l0 % st
    ll1 = l1 % st

    # Memory extraction
    u0000 = cache[slot0000, v, ll0, kk0, jj0, ii0]
    u0001 = cache[slot0001, v, ll0, kk0, jj0, ii1]
    u0010 = cache[slot0010, v, ll0, kk1, jj0, ii0]
    u0011 = cache[slot0011, v, ll0, kk1, jj0, ii1]
    u0100 = cache[slot0000, v, ll1, kk0, jj1, ii0]
    u0101 = cache[slot0001, v, ll1, kk0, jj1, ii1]
    u0110 = cache[slot0010, v, ll1, kk1, jj1, ii0]
    u0111 = cache[slot0011, v, ll1, kk1, jj1, ii1]
    u1000 = cache[slot1000, v, ll0, kk0, jj0, ii0]
    u1001 = cache[slot1001, v, ll0, kk0, jj0, ii1]
    u1010 = cache[slot1010, v, ll0, kk1, jj0, ii0]
    u1011 = cache[slot1011, v, ll0, kk1, jj0, ii1]
    u1100 = cache[slot1000, v, ll1, kk0, jj1, ii0]
    u1101 = cache[slot1001, v, ll1, kk0, jj1, ii1]
    u1110 = cache[slot1010, v, ll1, kk1, jj1, ii0]
    u1111 = cache[slot1011, v, ll1, kk1, jj1, ii1]

    # Interpolation weights
    # The weights are calculated as the distance from the lower bound
    p = x - i0
    q = y - j0
    r = z - k0
    s = t - l0

    w0000 = (1 - s) * (1 - r) * (1 - q) * (1 - p)
    w0001 = (1 - s) * (1 - r) * (1 - q) * p
    w0010 = (1 - s) * (1 - r) * q * (1 - p)
    w0011 = (1 - s) * (1 - r) * q * p
    w0100 = (1 - s) * r * (1 - q) * (1 - p)
    w0101 = (1 - s) * r * (1 - q) * p
    w0110 = (1 - s) * r * q * (1 - p)
    w0111 = (1 - s) * r * q * p
    w1000 = s * (1 - r) * (1 - q) * (1 - p)
    w1001 = s * (1 - r) * (1 - q) * p
    w1010 = s * (1 - r) * q * (1 - p)
    w1011 = s * (1 - r) * q * p
    w1100 = s * r * (1 - q) * (1 - p)
    w1101 = s * r * (1 - q) * p
    w1110 = s * r * q * (1 - p)
    w1111 = s * r * q * p    
    
    # Interpolation
    result = (w0000 * u0000 + w0001 * u0001 + w0010 * u0010 + w0011 * u0011 +
              w0100 * u0100 + w0101 * u0101 + w0110 * u0110 + w0111 * u0111 +
              w1000 * u1000 + w1001 * u1001 + w1010 * u1010 + w1011 * u1011 +
              w1100 * u1100 + w1101 * u1101 + w1110 * u1110 + w1111 * u1111)

    return result