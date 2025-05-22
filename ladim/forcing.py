import typing
if typing.TYPE_CHECKING:
    from ladim.model import Model
import numexpr
import string
import numpy as np


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
    def __init__(self, name):
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

        # Index block
        start = 32
        stop = start + 8*self.num_chunks
        self.chunk_id = np.ndarray(
            shape=(self.num_chunks, ),
            dtype=np.int64,
            buffer=mem.buf[start:stop])
        
        # Data block
        start = stop
        stop = start + self.num_chunks * self.chunksize * self.itemsize
        self.data = np.ndarray(
            shape=(self.num_chunks, self.chunksize),
            dtype=self.datatype.item().decode('ascii'),
            buffer=mem.buf[start:stop])

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError(f"Cannot reassign attribute '{name}'")
        super().__setattr__(name, value)

    @staticmethod
    def create(slots, chunksize, datatype='f4'):
        from multiprocessing.shared_memory import SharedMemory

        test_item = np.empty((), dtype=datatype)
        str_dtype = str(test_item.dtype)
        if len(str_dtype) > 8:
            raise ValueError('Unsupported data type: {str_dtype}')
        
        # Reserve memory space for the cache block
        size_header_block = 32
        size_index_block = 8 * slots
        size_data_block = slots * chunksize * test_item.itemsize
        bytes = size_header_block + size_index_block + size_data_block
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

        return ChunkCache(mem.name)

    
    def close(self):
        self.mem.close()


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