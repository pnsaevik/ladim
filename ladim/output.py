import netCDF4 as nc
import numpy as np
import typing
if typing.TYPE_CHECKING:
    from .model import Model
import os
import xarray as xr
import contextlib


class Output:
    def __init__(self, variables: dict, file: str, frequency, numrec: int=0):
        """
        Writes simulation output to netCDF file in ragged array format

        :param variables: Simulation variables to include in output, and their formatting
        :param file: Name of output file, or empty if a diskless dataset is desired
        :param frequency: Output frequency in seconds. Alternatively, as a two-element
            tuple (freq_value, freq_unit) where freq_unit can be any numpy-compatible time
            unit.
        :param numrec: Number of records per output file. Zero means a single output file.

        """
        # Convert output format specification from ladim.yaml config to OutputFormat
        user_formats = {
            k: OutputFormat.from_ladim_conf(v)
            for k, v in variables.items()
        }
        formats = {**self._default_formats(), **user_formats}

        if numrec == 0:
            self.writer = Writer.netcdf(file, formats)
        else:
            self.writer = Writer.mf_netcdf(file, formats, numrec)

        self._init_vars = {k for k, v in formats.items() if v.is_initial()}
        self._inst_vars = {k for k, v in formats.items() if v.is_instance()}

        try:
            freq_num, freq_unit = frequency
        except TypeError:
            freq_num = frequency
            freq_unit = 's'
        self._write_frequency = np.timedelta64(freq_num, freq_unit).astype('timedelta64[s]').astype('int64')

        self._num_writes = 0
        self._last_write_time = np.int64(-4611686018427387904)

    @staticmethod
    def create(variables: dict, file: str, frequency, numrec: int=0):
        """
        Writes simulation output to netCDF file in ragged array format

        :param variables: Simulation variables to include in output, and their formatting
        :param file: Name of output file, or empty if a diskless dataset is desired
        :param frequency: Output frequency in seconds. Alternatively, as a two-element
        tuple (freq_value, freq_unit) where freq_unit can be any numpy-compatible time
        unit.
        :param numrec: Number of records per output file. Zero means a single output file.

        """
        return Output(variables, file, frequency, numrec)

    def update(self, model: "Model"):
        self._write_init_vars(model)
        self._write_instance_vars(model)

    def _write_init_vars(self, model):
        """
        Write the initial state of new particles
        """

        # Check if there are any new particles
        part_size = self.writer.sizes['particle']
        num_new = model.state.released - part_size
        if num_new == 0:
            return

        # Extract data
        idx = model.state['pid'] > part_size - 1
        pid = model.state['pid'][idx]
        data_dict = {}
        for v in set(self._init_vars) - {'release_time'}:
            # The idx array is not necessarily monotonically increasing by 1
            # all the way. We therefore copy the data into a temporary,
            # continuous array.
            data_raw = model.state[v][idx]
            data = np.zeros(num_new, dtype=data_raw.dtype)
            data[pid - part_size] = data_raw
            data_dict[v] = data
        data_dict['release_time'] = np.broadcast_to(model.solver.time, shape=(num_new, ))

        self.writer.write(data_dict)

    def _write_instance_vars(self, model):
        """
        Write the current state of dynamic varaibles
        """

        # Check if this is a write time step
        current_time = model.solver.time
        elapsed_since_last_write = current_time - self._last_write_time
        if elapsed_since_last_write < self._write_frequency:
            return
        self._last_write_time = current_time

        # Get variable values
        data_dict = {k: model.state[k] for k in set(self._inst_vars) - {'lat', 'lon'}}
        data_dict['time'] = current_time.astype('datetime64[s]').astype('int64').ravel()
        data_dict['particle_count'] = np.asarray(model.state.size).ravel()
        if {'lat', 'lon'}.intersection(self._inst_vars):
            x, y = model.state['X'], model.state['Y']
            data_dict['lon'], data_dict['lat'] = model.grid.xy2ll(x, y)

        self.writer.write(data_dict)

    @staticmethod
    def _default_formats() -> dict[str, "OutputFormat"]:
        return dict(
            time=OutputFormat(
                ncformat='i8',
                dimensions='time',
                attributes=dict(
                    long_name="time",
                    standard_name="time",
                    units="seconds since 1970-01-01",
                ),
            ),
            instance_offset=OutputFormat(
                ncformat='i8',
                dimensions=(),
                attributes=dict(long_name='particle instance offset for file'),
            ),
            particle_count=OutputFormat(
                ncformat='i4',
                dimensions='time',
                attributes=dict(
                    long_name='number of particles in a given timestep',
                    ragged_row_count='particle count at nth timestep',
                ),
            ),
            release_time=OutputFormat(
                ncformat='i8',
                dimensions='particle',
                attributes=dict(
                    long_name='particle release time',
                    units='seconds since 1970-01-01',
                )
            )
        )

    def close(self):
        self.writer.close()


class OutputFormat:
    def __init__(self, ncformat: str, dimensions: str, attributes: dict = None,
                 kind: str = None):
        self.ncformat = ncformat
        self.dimensions = dimensions
        self.attributes = attributes or {}
        self.kind = kind

    def is_initial(self):
        return self.kind == 'initial'

    def is_instance(self):
        return self.kind == 'instance'

    @staticmethod
    def from_ladim_conf(conf) -> "OutputFormat":
        def get_keywords(ncformat='f4', kind='instance', **kwargs):
            return dict(
                props=dict(ncformat=ncformat, kind=kind),
                attrs=kwargs,
            )

        keywords = get_keywords(**conf)
        vkind = keywords['props']['kind']
        if vkind == 'initial':
            dims = 'particle'
        elif vkind == 'instance':
            dims = 'particle_instance'
        else:
            raise ValueError(f"Unknown kind: {vkind}")

        return OutputFormat(
            ncformat=keywords['props']['ncformat'],
            dimensions=dims,
            attributes=keywords['attrs'],
            kind=vkind,
        )


def create_netcdf_file(fname: str, formats: dict[str, OutputFormat], diskless=False) -> nc.Dataset:
    """
    Create new netCDF file

    :param fname: File name
    :param formats: Formats, one entry for each variable
    :param diskless: True if a memory dataset should be generated
    :return: Empty, initialized dataset
    """
    from . import __version__ as ladim_version

    dset = nc.Dataset(filename=fname, mode='w', format='NETCDF4', diskless=diskless)
    dset.set_auto_mask(False)

    # Create attributes
    dset.Conventions = "CF-1.8"
    dset.institution = "Institute of Marine Research"
    dset.source = "Lagrangian Advection and Diffusion Model"
    dset.history = "Created by ladim " + ladim_version
    dset.date = str(np.datetime64('now', 'D'))

    # Create dimensions
    dimnames = {f.dimensions for f in formats.values() if f.dimensions}
    for dimname in dimnames:
        dset.createDimension(dimname=dimname, size=None)

    # Create variables
    for varname, item in formats.items():
        dset.createVariable(
            varname=varname,
            datatype=item.ncformat,
            dimensions=item.dimensions or (),
        )
        dset.variables[varname].set_auto_mask(False)
        dset.variables[varname].setncatts(item.attributes)

    if 'instance_offset' in dset.variables:
        dset.variables['instance_offset'][...] = 0

    return dset


class Writer:
    """
    Abstract base class for output writers

    An output writer should be able to write tabular data in a thread-safe
    way to the output storage backend. The output may be in the form of one or
    more tables, distributed over one or more files. Each write operation should
    be atomic, i.e. all data is written sequentially to the file at once. The
    total number of rows in each table is not known at creation time.

    Each table may have a number of columns, each with a unique name, data type
    and potentially some metadata attributes. The number of columns and their
    attributes should be known at creation time.
    """
    @staticmethod
    def netcdf(file: str, formats: dict[str, OutputFormat]) -> "Writer":
        """
        Create a single-file netCDF writer

        :param file: File name, or empty string if in-memory object is desired
        :param formats: Formats, one entry for each variable
        :return: Writer instance
        """
        return _NCWriter(file, formats)
    
    @staticmethod
    def mf_netcdf(file: str, formats: dict[str, OutputFormat], numrec) -> "Writer":
        """
        Create a multi-file netCDF writer
        """

        return _MFNCWriter(file, formats, numrec)

    def write(self, data: dict[str, np.ndarray]):
        """
        Write data to file(s)

        :param data: Dictionary with variable names as keys and numpy arrays as
            data values
        """
        raise NotImplementedError()
    
    @property
    def sizes(self) -> dict[str, int]:
        """
        Return number of rows written in each table

        :return: Dictionary with variable names as keys and number of rows as values
        """
        raise NotImplementedError()

    @property
    def paths(self) -> list[typing.Any]:
        """
        Return list of paths to written files, or in-memory objects
        
        :return: List of file paths or in-memory objects
        """
        raise NotImplementedError()

    def close(self):
        """
        Close all open files and release resources
        """
        raise NotImplementedError()
        

class _NCWriter(Writer):
    def __init__(self, file: str, formats: dict[str, OutputFormat]):
        """
        Create a single-file netCDF writer

        :param file: File name, or empty string if in-memory object is desired
        :param formats: Formats, one entry for each variable
        """

        if not file:
            from uuid import uuid4
            file = str(uuid4())
            diskless = True
        else:
            diskless = False
            file = str(file)

        dset = create_netcdf_file(fname=file, formats=formats, diskless=diskless)
        dset.sync()
        self._sizes = {k: v.size for k, v in dset.dimensions.items()}

        if diskless:
            self._paths = [dset]
        else:
            self._paths = [file]
            dset.close()

    @property
    def sizes(self) -> dict[str, int]:
        return self._sizes
    
    @property
    def paths(self) -> list[typing.Any]:
        return self._paths

    def write(self, data: dict[str, np.ndarray]):
        if isinstance(self._paths[0], str):
            with nc.Dataset(self._paths[0], mode='a') as dset:
                self._write(dset, data)
                self._sizes = {k: v.size for k, v in dset.dimensions.items()}
        else:
            dset = self._paths[0]
            self._write(dset, data)
            self._sizes = {k: v.size for k, v in dset.dimensions.items()}

    @staticmethod
    def _write(dset: nc.Dataset, data: dict[str, np.ndarray]):
        old_sizes = {k: v.size for k, v in dset.dimensions.items()}

        for k, v in data.items():
            dimname, = dset.variables[k].dimensions  # Assume single dimension
            sz = old_sizes[dimname]
            dset.variables[k][sz:sz + len(v)] = v
        dset.sync()

    def close(self):
        # Have already closed files after each write
        pass


class _MFNCWriter(Writer):
    def __init__(self, file: str, formats: dict[str, OutputFormat], numrec: int):
        """
        Create a multi-file netCDF writer

        :param file: File name, or empty string if in-memory object is desired
        :param formats: Formats, one entry for each variable
        :param numrec: Number of records per output file. Zero means a single output file.
        """

        self.file = file
        self.formats = formats
        self.numrec = numrec
        self._paths = []
        self._step_counter = 0
        self._padding = 4

        self._append_next_file()

    def _append_next_file(self):
        diskless = False
        if not self.file:
            from uuid import uuid4
            file = str(uuid4())
            diskless = True
        else:
            base_name, ext = os.path.splitext(str(self.file))
            file = f"{base_name}_{len(self._paths):0{self._padding}d}{ext}"

        old_release_time = None
        if len(self._paths) > 0:
            if not diskless:
                with nc.Dataset(self._paths[-1], mode='r') as dset:
                    if 'release_time' in dset.variables:
                        old_release_time = dset.variables['release_time'][:]
            else:
                dset = self._paths[-1]
                if 'release_time' in dset.variables:
                    old_release_time = dset.variables['release_time'][:]

        dset = create_netcdf_file(fname=file, formats=self.formats, diskless=diskless)
        if old_release_time is not None:
            dset.variables['release_time'][:] = old_release_time
        dset.sync()
        self._sizes = {k: v.size for k, v in dset.dimensions.items()}

        if diskless:
            self._paths.append(dset)
        else:
            self._paths.append(file)
            dset.close()

    @property
    def sizes(self) -> dict[str, int]:
        return self._sizes

    @property
    def paths(self) -> list[typing.Any]:
        return self._paths

    def write(self, data: dict[str, np.ndarray]):
        self._step_counter += 1
        particle_instance = 0
        if (self._step_counter > 1) and not((self._step_counter - 1) % (self.numrec * 2)):
            particle_instance = self._get_instance_offset()
            self._append_next_file()

        if isinstance(self._paths[-1], str):
            with nc.Dataset(self._paths[-1], mode='a') as dset:
                self._write(dset, data, particle_instance)
                self._sizes = {k: v.size for k, v in dset.dimensions.items()}
        else:
            dset = self._paths[-1]
            self._write(dset, data, particle_instance)
            self._sizes = {k: v.size for k, v in dset.dimensions.items()}

    def _get_instance_offset(self):
        old_instance_offset = 0
        if isinstance(self._paths[-1], str):
            with nc.Dataset(self._paths[-1], mode='r') as dset:
                if 'instance_offset' in dset.variables:
                    old_instance_offset = dset.variables['instance_offset'][...]
        else:
            dset = self._paths[-1]
            if 'instance_offset' in dset.variables:
                old_instance_offset = dset.variables['instance_offset'][...]

        if 'particle_instance' in self._sizes:
            return self._sizes.get('particle_instance') + old_instance_offset
        else:
            return 0

    @staticmethod
    def _write(dset: nc.Dataset, data: dict[str, np.ndarray], particle_instance: int):
        old_sizes = {k: v.size for k, v in dset.dimensions.items()}
        for k, v in data.items():
            dimname, = dset.variables[k].dimensions  # Assume single dimension
            sz = old_sizes[dimname]
            dset.variables[k][sz:sz + len(v)] = v
        if 'instance_offset' in dset.variables and (particle_instance > 0):
            dset.variables['instance_offset'][...] = particle_instance
        dset.sync()

    def close(self):
        # Have already closed files after each write
        pass


@contextlib.contextmanager
def _open_or_relay(path_or_object: str | nc.Dataset, mode='r'):
    if isinstance(path_or_object, str):
        with nc.Dataset(path_or_object, mode=mode) as dset:
            yield dset
    else:
        yield path_or_object
