import netCDF4 as nc
import numpy as np
import typing
if typing.TYPE_CHECKING:
    from .model import Model
import os
import contextlib
import abc


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
            copy_tabs = ('particle', )
            offset_vars = {'particle_instance': 'instance_offset'}
            self.writer = Writer.mf_netcdf(file, formats, numrec, offset_vars, copy_tabs)

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
        # Check if this is a write step for instance (dynamic) variables
        data_dict_inst = self._update_instance_vars(model)
        self.writer.write_instance(data_dict_inst)

        # Check if there are any new particles to write
        # Do this after write_instance in case there is a file name rotation
        data_dict_init = self._update_init_vars(model)
        self.writer.write_init(data_dict_init)

    def _update_init_vars(self, model) -> dict[str, np.ndarray]:
        """
        Build output arrays for newly released particles only.

        This method inspects how many particle-table rows have already been
        written to the output and compares that with the cumulative number of
        released particles in the model state. Any particles that have been
        released but not yet written are treated as "new" and their initial
        variables are returned as contiguous arrays ordered by pid offset.

        Returns an empty dict when there are no new particles on this solver
        step.
        """

        # Check if there are any new particles
        part_size = self.writer.sizes['particle']
        num_new = model.state.released - part_size
        if num_new == 0:
            return dict()

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

        return data_dict

    def _update_instance_vars(self, model) -> dict[str, np.ndarray]:
        """
        Build one time-record payload for dynamic (instance) variables.

        Instance variables are written at the configured output frequency,
        not on every solver step. If the current solver time is not a write
        time, this returns an empty dict. Otherwise it returns the per-particle
        instance arrays plus the scalar time metadata fields (`time` and
        `particle_count`) that define the ragged record.
        """

        # Check if this is a write time step
        current_time = model.solver.time
        elapsed_since_last_write = current_time - self._last_write_time
        if elapsed_since_last_write < self._write_frequency:
            return dict()
        self._last_write_time = current_time

        # Get variable values
        data_dict = {k: model.state[k] for k in set(self._inst_vars) - {'lat', 'lon'}}
        data_dict['time'] = current_time.astype('datetime64[s]').astype('int64').ravel()
        data_dict['particle_count'] = np.asarray(model.state.size).ravel()
        if {'lat', 'lon'}.intersection(self._inst_vars):
            x, y = model.state['X'], model.state['Y']
            data_dict['lon'], data_dict['lat'] = model.grid.xy2ll(x, y)

        return data_dict

    @staticmethod
    def _default_formats() -> dict[str, "OutputFormat"]:
        return dict(
            time=OutputFormat(
                ncformat='i8',
                dimensions='time',
                attributes=dict(
                    long_name="time",
                    standard_name="time",
                    units="seconds since 1970-01-01 00:00:00",
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
                    units='seconds since 1970-01-01 00:00:00',
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

    An output writer stores tabular particle data in one or more files.
    Callers write initial (static) and instance (dynamic) data through
    separate methods. Only ``write_instance`` advances the time-record
    counter used for multi-file splitting (``numrec``).

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
    def mf_netcdf(
        file: str,
        formats: dict[str, OutputFormat],
        numrec,
        offset_variables: dict | None = None,
        copy_dims: tuple[str] = (),
        ) -> "Writer":
        """
        Create a multi-file netCDF writer

        :param file: File name, or empty string if in-memory object is desired
        :param formats: Formats, one entry for each variable
        :param numrec: Number of records per output file. Zero means a single output file.
        :param offset_variables: A mapping from dimension names to offset variables.
            An offset variable is a variable inside a multi-file dataset that
            indicates how many previous records have already been written to
            prior files. The format of the offset variable must be specified
            in the ``formats`` param.
        :param copy_dims: Tables that should be copied from the previous 
            file to the next one, when a new file is created.
        """

        return _MFNCWriter(file, formats, numrec, offset_variables, copy_dims)

    @abc.abstractmethod
    def write_init(self, data: dict[str, np.ndarray]):
        """
        Append initial (static/particle-table) variables.

        Does not count toward ``numrec`` file splitting. Empty ``data`` is a
        no-op.

        :param data: Dictionary with variable names as keys and numpy arrays as
            data values
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def write_instance(self, data: dict[str, np.ndarray]):
        """
        Append one time record of instance (dynamic) variables.

        Each non-empty call counts as one record for ``numrec`` file splitting.
        Empty ``data`` is a no-op.

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
    def offsets(self) -> dict[str, int]:
        """
        Return accumulated number of rows written in each table

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
    def offsets(self) -> dict[str, int]:
        return self._sizes

    @property
    def paths(self) -> list[typing.Any]:
        return self._paths

    def write_init(self, data: dict[str, np.ndarray]):
        if not data:
            return
        with _open_or_relay(self._paths[0], mode='a') as dset:
            self._write(dset, data)
            self._sizes = {k: v.size for k, v in dset.dimensions.items()}

    def write_instance(self, data: dict[str, np.ndarray]):
        if not data:
            return
        with _open_or_relay(self._paths[0], mode='a') as dset:
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
    def __init__(
            self,
            file: str,
            formats: dict[str, OutputFormat],
            numrec: int,
            offset_variables: dict | None = None,
            copy_dims: tuple[str] = (),
            ):
        """
        Create a multi-file netCDF writer

        :param file: File name, or empty string if in-memory object is desired
        :param formats: Formats, one entry for each variable
        :param numrec: Number of records per output file. Zero means a single output file.
        :param offset_variables: A mapping from dimension names to offset variables.
            An offset variable is a variable inside a multi-file dataset that
            indicates how many previous records have already been written to
            prior files. The format of the offset variable must be specified
            in the ``formats`` param.
        :param copy_dims: Tables that should be copied from the previous 
            file to the next one, when a new file is created.
        """

        offset_variables = offset_variables or dict()  # Empty dict if None

        self.file = file
        self.formats = formats
        self.numrec = numrec
        self._sizes = {fmt.dimensions: 0 for fmt in formats.values() if fmt.dimensions}
        self._offsets = self.sizes.copy()
        self._paths = []
        self._step_counter = 0
        self._padding = 4
        self._tables_to_be_copied = copy_dims
        self._offset_variables = offset_variables

    def _initialize_next_file(self):
        if not self.file:
            from uuid import uuid4
            file_name = str(uuid4())
            diskless = True
        else:
            base_name, ext = os.path.splitext(str(self.file))
            file_name = f"{base_name}_{len(self._paths):0{self._padding}d}{ext}"
            diskless = False

        dset = create_netcdf_file(fname=file_name, formats=self.formats, diskless=diskless)

        if len(self._paths) > 0:
            with _open_or_relay(self._paths[-1]) as source_dataset:
                _copy_nc_tables(source_dataset, dset, self._tables_to_be_copied)
            self._offsets = self._sizes.copy()
            for k in self._tables_to_be_copied:
                self._offsets[k] = 0

        dset.sync()

        if diskless:
            self._paths.append(dset)
        else:
            self._paths.append(file_name)
            dset.close()

    @property
    def sizes(self) -> dict[str, int]:
        return self._sizes

    @property
    def offsets(self) -> dict[str, int]:
        return self._offsets

    @property
    def paths(self) -> list[typing.Any]:
        return self._paths

    def write_init(self, data: dict[str, np.ndarray]):
        if not data:
            return
        if not self._paths:
            self._initialize_next_file()

        with _open_or_relay(self._paths[-1], mode='a') as dset:
            self._write(dset, data)

    def write_instance(self, data: dict[str, np.ndarray]):
        if not data:
            return

        if not self._paths:
            self._initialize_next_file()
        elif self._step_counter > 0 and self._step_counter % self.numrec == 0:
            self._initialize_next_file()

        with _open_or_relay(self._paths[-1], mode='a') as dset:
            self._write(dset, data)

        self._step_counter += 1

    def _write(self, dset: nc.Dataset, data: dict[str, np.ndarray]):
        old_sizes = {k: v.size for k, v in dset.dimensions.items()}
        for k, v in data.items():
            dimname, = dset.variables[k].dimensions  # Assume single dimension
            sz = old_sizes[dimname]
            dset.variables[k][sz:sz + len(v)] = v

        for dimname, varname in self._offset_variables.items():
            if varname in dset.variables and dimname in self._offsets:
                dset.variables[varname][...] = self._offsets[dimname]

        self._sizes = {k: v.size + self._offsets.get(k, 0) for k, v in dset.dimensions.items()}

        dset.sync()

    def close(self):
        # Have already closed files after each write
        pass


@contextlib.contextmanager
def _open_or_relay(path_or_object: str | nc.Dataset, mode='r') -> typing.Generator[nc.Dataset, typing.Any, typing.Any]:
    if isinstance(path_or_object, str):
        with nc.Dataset(path_or_object, mode=mode) as dset:
            yield dset
    else:
        yield path_or_object


def _copy_nc_tables(src_dset: nc.Dataset, dst_dset: nc.Dataset, dims: tuple[str]):
    """
    Copy tables from one netCDF dataset to another

    A "table" is a set of single-dimension netCDF variables sharing the same 
    dimension.

    :param src_dset: Source dataset
    :param dst_dset: Destination dataset
    :param dims: Dimensions to copy
    """
    for k, v in src_dset.variables.items():
        if len(v.dimensions) != 1:
            continue
        if v.dimensions[0] not in dims:
            continue
        dst_dset[k][:] = src_dset[k][:]
