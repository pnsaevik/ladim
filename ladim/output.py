import netCDF4 as nc
import numpy as np
import typing
if typing.TYPE_CHECKING:
    from .model import Model


class Output:
    def __init__(self, variables: dict, file: str, frequency):
        """
        Writes simulation output to netCDF file in ragged array format

        :param variables: Simulation variables to include in output, and their formatting
        :param file: Name of output file, or empty if a diskless dataset is desired
        :param frequency: Output frequency in seconds. Alternatively, as a two-element
        tuple (freq_value, freq_unit) where freq_unit can be any numpy-compatible time
        unit.

        """
        # Convert output format specification from ladim.yaml config to OutputFormat
        self._formats = {
            k: OutputFormat.from_ladim_conf(v)
            for k, v in variables.items()
        }

        self._init_vars = {k for k, v in self._formats.items() if v.is_initial()}
        self._inst_vars = {k for k, v in self._formats.items() if v.is_instance()}

        if not file:
            from uuid import uuid4
            self._fname = uuid4()
            self._diskless = True
        else:
            self._fname = file
            self._diskless = False

        try:
            freq_num, freq_unit = frequency
        except TypeError:
            freq_num = frequency
            freq_unit = 's'
        self._write_frequency = np.timedelta64(freq_num, freq_unit).astype('timedelta64[s]').astype('int64')

        self._dset = None  # type: nc.Dataset | None
        self._num_writes = 0
        self._last_write_time = np.int64(-4611686018427387904)

    @property
    def dataset(self) -> nc.Dataset:
        """Returns a handle to the netCDF dataset currently being written to"""
        return self._dset

    def update(self, model: "Model"):
        if self._dset is None:
            self._create_dset()

        self._write_init_vars(model)
        self._write_instance_vars(model)

    def _write_init_vars(self, model):
        """
        Write the initial state of new particles
        """

        # Check if there are any new particles
        part_size = self._dset.dimensions['particle'].size
        num_new = model.state.released - part_size
        if num_new == 0:
            return

        # Write variable data
        idx = model.state['pid'] > part_size - 1
        pid = model.state['pid'][idx]
        for v in set(self._init_vars) - {'release_time'}:
            # The idx array is not necessarily monotonically increasing by 1
            # all the way. We therefore copy the data into a temporary,
            # continuous array.
            data_raw = model.state[v][idx]
            data = np.zeros(num_new, dtype=data_raw.dtype)
            data[pid - part_size] = data_raw
            self._dset.variables[v][part_size:part_size + num_new] = data

        # Write release time variable
        data = np.broadcast_to(model.solver.time, shape=(num_new, ))
        self._dset.variables['release_time'][part_size:part_size + num_new] = data
        self._dset.sync()

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

        # Write current time
        time_size = self._dset.dimensions['time'].size
        time_value = current_time.astype('datetime64[s]').astype('int64')
        self._dset.variables['time'][time_size] = time_value

        # Write variable values
        inst_size = self._dset.dimensions['particle_instance'].size
        inst_num = model.state.size
        inst_vars = {k: model.state[k] for k in set(self._inst_vars) - {'lat', 'lon'}}
        if {'lat', 'lon'}.intersection(self._inst_vars):
            x, y = model.state['X'], model.state['Y']
            inst_vars['lon'], inst_vars['lat'] = model.grid.xy2ll(x, y)
        for name, data in inst_vars.items():
            self._dset.variables[name][inst_size:inst_size + inst_num] = data

        # Write particle count
        self._dset.variables['particle_count'][time_size] = inst_num
        self._dset.sync()

    def _create_dset(self):
        default_formats = dict(
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

        self._dset = create_netcdf_file(
            fname=self._fname,
            formats={**default_formats, **self._formats},
            diskless=self._diskless,
        )

        self._dset.variables['instance_offset'][:] = 0
        self._dset.sync()

    def close(self):
        if self._dset is not None:
            self._dset.close()
            self._dset = None


class OutputFormat:
    def __init__(self, ncformat, dimensions, attributes, kind=None):
        self.ncformat = ncformat
        self.dimensions = dimensions
        self.attributes = attributes
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
    dset.createDimension(dimname="particle", size=None)
    dset.createDimension(dimname="particle_instance", size=None)
    dset.createDimension(dimname="time", size=None)

    # Create variables
    for varname, item in formats.items():
        dset.createVariable(
            varname=varname,
            datatype=item.ncformat,
            dimensions=item.dimensions,
        )
        dset.variables[varname].set_auto_mask(False)
        dset.variables[varname].setncatts(item.attributes)

    return dset
