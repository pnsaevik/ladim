from .model import Model, Module
import netCDF4 as nc
import numpy as np


class Output(Module):
    def __init__(self, model: Model):
        super().__init__(model)


class RaggedOutput(Output):
    def __init__(self, model: Model, **conf):
        super().__init__(model)

        self._fname = conf['output_file']
        self._init_vars = conf['output_particle']
        self._instance_vars = conf['output_instance']
        self._write_frequency = conf['output_period']

        # Convert output format specification from ladim.yaml config to OutputFormat
        self._formats = {
            k: OutputFormat.from_ladim_conf(k, self._init_vars, v)
            for k, v in conf['nc_attributes'].items()
        }

        self._dset = None
        self._num_writes = 0
        self._last_write_time = np.datetime64('NaT')

    def update(self):
        if self._dset is None:
            self._create_dset()

        self._write_init_vars()
        self._write_instance_vars()

    def _write_init_vars(self):
        """
        Write the initial state of new particles
        """

        # Check if there are any new particles
        part_size = self._dset.dimensions['particle'].size
        num_new = self.model.state['pid'].max() - part_size + 1
        if num_new == 0:
            return

        # Write variable data
        idx = self.model.state['pid'] > part_size - 1
        pid = self.model.state['pid'][idx]
        for v in set(self._init_vars) - {'release_time'}:
            # The idx array is not necessarily monotonically increasing by 1
            # all the way. We therefore copy the data into a temporary,
            # continuous array.
            data_raw = self.model.state[v][idx]
            data = np.zeros(num_new, dtype=data_raw.dtype)
            data[pid - part_size] = data_raw
            self._dset.variables[v][part_size:part_size + num_new] = data

        if 'release_time' in self._init_vars:
            data = np.broadcast_to(self.model.solver.time, shape=(num_new, ))
            v = 'release_time'
            self._dset.variables[v][part_size:part_size + num_new] = data

    def _write_instance_vars(self):
        """
        Write the current state of dynamic varaibles
        """

        # Check if this is a write time step
        current_time = self.model.solver.time
        write_frequency = self._write_frequency * self.model.solver.step
        elapsed_since_last_write = current_time - self._last_write_time
        if elapsed_since_last_write < write_frequency:
            return
        self._last_write_time = current_time

        # Write current time
        time_size = self._dset.dimensions['time'].size
        time_value = current_time.astype('datetime64[s]').astype('int64')
        self._dset.variables['time'][time_size] = time_value

        # Write variable values
        inst_size = self._dset.dimensions['particle_instance'].size
        inst_num = self.model.state.size
        for v in self._instance_vars:
            data = self.model.state[v]
            self._dset.variables[v][inst_size:inst_size + inst_num] = data

        # Write particle count
        self._dset.variables['particle_count'][time_size] = inst_num

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
            )
        )

        self._dset = create_netcdf_file(
            fname=self._fname,
            formats={**default_formats, **self._formats},
        )

        self._dset.variables['instance_offset'][:] = 0

    def close(self):
        if self._dset is not None:
            self._dset.close()
            self._dset = None


class OutputFormat:
    def __init__(self, ncformat, dimensions, attributes):
        self.ncformat = ncformat
        self.dimensions = dimensions
        self.attributes = attributes

    @staticmethod
    def from_ladim_conf(varname, init_vars, attrs):
        if varname in init_vars:
            dims = 'particle'
        else:
            dims = 'particle_instance'

        return OutputFormat(
            ncformat=attrs.get('ncformat', 'f4'),
            dimensions=dims,
            attributes={k: v for k, v in attrs.items() if k != 'ncformat'},
        )


def create_netcdf_file(fname: str, formats: dict[OutputFormat], diskless=False) -> nc.Dataset:
    """
    Create new netCDF file

    :param fname: File name
    :param formats: Formats, one entry for each variable
    :param diskless: True if a memory dataset should be generated
    :return: Empty, initialized dataset
    """
    from . import __version__ as ladim_version

    dset = nc.Dataset(filename=fname, mode='w', format='NETCDF4', diskless=diskless)

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
        dset.variables[varname].setncatts(item.attributes)

    return dset
