"""
Output data write routines
"""

import re
import netCDF4 as nc
import numpy as np
import typing
if typing.TYPE_CHECKING:
    from .model import Model
import os
import contextlib
import queue
import threading
from pathlib import Path
import xarray as xr


def store(
        state: dict[str, np.ndarray],
        time: int,
        dt: int,
        freq: int,
        target: str,
        encoding: dict[str, dict] | None = None,
        engine: str = 'netcdf4'
        ):
    """
    Main entry point for storing particle state

    The function stores state and time to the desired target using the
    specified engine and encoding.

    :param state: A mapping from variable name to data values. All data arrays
        should have the same number of elements.

    :param time: Simulation time, in number of seconds since unix
        epoch (1970-01-01).

    :param dt: Length of time step, in seconds. 

    :param freq: Output frequency, in seconds. The function will only write
        time-dependent variables to file if the time interval [time, time + dt)
        encloses an output time.

    :param target: URL or file name of desired target, possibly date-encoded.
        The following tokens are allowed in the target:
        - %Y Year (4-digit)
        - %m Month (2-digit)
        - %d Day of month (2-digit)
        - %H Hour (2-digit, 24-hour)
        - %M Minute (2-digit)
        - %S Second (2-digit)
        - %j Day of year (3-digit)

        Example: 'target_output_file_%Y%m%d.nc'

        If there are time-independent variables in the output, these will be
        stored in a separate file. The name of this file is generated from the
        target string, but with all date tokens removed.

    :param encoding: A mapping from parameter names to engine-specific
        encodings, as described below:

        **netcdf4:** Valid keywords are
        - ``ncformat`` for data type
        - ``kind`` should be 'instance' (default) if variable should be considered
          time-independent or 'initial' if variable is time-independent
        - Other keywords are considered attributes

    :param engine: The output engine to use. At present, the only valid engine
        is 'netcdf4'.
    """

    # Delegate function call to correct engine

    store_functions = {
        'netcdf4': _store_netcdf
    }
    store_fn = store_functions[engine]
    store_fn(state, time, dt, freq, encoding, target)


def _store_netcdf(
        state: dict[str, np.ndarray],
        time: int,
        dt: int,
        freq: int,
        encoding: dict[str, dict],
        target: str,
        ):
    
    formats = {k: OutputFormat.from_ladim_conf(v) for k, v in encoding.items()}
    formats |= _default_netcdf_formats()
    _store_netcdf_timeconstant(state, time, dt, formats, target)
    _store_netcdf_timedependent(state, time, dt, freq, formats, target)


def _store_netcdf_timeconstant(
        state: dict[str, np.ndarray],
        time: int,
        dt: int,
        formats: dict[str, "OutputFormat"],
        target: str,
        ):
        
    # Extract data
    seconds_since_release = np.int64(time) - state['release_time']
    idx_to_be_stored = (0 <= seconds_since_release) & (seconds_since_release < dt)
    pid = state['pid'][idx_to_be_stored]
    if len(pid) == 0:
        return

    # Open file
    fname = resolve_datestring(target, None)
    with _open_or_create_netcdf(fname) as fp:

        # Ensure dimensions
        if not 'particle' in fp.dimensions:
            fp.createDimension('particle')
        
        # Create or append variables
        for varname, fmt in formats.items():
            # Skip if time-dependent variable
            if not fmt.dimensions == 'particle':
                continue
            
            # Ensure variable
            if varname not in fp.variables:
                fp.createVariable(varname, fmt.ncformat, fmt.dimensions or ())
                fp.variables[varname].set_auto_mask(False)
                fp.variables[varname].setncatts(fmt.attributes)

            # Store variable data
            fp.variables[varname][pid] = state[varname][idx_to_be_stored]


def _store_netcdf_timedependent(
        state: dict[str, np.ndarray],
        time: int,
        dt: int,
        freq: int,
        formats: dict[str, "OutputFormat"],
        target: str,
        ):
    
    # Check if we are at an output time
    is_output_time = np.ceil(time / freq) < time + dt
    if not is_output_time:
        return

    # Count number of particles
    num_particles_allvars = [len(v) for v in state.values()]
    num_particles = num_particles_allvars[0]
    assert all(num_particles == v for v in num_particles_allvars)

    # Open file
    time64 = np.asarray(time).astype('datetime64[s]')
    fname = resolve_datestring(target, time64)
    with _open_or_create_netcdf(fname) as fp:
        # Ensure dimension
        if not 'particle_instance' in fp.dimensions:
            fp.createDimension('particle_instance')
        old_size = fp.dimensions['particle_instance'].size
        new_size = old_size + num_particles

        # Create or append state variables
        for varname, fmt in formats.items():
            # Skip if time-constant variable
            if fmt.dimensions != 'particle_instance' or varname not in state:
                continue
            
            # Ensure variable
            if varname not in fp.variables:
                fp.createVariable(varname, fmt.ncformat, fmt.dimensions)
                fp.variables[varname].set_auto_mask(False)
                fp.variables[varname].setncatts(fmt.attributes)

            # Append data
            fp.variables[varname][old_size:new_size] = state[varname]
        
        # Ensure time dimension
        if not 'time' in fp.dimensions:
            fp.createDimension('time')
        time_idx = fp.dimensions['time'].size

        # Create or append time variables
        for varname in ['time', 'particle_count', 'instance_offset']:
            if varname not in fp.variables:
                fmt = formats[varname]
                fp.createVariable(varname, fmt.ncformat, fmt.dimensions or ())
                fp.variables[varname].set_auto_mask(False)
                fp.variables[varname].setncatts(fmt.attributes)
                if not fmt.dimensions:
                    fp.variables[varname][...] = 0
        
        fp.variables['time'][time_idx] = time
        fp.variables['particle_count'][time_idx] = num_particles


def _open_or_create_netcdf(fname):
    if not Path(fname).exists():
        Path(fname).parent.mkdir(exist_ok=True, parents=True)
        fp = nc.Dataset(fname, 'w')
        fp.set_auto_mask(False)
        _initialize_netcdf(fp)
        return fp

    else:
        fp = nc.Dataset(fname, 'a')
        fp.set_auto_mask(False)
        return fp


def _initialize_netcdf(fp: nc.Dataset):

    fp.set_auto_mask(False)

    # Create root-level attributes
    if not fp.ncattrs():
        from ladim import __version__ as ladim_version
        fp.setncatts({
            "Conventions": "CF-1.8",
            "institution": "Institute of Marine Research",
            "source": "Lagrangian Advection and Diffusion Model",
            "history": "Created by ladim " + ladim_version,
            "date": str(np.datetime64('now', 'D')),
        })
        try:
            from ladim_plugins import __version__ as ladim_plugins_version  # type: ignore
            fp.setncattr('ladim_plugins_version', ladim_plugins_version)
        except ImportError:
            pass

    return fp


def resolve_datestring(s, t):
    """
    Replace date format tokens with formatted times

    The following tokens are allowed in the target:
    - %Y Year (4-digit)
    - %m Month (2-digit)
    - %d Day of month (2-digit)
    - %H Hour (2-digit, 24-hour)
    - %M Minute (2-digit)
    - %S Second (2-digit)
    - %j Day of year (3-digit)
    - %% Literal '%'
    
    :param s: String with date tokens
    :param t: Numpy datetime or None (if all tokens should be replaced
        with empty string)
    :returns: Formatted string
    """
    tokens = {
    "%Y": lambda d: f"{d.year:04d}",
    "%m": lambda d: f"{d.month:02d}",
    "%d": lambda d: f"{d.day:02d}",
    "%H": lambda d: f"{d.hour:02d}",
    "%M": lambda d: f"{d.minute:02d}",
    "%S": lambda d: f"{d.second:02d}",
    "%j": lambda d: f"{d.timetuple().tm_yday:03d}",  # day of year
}
    placeholder = "\0"
    result = s.replace("%%", placeholder)

    if t is None:
        for token in tokens.keys():
            result = result.replace(token, '')

    else:
        py_t = np.datetime64(t).astype(object)
        for token, fn in tokens.items():
            result = result.replace(token, fn(py_t))

    return result.replace(placeholder, "%")    


def _default_netcdf_formats() -> dict[str, "OutputFormat"]:
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
        ),
        instance_offset=OutputFormat(
            ncformat='i8',
            dimensions='',
            attributes=dict(
                long_name="particle instance offset for file",
            ),
        )
    )


def append_netcdf(data: xr.Dataset, fp: nc.Dataset):
    """
    Append data to netcdf file
    
    :param data: Dataset to write
    :param fp: Handle to output dataset
    """
    fp.set_auto_mask(False)

    # Create root-level attributes
    if not fp.ncattrs():
        fp.setncatts(data.attrs)

    # Create dimensions
    for dimname in data.dims:
        if dimname not in fp.dimensions:
            fp.createDimension(dimname=dimname, size=None)

    # Create variables
    for varname, item in data.variables.items():
        if str(varname) not in fp.variables:
            fp.createVariable(
                varname=str(varname),
                datatype=item.dtype,
                dimensions=tuple(str(d) for d in item.dims),
            )

            fp.variables[varname].set_auto_mask(False)
            fp.variables[varname].setncatts(item.attrs)

    # Store old dimension sizes
    old_dims = {k: v.size for k, v in fp.dimensions.items()}

    # Append data
    for varname, item in data.variables.items():
        idx = tuple(
            slice(old_dims[k], old_dims[k] + sz)
            for k, sz in zip(item.dims, item.shape)
        )
        fp.variables[varname][idx] = item.to_numpy()

    return fp
        

def _to_seconds(spec):
    try:
        num, unit = spec
    except TypeError:
        num = spec
        unit = 's'
    return np.timedelta64(num, unit).astype('timedelta64[s]').astype('int64')


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
        self._variables = variables
        self._file = file
        self._frequency = _to_seconds(frequency)
        self._numrec = numrec

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
        store(
            state=model.state.values,
            time=model.solver.time,
            dt=model.solver.step,
            freq=self._frequency,
            target=self._file,
            encoding=self._variables,
            engine='netcdf4',
        )

    def close(self):
        pass


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
