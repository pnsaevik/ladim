import numpy as np
import pandas as pd
from .utilities import read_timedelta
import logging
import typing
import netCDF4 as nc

if typing.TYPE_CHECKING:
    from ladim.model import Model

logger = logging.getLogger(__name__)

CoordTransform = typing.Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
ParticleGenerator = typing.Callable[[float, float], pd.DataFrame]


class Releaser:
    def __init__(self, schedule: pd.DataFrame):
        """
        Create a releaser from a schedule

        A schedule is a table of release times and release intervals, together
        with initial properties of released particles. Each row represents a
        particle source. It is assumed that the table is sorted by
        `release_start`, and that `release_stop` of each line never exceeds the
        release_start value of the next line.

        Some columns have special meanings:

        release_start
            First release time of this particle type, in seconds since the
            posix epoch (1970-01-01).

        release_stop
            Stop time (not inclusive) for this particle source, in seconds
            since the posix epoch (1970-01-01).
        
        release_step
            Number of seconds between releases. Set this to a large number if
            the release should not be repeated.

        X, Y, Z
            Initial particle X, Y, Z position, using internal model coordinates

        :param schedule: A table of release times, release intervals and initial
            properties of released particles.
        """
        self._schedule = schedule

    @staticmethod
    def create(
            file=None, colnames: list = None, formats: dict = None,
            frequency=(0, 's'), defaults=None, lonlat_converter=None,
            warm_start_file: str | nc.Dataset | None = None
    ):
        """
        Create release module from whitespace-separated csv file

        :param lonlat_converter: Function that converts lon, lat coordinates to
            x, y coordinates

        :param file: Either a release file (path name or stream), or an
            in-memory table (object that is convertible to pandas data frame)

        :param colnames: Column names, if the release file does not contain any

        :param formats: Data column formats, one dict entry per column. If any column
            is missing, the default format is used. Keys should correspond to column names.
            Values should be either ``"float"``, ``"int"`` or ``"time"``. Default value
            is ``"float"`` for all columns except ``release_time``, which has default
            value ``"time"``.

        :param frequency: A two-element list with entries ``[value, unit]``, where
            ``unit`` can be any numpy-compatible timedelta unit (such as "s", "m", "h", "D").

        :param defaults: A dict of variables to be added to each particle. The keys
            are the variable names, the values are the initial values at particle
            release.
        
        :param warm_start_file: Ladim output file containing start values
            for the first time step
        """
        # Handle empty arguments
        defaults = defaults or {}
        formats = formats or {}
        lonlat_converter = lonlat_converter or (lambda lon, lat: (lon, lat))

        # Add standard defaults
        standard_defaults = {
            'release_interval': np.asarray(read_timedelta(frequency) / np.timedelta64(1, 's'), dtype='int64'),
            'release_time': np.array(0, dtype='datetime64[s]'),
        }
        defaults = {**standard_defaults, **defaults}

        # Create releaser from release table
        df = load_table(file, names=colnames, formats=formats)
        df = add_default_variables_in_release_table(df, defaults)
        df = add_start_stop_step_to_release_table(df)
        df = apply_warm_start_file(df, file=warm_start_file)
        df = replace_lonlat_in_release_table(df, lonlat_converter)
        df = df.sort_values('release_time')
        releaser = Releaser(df)

        return releaser

    def new_particles(self, start: float, stop: float) -> pd.DataFrame:
        """
        Initial properties of particles released within a given period

        :param start: Start time of period
        :param stop: Stop time of period (not inclusive)
        :returns: Initial properties of particles released within period
        """
        df = truncate_schedule_period(self._schedule, start, stop)
        df = expand_schedule_range(df)
        df = expand_schedule_multiplicity(df)
        return df

    def first_release_time(self) -> int:
        """First scheduled release time"""
        return int(self._schedule['release_start'].min())

    def update(self, model: "Model"):
        self._add_new(model)
        self._kill_old(model)

    # noinspection PyMethodMayBeStatic
    def _kill_old(self, model: "Model"):
        state = model.state
        if 'alive' in state:
            alive = state['alive']
            alive &= model.grid.ingrid(state['X'], state['Y'])
            state.remove(~alive)

    def _add_new(self, model: "Model"):
        # Get the portion of the release dataset that corresponds to
        # current simulation time
        df = self.new_particles(
            model.solver.time,
            model.solver.time + model.solver.step,
        )

        # If there are no new particles, but the state is empty, we should
        # still initialize the state by adding the appropriate columns
        if (len(df) == 0) and ('X' not in model.state):
            model.state.append(df.to_dict(orient='list'))

        # If there are no new particles, we are done.
        if len(df) == 0:
            return

        # If we are at the final time step, we should not release any more particles
        if model.solver.time >= model.solver.stop:
            return

        # Add new particles
        colnames = df.columns.tolist()
        new_particles = {k: df[k].to_numpy() for k in colnames}
        model.state.append(new_particles)


def load_release_file(stream, names: list, formats: dict) -> pd.DataFrame:
    if names is None:
        import re
        first_line = stream.readline()
        names = re.split(pattern=r'\s+', string=first_line.strip())

    converters = get_converters(varnames=names, conf=formats)

    df = pd.read_csv(
        stream,
        names=names,
        converters=converters,
        sep='\\s+',
    )
    df = df.sort_values(by='release_time')
    return df


def get_converters(varnames: list, conf: dict) -> dict:
    """
    Given a list of varnames and config keywords, return a dict of converters

    Returns a dict where the keys are ``varnames`` and the values are
    callables.

    :param varnames: For instance, ['release_time', 'X', 'Y']
    :param conf: For instance, {'release_time': 'time', 'X': 'float'}
    :return: A mapping of varnames to converters
    """
    dtype_funcs = dict(
        time=lambda item: np.datetime64(item, 's').astype('int64'),
        int=int,
        float=float,
    )

    dtype_defaults = dict(
        release_time='time',
    )

    converters = {}
    for varname in varnames:
        dtype_default = dtype_defaults.get(varname, 'float')
        dtype_str = conf.get(varname, dtype_default)
        dtype_func = dtype_funcs[dtype_str]
        converters[varname] = dtype_func

    return converters


def replace_lonlat_in_release_table(df, lonlat_converter):
    if "lon" not in df.columns or "lat" not in df.columns:
        return df

    X, Y = lonlat_converter(df["lon"].values, df["lat"].values)
    df_new = df.drop(columns=['X', 'Y', 'lat', 'lon'], errors='ignore')

    # If there were X/Y columns in the original dataset, only replace with
    # lat/lon conversion where the X/Y data were missing
    if 'X' in df.columns and 'Y' in df.columns:
        x_old = df['X'].to_numpy()
        y_old = df['Y'].to_numpy()
        is_invalid = np.isnan(x_old) | np.isnan(y_old)
        X[~is_invalid] = x_old
        Y[~is_invalid] = y_old

    df_new["X"] = X
    df_new["Y"] = Y
    return df_new


def add_start_stop_step_to_release_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts release times to release schedules

    A release table includes a column ``release_time`` which may be in string
    format or date format or posix seconds. Potentially, the release table also
    includes a column ``release_interval`` containing the scheduled intervals.
    This function appends three columns to the incoming table, namely
    release_start, release_stop and release_step. These are all in units of
    posix seconds. They define when the scheduled release starts, when it stops,
    and the interval between releases.
    """

    # Convert start of release events to posix seconds
    start = df['release_time'].to_numpy(dtype='datetime64[s]').astype('int64')

    # Load release intervals (0 = no repeats)
    max_step = 60 * 60 * 24 * 366 * 1_000_000
    if 'release_interval' in df.columns:
        step = df['release_interval'].to_numpy(dtype='int64', copy=True)
        step[step == 0] = max_step
    else:
        step = np.full(start.shape, fill_value=max_step, dtype='int64')

    # Define stop times for release events
    # Every time there is a new release, all previous continuous releases stop
    unq_start, unq_start_inv = np.unique(start, return_inverse=True)
    unq_stop = np.roll(unq_start, -1)
    if len(unq_stop):  # The last stop time is set to be "infinitely" large
        unq_stop[-1] = np.iinfo(unq_stop.dtype).max
    stop = unq_stop[unq_start_inv]

    return df.assign(release_start=start, release_stop=stop, release_step=step)


def truncate_schedule_period(
        df: pd.DataFrame,
        t1: int | None = None,
        t2: int | None = None,
) -> pd.DataFrame:
    """
    Return a release schedule truncated to the requested time interval.

    The input table is a pandas data frame containing the columns
    ``release_start``, ``release_stop`` and ``release_step``. The function
    removes rows whose release interval falls outside ``[t1, t2)`` and clamps
    the surviving row boundaries to that interval, while preserving the
    schedule columns needed for later expansion.

    :param df: Input release schedule as a pandas data frame.
    :param t1: Optional start time of the interval in POSIX seconds. If
        omitted, the first scheduled release start is used.
    :param t2: Optional stop time of the interval in POSIX seconds. If
        omitted, the last scheduled release stop is used.
    :returns: A truncated copy of the schedule with rows outside the interval
        removed and the relevant schedule bounds clipped to ``t1``/``t2``.

    .. note::
       The returned frame keeps the original schedule columns and is suitable
       for downstream expansion with ``expand_schedule_range``.
    """

    release_start = df['release_start'].to_numpy()
    release_stop = df['release_stop'].to_numpy()

    start = release_start[0] if t1 is None else t1
    stop = release_stop[-1] if t2 is None else t2

    # First row whose release_stop is greater than start.
    first = np.searchsorted(release_stop, start, side="right")

    # First row whose release_start is greater than or equal to stop.
    last = np.searchsorted(release_start, stop, side="left")

    if first >= last:
        return df.iloc[0:0].copy()

    # Remove irrelevant rows
    df = df.iloc[first:last].copy()

    # Truncate start times if necessary
    new_start = df['release_start'].to_numpy().copy()
    new_step = df['release_step'].to_numpy()
    idx = new_start < start
    new_start[idx] = reset_range_start(new_start[idx], new_step[idx], start)

    # Truncate stop times if necessary
    new_stop = np.minimum(df['release_stop'].to_numpy(), stop)

    df['release_start'] = new_start
    df['release_stop'] = new_stop

    return df


def expand_schedule_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands scheduled releases to actual releases

    A schedule is a data frame with columns release_start, release_stop and
    release_step. The function returns a data frame where each row is expanded
    to a number of rows according to the start, stop, step specification.
    
    Also, the columns release_start, release_stop and release_step are replaced
    with a single release_time column.
    """

    start = df['release_start'].to_numpy()
    stop = df['release_stop'].to_numpy()
    step = df['release_step'].to_numpy()
    num = np.maximum(np.ceil((stop - start) / step).astype('int64'), 0)
    seq = [i for n in num for i in range(n)]
    idx = np.repeat(np.arange(len(num)), num)
    times = start[idx] + seq * step[idx]

    drop_cols = ['release_start', 'release_stop', 'release_step', 'release_time']
    new_df = df.drop(columns=drop_cols, errors='ignore').iloc[idx]
    new_df['release_time'] = times

    return new_df


def reset_range_start(start, step, limit):
    """
    Get new start of range, keeping the new range aligned with the old
    
    Old range: [start, start + step, start + 2*step, ...]
    New range: [start + N*step, start + (N+1)*step, ...]

    The new range should be as large as possible while keeping limit <= start + N*step
    
    :returns: First element of new range
    """
    N_min = (limit - start) / step
    N = np.ceil(N_min).astype('int64')
    return start + N * step


def load_table(table, **open_kwargs) -> pd.DataFrame:
    """
    Load a table as a pandas data frame

    The input may be a path name or stream, or an in-memory table
    """
    import os

    if isinstance(table, (str, os.PathLike)):
        with open(table, mode='r', encoding='utf-8') as fp:
            return load_release_file(fp, **open_kwargs)

    elif hasattr(table, 'read'):
        return load_release_file(table, **open_kwargs)

    else:
        return pd.DataFrame(table)


def add_default_variables_in_release_table(df, defaults):
    df_new = df.copy()
    for k, v in defaults.items():
        if k not in df:
            df_new[k] = v
    return df_new


def expand_schedule_multiplicity(df):
    if 'mult' not in df:
        return df
    df = df.loc[np.repeat(df.index, df['mult'].values.astype('i4'))]
    df = df.reset_index(drop=True).drop(columns='mult')
    return df


def apply_warm_start_file(
        df: pd.DataFrame,
        file: str | nc.Dataset | None
) -> pd.DataFrame:
    """
    Apply warm start file to release schedule

    A warm start file is a ladim output file which can be used to re-start a
    simulation. The function applies the following changes to the
    input schedule:

    - The particles contained in the last time step of the warm start file are
      added to the release schedule

    - All previously scheduled releases are removed
    
    :param df: Input release schedule
    :param file: Input warm start file, or None if no-op
    :returns: New release schedule after warm start file has been applied
    """

    if file is None:
        return df  # No-op

    elif not isinstance(file, nc.Dataset):
        with nc.Dataset(file) as dset:
            return apply_warm_start_file(df, dset)

    warm_start_particles = load_last_timestep(file)

    if len(warm_start_particles) == 0:
        return df

    new_start_time = np.asarray(warm_start_particles['time'][0], dtype='datetime64[s]')

    df_truncated = truncate_schedule_period(df, new_start_time.astype(int) + 1, None)

    # Add warm start particles, using first row of the input schedule as a
    # template for default values.
    num_new = len(warm_start_particles)
    warm_df = df.loc[df.index[:1].repeat(num_new)].reset_index(drop=True)
    warm_df[['X', 'Y']] = np.nan
    for colname in set(df.columns).intersection(warm_start_particles.columns):
        warm_start_values = warm_start_particles[colname].to_numpy()
        if np.issubdtype(warm_start_values.dtype, np.datetime64):
            warm_start_values = warm_start_values.astype('datetime64[s]')
        target_dtype = np.dtype(str(warm_df[colname].dtype))
        warm_df.loc[:, colname] = warm_start_values.astype(target_dtype)
    warm_df['release_start'] = new_start_time.astype(np.int64)
    warm_df['release_stop'] = np.iinfo(df['release_stop'].dtype).max  # type: ignore
    warm_df['release_step'] = 60 * 60 * 24 * 366 * 1_000_000

    return pd.concat([warm_df, df_truncated], ignore_index=True)


def load_last_timestep(dset: nc.Dataset) -> pd.DataFrame:
    """Load particles from last time step of ladim output file"""

    if 'particle_count' not in dset.variables:
        raise ValueError('Missing variable "particle_count"')
    elif 'pid' not in dset.variables:
        raise ValueError('Missing variable "pid"')

    df = pd.DataFrame()

    # Group variable names by dimension
    varnames_by_dimension = {}
    for v in dset.variables:
        dims = dset.variables[v].dimensions
        if len(dims) != 1:
            continue
        dimname = str(dims[0]) if len(dims) > 0 else ''
        variable_list = varnames_by_dimension.get(dimname, [])
        variable_list.append(v)
        varnames_by_dimension[dimname] = variable_list

    # Load instance variables
    num_particles_last_step = dset['particle_count'][-1]
    for v in varnames_by_dimension.get('particle_instance', []):
        dset.variables[v].set_auto_mask(False)
        values = dset.variables[v][-num_particles_last_step:]
        df[v] = apply_cf_encoding(values, dset.variables[v])

    # Load time variables
    for v in varnames_by_dimension.get('time', []):
        dset.variables[v].set_auto_mask(False)
        values = dset.variables[v][-1]
        df[v] = apply_cf_encoding(values, dset.variables[v])

    # Load particle variables
    pid = df['pid'].values
    for v in varnames_by_dimension.get('particle', []):
        dset.variables[v].set_auto_mask(False)
        values = dset.variables[v][:][pid]
        df[v] = apply_cf_encoding(values, dset.variables[v])

    return df


def get_nc_attrs(variable: nc.Variable) -> dict:
    return {k: variable.getncattr(k) for k in variable.ncattrs()}


def apply_cf_encoding(values, variable: nc.Variable):
    attrs = get_nc_attrs(variable)

    if 'since' in attrs.get('units', ''):
        values = np.asarray(nc.num2date(
            times=values,
            units=attrs['units'],
            calendar=attrs.get('calendar', 'standard'),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )).astype('datetime64')

    return values
