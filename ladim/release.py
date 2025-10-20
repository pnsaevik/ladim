import contextlib
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
        particle source. Some columns have special meanings:

        release_start
            First release time of this particle type, in seconds since the posix
            epoch (1970-01-01).

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
            file = None, colnames: list = None, formats: dict = None,
            frequency=(0, 's'), defaults=None, lonlat_converter=None,
            warm_start_file: typing.Union[str, nc.Dataset, None] = None
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
        df = replace_lonlat_in_release_table(df, lonlat_converter)
        releaser = Releaser(df)

        releaser.warm_start_file = warm_start_file

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
        new_particles = df.to_dict(orient='list')
        state = model.state
        state.append(new_particles)

    def warm_start_time(self):
        with _open_nc_or_relay(self.warm_start_file) as dset:
            warm_start_time = dset.variables['time'][-1]

        return int(warm_start_time)

    def from_warm_start_file(self, model: "Model"):
        bool_vars_to_be_copied = ('alive', 'active')
        state = model.state

        with _open_nc_or_relay(self.warm_start_file) as dset:
            if not all(var in dset.variables for var in ('pid', 'particle_count')):
                raise ValueError("Warm start file must contain 'pid' and 'particle_count' variables")
            if 'particle' not in dset.dimensions:
                raise ValueError("Warm start file must contain 'particle' dimension")

            particle_count = dset.variables['particle_count'][-1]
            rows = dset.variables['pid'][:]
            pid = rows[-particle_count:]
            slice_dict = {
                'particle_instance': slice(-particle_count, None),
                'particle': pid
            }

            state['pid'] = pid
            state.released = len(dset.dimensions['particle'])
            for var_name, var in dset.variables.items():
                if len(var.dimensions) == 0:
                    continue
                dim_name, = var.dimensions
                if (dim_name in slice_dict) and (var_name != "pid"):
                    rows = var[:]
                    state[var_name] = rows[slice_dict[dim_name]]

        for x in bool_vars_to_be_copied:
            if x not in state:
                state[x] = np.ones(len(pid), dtype=bool)


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
    max_step = 60*60*24*366*1_000_000
    if 'release_interval' in df.columns:
        step = df['release_interval'].to_numpy(dtype='int64')
        step[step == 0] = max_step
    else:
        step = np.full(start.shape, fill_value=max_step, dtype='int64')
    
    # Define stop times for release events
    unq_start, unq_start_inv = np.unique(start, return_inverse=True)
    unq_stop = np.roll(unq_start, -1)
    if len(unq_stop):
        unq_stop[-1] = np.iinfo(unq_stop.dtype).max
    stop = unq_stop[unq_start_inv]

    return df.assign(release_start=start, release_stop=stop, release_step=step)


def truncate_schedule_period(df: pd.DataFrame, t1, t2) -> pd.DataFrame:
    """
    Returns a schedule truncated by start and stop time

    A schedule is a data frame with columns release_start, release_stop and
    release_step. The function returns a truncated version of the data frame
    with irrelevant rows removed, and with start- and stop times truncated to
    the given interval.
    """
    # Remove irrelevant rows
    idx = df['release_start'].values < t2
    idx &= df['release_stop'].values > t1
    df_subset = df.loc[idx].copy(deep=True)

    start = df_subset['release_start'].values
    stop = df_subset['release_stop'].values
    step = df_subset['release_step'].values

    # Truncate start times if necessary
    idx = start < t1
    new_start = reset_range_start(start[idx], step[idx], t1)
    df_subset.loc[idx, 'release_start'] = new_start

    # Truncate stop times if necessary
    idx = stop > t2
    df_subset.loc[idx, 'release_stop'] = t2

    return df_subset


def expand_schedule_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands scheduled releases to actual releases

    A schedule is a data frame with columns release_start, release_stop and
    release_step. The function returns a data frame where each row is expanded
    to a number of rows according to the start, stop, step specification.
    
    Also, the columns release_start, release_stop and release_step are replaced
    with a single release_time column.
    """

    start = df['release_start'].values
    stop = df['release_stop'].values
    step = df['release_step'].values
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

@contextlib.contextmanager
def _open_nc_or_relay(path_or_object: str | nc.Dataset, mode='r') -> typing.Generator[nc.Dataset, typing.Any, typing.Any]:
    if isinstance(path_or_object, str):
        with nc.Dataset(path_or_object, mode=mode) as dset:
            yield dset
    else:
        yield path_or_object
