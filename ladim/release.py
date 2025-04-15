import contextlib
import numpy as np
import pandas as pd
from .utilities import read_timedelta
import logging
import typing

if typing.TYPE_CHECKING:
    from ladim.model import Model


logger = logging.getLogger(__name__)


class Releaser:
    def __init__(self, particle_generator: typing.Callable[[float, float], pd.DataFrame]):
        self.particle_generator = particle_generator

    @staticmethod
    def from_textfile(
            file, colnames: list = None, formats: dict = None,
            frequency=(0, 's'), defaults=None, lonlat_converter=None,
    ):
        """
        Release module which reads from a text file

        The text file must be a whitespace-separated csv file

        :param lonlat_converter: Function that converts lon, lat coordinates to
            x, y coordinates

        :param file: Release file

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
        """

        release_table = ReleaseTable.from_filename_or_stream(
            file=file,
            column_names=colnames,
            column_formats=formats or dict(),
            interval=read_timedelta(frequency) / np.timedelta64(1, 's'),
            defaults=defaults or dict(),
            lonlat_converter=lonlat_converter,
        )
        return Releaser(particle_generator=release_table.subset)

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
        df = self.particle_generator(
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


def release_data_subset(dataframe, start_time, stop_time, interval: typing.Any = 0):
    events = resolve_schedule(
        times=dataframe['release_time'].values,
        interval=interval,
        start_time=start_time,
        stop_time=stop_time,
    )

    return dataframe.iloc[events]


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


class ReleaseTable:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            interval: float,
            defaults: dict[str, typing.Any],
            lonlat_converter: typing.Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ):
        self.dataframe = dataframe
        self.interval = interval
        self.defaults = defaults
        self.lonlat_converter = lonlat_converter

    def subset(self, start_time, stop_time):
        events = resolve_schedule(
            times=self.dataframe['release_time'].values,
            interval=self.interval,
            start_time=start_time,
            stop_time=stop_time,
        )

        df = self.dataframe.iloc[events].copy(deep=True)
        df = replace_lonlat_in_release_table(df, self.lonlat_converter)
        df = add_default_variables_in_release_table(df, self.defaults)
        df = expand_multiplicity_in_release_table(df)

        return df

    @staticmethod
    def from_filename_or_stream(file, column_names, column_formats, interval, defaults, lonlat_converter):
        with open_or_relay(file, 'r', encoding='utf-8') as fp:
            return ReleaseTable.from_stream(
                fp, column_names, column_formats, interval, defaults, lonlat_converter)

    @staticmethod
    def from_stream(fp, column_names, column_formats, interval, defaults, lonlat_converter):
        df = load_release_file(stream=fp, names=column_names, formats=column_formats)
        return ReleaseTable(df, interval, defaults, lonlat_converter)


def replace_lonlat_in_release_table(df, lonlat_converter):
    if "lon" not in df.columns or "lat" not in df.columns:
        return df

    X, Y = lonlat_converter(df["lon"].values, df["lat"].values)
    df_new = df.drop(columns=['X', 'Y', 'lat', 'lon'], errors='ignore')
    df_new["X"] = X
    df_new["Y"] = Y
    return df_new


def add_default_variables_in_release_table(df, defaults):
    df_new = df.copy()
    for k, v in defaults.items():
        if k not in df:
            df_new[k] = v
    return df_new


def expand_multiplicity_in_release_table(df):
    if 'mult' not in df:
        return df
    df = df.loc[np.repeat(df.index, df['mult'].values.astype('i4'))]
    df = df.reset_index(drop=True).drop(columns='mult')
    return df


def resolve_schedule(times, interval, start_time, stop_time):
    """
    Convert decriptions of repeated events to actual event indices

    The variable `times` specifies start time of scheduled events. Each event occurs
    repeatedly (specified by `interval`) until there is a new scheduling time.
    The function returns the index of all events occuring within the time span.

    Example 1: times = [0, 0], interval = 2. These are 2 events (index [0, 1]),
    occuring at times [0, 2, 4, 6, ...], starting at time = 0. The time interval
    start_time = 0, stop_time = 6 will contain the event times 0, 2, 4. The
    returned event indices are [0, 1, 0, 1, 0, 1].

    Example 2: times = [0, 0, 3, 3, 3], interval = 2. The schedule starts with
    2 events (index [0, 1]) occuring at time = 0. At time = 2, there are no new
    scheduled events, and the previous events are repeated. At time = 3 there
    are 3 new events scheduled (index [2, 3, 4]), which cancel the previous
    events. The new events are repeated at times [3, 5, 7, ...]. The time
    interval start_time = 0, stop_time = 7 contain the event times [0, 2, 3, 5].
    The returned event indices are [0, 1, 0, 1, 2, 3, 4, 2, 3, 4].

    :param times: Nondecreasing list of event times
    :param interval: Maximum interval between scheduled times
    :param start_time: Start time of schedule
    :param stop_time: Stop time of schedule (not inclusive)
    :return: Index of events in resolved schedule
    """

    sched = Schedule(times=np.asarray(times), events=np.arange(len(times)))
    sched2 = sched.resolve(start_time, stop_time, interval)
    return sched2.events


class Schedule:
    def __init__(self, times: np.ndarray, events: np.ndarray):
        self.times = times.view()
        self.events = events.view()
        self.times.flags.writeable = False
        self.events.flags.writeable = False

    def valid(self):
        return np.all(np.diff(self.times) >= 0)

    def copy(self):
        return Schedule(times=self.times.copy(), events=self.events.copy())

    def append(self, other: "Schedule"):
        return Schedule(
            times=np.concatenate((self.times, other.times)),
            events=np.concatenate((self.events, other.events)),
        )

    def extend_backwards_using_interval(self, time, interval):
        min_time = self.times[0]
        if min_time <= time:
            return self

        num_extensions = int(np.ceil((min_time - time) / interval))
        new_time = min_time - num_extensions * interval
        return self.extend_backwards(new_time)

    def extend_backwards(self, new_minimum_time):
        idx_to_be_copied = (self.times == self.times[0])
        num_to_be_copied = np.count_nonzero(idx_to_be_copied)
        extension = Schedule(
            times=np.repeat(new_minimum_time, num_to_be_copied),
            events=self.events[idx_to_be_copied],
        )
        return extension.append(self)

    def trim_tail(self, stop_time):
        num = np.sum(self.times < stop_time)
        return Schedule(
            times=self.times[:num],
            events=self.events[:num],
        )

    def trim_head(self, start_time):
        num = np.sum(self.times < start_time)
        return Schedule(
            times=self.times[num:],
            events=self.events[num:],
        )

    def rightshift_closest_time_value(self, time, interval):
        # If interval=0 is specified, this means there is nothing to right-shift
        if interval <= 0:
            return self

        # Find largest value that is smaller than or equal to time
        idx_target_time = sum(self.times <= time) - 1

        # If no tabulated time values are smaller than the given time, there
        # is nothing to right-shift
        if idx_target_time == -1:
            return self

        # Compute new value to write
        target_time = self.times[idx_target_time]
        num_offsets = np.ceil((time - target_time) / interval)
        new_target_time = target_time + num_offsets * interval

        # Check if the new value is larger than the next value
        if idx_target_time + 1 < len(self.times):  # If not, then there is no next value
            next_time = self.times[idx_target_time + 1]
            if new_target_time > next_time:
                return self

        # Change times
        new_times = self.times.copy()
        new_times[self.times == target_time] = new_target_time
        return Schedule(times=new_times, events=self.events)

    def expand(self, interval, stop):
        # If there are no times, there should be no expansion
        # Also, interval = 0 means no expansion
        if (len(self.times) == 0) or (interval <= 0):
            return self

        t_unq, t_inv, t_cnt = np.unique(self.times, return_inverse=True, return_counts=True)
        stop2 = np.maximum(stop, t_unq[-1])
        diff = np.diff(np.concatenate((t_unq, [stop2])))
        unq_repeats = np.ceil(diff / interval).astype(int)
        repeats = np.repeat(unq_repeats, t_cnt)

        base_times = np.repeat(self.times, repeats)
        offsets = [i * interval for n in repeats for i in range(n)]
        times = base_times + offsets
        events = np.repeat(self.events, repeats)

        idx = np.lexsort((events, times))

        return Schedule(times=times[idx], events=events[idx])

    def resolve(self, start, stop, interval):
        s = self
        if interval:
            s = s.rightshift_closest_time_value(start, interval)
        s = s.trim_head(start)
        s = s.trim_tail(stop)
        s = s.expand(interval, stop)
        return s


@contextlib.contextmanager
def open_or_relay(file_or_buf, *args, **kwargs):
    if hasattr(file_or_buf, 'read'):
        yield file_or_buf
    else:
        with open(file_or_buf, *args, **kwargs) as f:
            yield f
