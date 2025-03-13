import contextlib
import typing

from .model import Model, Module
import numpy as np
import pandas as pd
from .utilities import read_timedelta
import logging


logger = logging.getLogger(__name__)


class Releaser(Module):
    pass


class TextFileReleaser(Releaser):
    def __init__(
            self, file, colnames: list = None, formats: dict = None,
            frequency=(0, 's'), defaults=None,
    ):
        """
        Release module which reads from a text file

        The text file must be a whitespace-separated csv file

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

        # Release file
        self._csv_fname = file   # Path name
        self._csv_column_names = colnames   # Column headers
        self._csv_column_formats = formats or dict()
        self._dataframe = None

        # Continuous release variables
        self._frequency = read_timedelta(frequency) / np.timedelta64(1, 's')

        # Other parameters
        self._defaults = defaults or dict()

    def update(self, model: Model):
        self._add_new(model)
        self._kill_old(model)

    # noinspection PyMethodMayBeStatic
    def _kill_old(self, model: Model):
        state = model.state
        if 'alive' in state:
            alive = state['alive']
            alive &= model.grid.ingrid(state['X'], state['Y'])
            state.remove(~alive)

    def _add_new(self, model: Model):
        # Get the portion of the release dataset that corresponds to
        # current simulation time
        df = release_data_subset(
            dataframe=self.dataframe,
            start_time=model.solver.time,
            stop_time=model.solver.time + model.solver.step,
            interval=self._frequency,
        ).copy(deep=True)

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

        # If positions are given as lat/lon coordinates, we should convert
        if "X" not in df.columns or "Y" not in df.columns:
            if "lon" not in df.columns or "lat" not in df.columns:
                logger.critical("Particle release must have position")
                raise ValueError()
            # else
            X, Y = model.grid.ll2xy(df["lon"].values, df["lat"].values)
            df.rename(columns=dict(lon="X", lat="Y"), inplace=True)
            df["X"] = X
            df["Y"] = Y

        # Add default variables, if any
        for k, v in self._defaults.items():
            if k not in df:
                df[k] = v

        # Expand multiplicity variable, if any
        if 'mult' in df:
            df = df.loc[np.repeat(df.index, df['mult'].values.astype('i4'))]
            df = df.reset_index(drop=True).drop(columns='mult')

        # Add new particles
        new_particles = df.to_dict(orient='list')
        state = model.state
        state.append(new_particles)

    @property
    def dataframe(self):
        @contextlib.contextmanager
        def open_or_relay(file_or_buf, *args, **kwargs):
            if hasattr(file_or_buf, 'read'):
                yield file_or_buf
            else:
                with open(file_or_buf, *args, **kwargs) as f:
                    yield f

        if self._dataframe is None:
            if isinstance(self._csv_fname, pd.DataFrame):
                self._dataframe = self._csv_fname

            else:
                # noinspection PyArgumentList
                with open_or_relay(self._csv_fname, 'r', encoding='utf-8') as fp:
                    self._dataframe = load_release_file(
                        stream=fp,
                        names=self._csv_column_names,
                        formats=self._csv_column_formats,
                    )
        return self._dataframe


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


def sorted_interval(v, a, b):
    """
    Searches for an interval in a sorted array

    Returns the start (inclusive) and stop (exclusive) indices of
    elements in *v* that are greater than or equal to *a* and
    less than *b*. In other words, returns *start* and *stop* such
    that v[start:stop] == v[(v >= a) & (v < b)]

    :param v: Sorted input array
    :param a: Lower bound of array values (inclusive)
    :param b: Upper bound of array values (exclusive)
    :returns: A tuple (start, stop) defining the output interval
    """
    start = np.searchsorted(v, a, side='left')
    stop = np.searchsorted(v, b, side='left')
    return start, stop


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
