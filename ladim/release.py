import contextlib

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
        self._last_release_dataframe = pd.DataFrame()
        self._last_release_time = np.int64(-4611686018427387904)

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
        ).copy(deep=True)

        # If there are no new particles, but the state is empty, we should
        # still initialize the state by adding the appropriate columns
        if (len(df) == 0) and ('X' not in model.state):
            model.state.append(df.to_dict(orient='list'))
            self._last_release_dataframe = df

        # If there are no new particles and we don't use continuous release,
        # we are done.
        continuous_release = bool(self._frequency)
        if (len(df) == 0) and not continuous_release:
            return

        # If we have continuous release, but there are no new particles and
        # the last release is recent, we are also done
        current_time = model.solver.time
        elapsed_since_last_write = current_time - self._last_release_time
        last_release_is_recent = (elapsed_since_last_write < self._frequency)
        if continuous_release and (len(df) == 0) and last_release_is_recent:
            return

        # If we are at the final time step, we should not release any more particles
        if continuous_release and model.solver.time >= model.solver.stop:
            return

        # If we have continuous release, but there are no new particles and
        # the last release is NOT recent, we should replace empty
        # dataframe with the previously released dataframe
        if continuous_release:
            if (len(df) == 0) and not last_release_is_recent:
                df = self._last_release_dataframe
            self._last_release_dataframe = df  # Update release dataframe
            self._last_release_time = current_time

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


def release_data_subset(dataframe, start_time, stop_time):
    start_idx, stop_idx = sorted_interval(
        dataframe['release_time'].values,
        start_time,
        stop_time,
    )
    return dataframe.iloc[start_idx:stop_idx]


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
