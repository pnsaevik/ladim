from .model import Model, Module
import numpy as np
import pandas as pd
from .utilities import read_timedelta


class Releaser(Module):
    def __init__(self, model: Model):
        super().__init__(model)


class TextFileReleaser(Releaser):
    def __init__(self, model: Model, **conf):
        super().__init__(model)

        # Release file
        self._csv_fname = conf['particle_release_file']   # Path name
        self._csv_column_names = conf['release_format']   # Column headers
        self._csv_column_formats = conf['release_dtype']  # Column data formats
        self._dataframe = None                            # Loaded dataframe

        # Continuous release variables
        self._frequency = read_timedelta(conf.get('release_frequency', [0, 's']))
        self._last_release_dataframe = pd.DataFrame()
        self._last_release_time = np.datetime64('NaT')

    @staticmethod
    def _get_converters(varnames: list, conf: dict) -> dict:
        """
        Given a list of varnames and config keywords, return a dict of converters

        Returns a dict where the keys are ``varnames`` and the values are
        callables.

        :param varnames: For instance, ['release_time', 'X', 'Y']
        :param conf: For instance, {'release_time': 'time', 'X': 'float'}
        :return: A mapping of varnames to converters
        """
        dtype_funcs = dict(
            time=np.datetime64,
            float=float,
        )

        converters = {}
        for varname in varnames:
            dtype_str = conf.get(varname, 'float')
            dtype_func = dtype_funcs[dtype_str]
            converters[varname] = dtype_func

        return converters

    def update(self):
        # Get the portion of the release dataset that corresponds to
        # current simulation time
        df = release_data_subset(
            dataframe=self.dataframe,
            start_time=self.model.solver.time,
            stop_time=self.model.solver.time + self.model.solver.step,
        )

        # If there are no new particles and we don't use continuous release,
        # we are done.
        continuous_release = bool(self._frequency)
        if (len(df) == 0) and not continuous_release:
            return

        # If we have continuous release, but there are no new particles and
        # the last release is recent, we are also done
        current_time = self.model.solver.time
        elapsed_since_last_write = current_time - self._last_release_time
        last_release_is_recent = (elapsed_since_last_write < self._frequency)
        if continuous_release and (len(df) == 0) and last_release_is_recent:
            return

        # If we have continuous release, but there are no new particles and
        # the last release is NOT recent, we should replace the empty
        # dataframe with the previously released dataframe
        if continuous_release:
            if (len(df) == 0) and not last_release_is_recent:
                df = self._last_release_dataframe
            self._last_release_dataframe = df  # Update release dataframe
            self._last_release_time = current_time

        # Add new particles
        new_particles = df.to_dict(orient='list')
        self.model.state.append(new_particles)

    @property
    def dataframe(self):
        if self._dataframe is None:
            converters = self._get_converters(
                varnames=self._csv_column_names,
                conf=self._csv_column_formats,
            )
            self._dataframe = load_release_file(
                fname=self._csv_fname,
                names=self._csv_column_names,
                formats=converters,
            )
        return self._dataframe


def release_data_subset(dataframe, start_time, stop_time):
    start_idx, stop_idx = sorted_interval(
        dataframe['release_time'].values,
        start_time,
        stop_time,
    )
    return dataframe.iloc[start_idx:stop_idx]


def load_release_file(fname: str, names: list, formats: dict) -> pd.DataFrame:
    df = pd.read_csv(
        fname,
        names=names,
        converters=formats,
        delim_whitespace=True,
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
