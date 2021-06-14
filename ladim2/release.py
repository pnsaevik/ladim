"""Particle release module for LADiM"""

from collections.abc import Iterator
from pathlib import Path
import logging
from typing import List, Optional, Union, Dict, Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from netCDF4 import Dataset  # type: ignore

# from ladim2.timekeeper import TimeKeeper, normalize_period
from ladim2.timekeeper import normalize_period
from ladim2.grid import BaseGrid


# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# Bergen, Norway
# ----------------------------------

DEBUG = False
logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)


class ParticleReleaser(Iterator):
    """Particle Release Class"""

    def __init__(
        self,
        modules: Dict[str, Any],
        release_file: Union[Path, str],
        names: Optional[List[str]] = None,
        continuous: Optional[bool] = False,
        release_frequency: int = 0,  # frequency in seconds
        warm_start_file: Optional[str] = None,
        **args,
    ) -> None:
        timer = modules["time"]
        grid = modules["grid"]
        datatypes = modules["state"].dtypes
        self.modules = modules
        self.start_time = timer.start_time
        self.stop_time = timer.stop_time
        self.time_reversal = timer.time_reversal

        logger.info("Initializing the particle releaser")

        self._df = self.read_release_file(release_file, datatypes, names)
        if continuous:
            logger.info("  Continuous release")
        else:
            logger.info("  Discrete release")
        logger.info("  Release file: %s", release_file)

        # If no mult column, add a column of ones
        if "mult" not in self._df.columns:
            self._df["mult"] = 1

        self.clean_position(grid)

        # Remove everything after simulation stop time
        if timer.time_reversal:
            self._df = self._df[self._df.index >= self.stop_time]  # Use < ?
        else:
            self._df = self._df[self._df.index <= self.stop_time]  # Use < ?
        if len(self._df) == 0:  # All release after simulation time
            logger.critical("All particles released after similation stop")
            raise SystemExit(3)

        # Make the dataframe explicitly discrete
        # REMARK: continuous flag is unnecessary,
        # could use release_frequency == True (not None, > 0)
        if continuous:
            self.release_frequency = normalize_period(release_frequency)
            self.discretize()

        # Now discrete, remove everything before start
        if timer.time_reversal:
            self._df = self._df[self._df.index <= self.start_time]
        else:
            self._df = self._df[self._df.index >= self.start_time]

        # With warm start skip release at start time (already accounted for)
        if warm_start_file:
            logging.debug("warm start in release")
            self._df = self._df[self._df.index > self.start_time]

        # Avoid simulations without particles
        # Cold start and all particles released before start
        if len(self._df) == 0 and not warm_start_file:
            logger.critical("All particles released before simulation start")
            raise SystemExit(3)

        # Add a release_time column if requested by the datatypes
        if "release_time" in datatypes.keys():
            self._df["release_time"] = self._df.index

        # # Optionally, remove everything outside a subgrid
        # try:
        #     subgrid: List[int] = config["grid_args"]["subgrid"]
        # except KeyError:
        #     subgrid = []
        # if subgrid:
        #     lenA = len(A)
        #     A = A[ingrid(A["X"], A["Y"], subgrid)]
        #     if len(A) < lenA:
        #         logger.warning("Ignoring particle release outside subgrid")

        if warm_start_file:
            # Get particle data from  warm start file
            with Dataset(warm_start_file) as f:
                warm_particle_count = np.max(f.variables["pid"][:]) + 1
            logger.info("  warm_particle_count: %d", warm_particle_count)
        #         for name in config["particle_variables"]:
        #             pvars[name] = f.variables[name][:warm_particle_count]
        else:  # cold start
            warm_particle_count = 0

        # Total number of particles released
        self.total_particle_count = self._df.mult.sum() + warm_particle_count
        logger.info("  Total particle count: %d", self.total_particle_count)

        # Release times
        self.times = self._df.index.unique()
        self.steps = [timer.time2step(t) for t in self.times]
        logger.info("  Number of release times: %d", len(self.times))

        # Make dataframes for each timeframe
        self._B = [x[1] for x in self._df.groupby(self._df.index)]

        # # Read the particle variables
        self._index = 0  # Index of next release
        self._particle_count = warm_particle_count

    def update(self) -> None:
        """Release new particles (if any)"""
        step = self.modules["time"].step
        if step in self.steps:
            V = next(self)
            self.modules["state"].append(**V)

    def __next__(self) -> pd.DataFrame:
        """Perform the next particle release

           Return a DataFrame with the release info,
           repeated mult times

        """

        logger.debug("Particle release time")

        # This should not happen
        if self._index >= len(self.times):
            raise StopIteration

        # Skip first release if warm start (should be present in start file)
        # Not always, make better test
        # Moving test to state.py
        # if self._index == 0 and self._particle_count > 0:  # Warm start
        #    return

        # rel_time = self.times[self._index]
        # file_time = self._file_times[self._file_index]

        V = self._B[self._index]
        # nnew = V.mult.sum()
        # Workaround, missing repeat method for pandas DataFrame
        V0 = V.to_records(index=False)
        V0 = V0.repeat(V.mult)
        V = pd.DataFrame(V0)
        # Do not need the mult column any more
        V.drop("mult", axis=1, inplace=True)
        # Buffer the new values

        # Update the counters
        self._index += 1
        self._particle_count += len(V)

        logger.debug("Appending %d new particles", len(V))

        return V

    @staticmethod
    def read_release_file(
        rls_file: Union[Path, str],
        datatypes: Dict[str, Any],
        names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Read the release file into a pandas DataFrame"""

        # Handle time separately
        time_vars = ["release_time"]
        dtypes = dict()
        for var, dtype in datatypes.items():
            if dtype in ["M8[s]", np.dtype("M8[s]"), np.dtype("M8[ns]")]:
                time_vars.append(var)
            else:
                dtypes[var] = dtype
        # Standard dtypes
        dtypes["mult"] = int
        dtypes["X"] = float
        dtypes["Y"] = float
        dtypes["Z"] = float
        dtypes["lon"] = float
        dtypes["lat"] = float

        df = pd.read_csv(
            rls_file,
            parse_dates=list(set(time_vars)),
            names=names,
            dtype=dtypes,
            delim_whitespace=True,
            index_col="release_time",
        )

        return df

    def clean_position(self, grid: Optional[BaseGrid] = None) -> None:
        """Make sure the release data have mult, X, and Y columns

        X and Y may be inferred from lon and lat using grid.ll2xy
        """

        df = self._df

        # Conversion from longitude, latitude to grid coordinates
        if "X" not in df.columns or "Y" not in df.columns:
            if "lon" not in df.columns or "lat" not in df.columns:
                logger.critical("Particle release must include position")
                raise SystemExit(3)
            try:
                X, Y = grid.ll2xy(df["lon"], df["lat"])  # type: ignore
            except AttributeError as err:
                logger.critical("Can not convert from lon/lat to grid coordinates")
                raise SystemExit(3) from err

            df["lon"] = X
            df["lat"] = Y
            df.rename(columns={"lon": "X", "lat": "Y"}, inplace=True)

        self._df = df

    def discretize(self) -> None:
        """Make a continuous release sequence discrete"""

        df = self._df

        # Find last release time <= start_time
        if not self.time_reversal:
            n = np.sum(df.index <= self.start_time)
        else:
            n = np.sum(df.index >= self.start_time)

        if n == 0:
            logger.warning("No particles released at simulation start")
            n = 1  # Use first release entry
        release_time0 = df.index[n - 1]
        # if DEBUG:
        #    print("release_time0 =", release_time0)
        # Remove the early entries
        # NOTE: Makes a new DataFrame
        if not self.time_reversal:
            df = df[df.index >= release_time0]
        else:
            df = df[df.index <= release_time0]

        file_times = df.index.unique()

        if not self.time_reversal:
            freq = self.release_frequency
        else:
            freq = -self.release_frequency

        times = np.arange(file_times[0], self.stop_time, np.timedelta64(freq, "s"))

        # Reindex the index
        J = pd.Series(file_times, index=file_times).reindex(times, method="ffill")
        num_entries_per_time = {i: mylen(df.loc[i]) for i in file_times}
        df = df.loc[J]

        # Set non-unique index
        S: List[int] = []
        for t in times:
            S.extend(num_entries_per_time[J[t]] * [t])
        df.index = S

        self._df = df


def mylen(df: pd.DataFrame) -> int:
    """Number of rows in a DataFrame,

    A workaround for len() which does not
    have the expected behaviour with itemizing,
    """
    if df.ndim > 1:
        length: int = df.shape[0]
    else:
        length = 1
    return length
