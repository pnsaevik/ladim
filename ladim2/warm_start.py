"""Perform initialization for warm start in LADiM"""

import logging
from typing import List

import numpy as np
from netCDF4 import Dataset

from ladim2.state import State

DEBUG = False

logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)


def warm_start(
    warm_start_file: str, warm_start_variables: List[str], state: State
) -> None:
    """Initiate the state from a warm start"""

    logger.info("Warm start")
    logger.info("  Warm start file %s", warm_start_file)

    logger.info("  warm start variables: %s", warm_start_variables)

    # wvars = warm_start_variables.copy() + ["pid"]
    wvars: set = {"pid", "X", "Y", "Z", "alive", "active"}.union(warm_start_variables)

    # Open warm start file
    try:
        f = Dataset(warm_start_file)
        f.set_auto_mask(False)
    except FileNotFoundError as err:
        # logging.critical(f"Can not open warm start file: {warm_start_file}")
        raise SystemExit(1) from err

    # Use last record in file
    pstart = f.variables["particle_count"][:-1].sum()
    pcount = f.variables["particle_count"][-1]
    pend = pstart + pcount
    pid_max = np.max(f.variables["pid"][:]) + 1

    state.npid = pid_max

    # Variable loop
    for var in wvars:
        if var in f.variables:
            ncvar = f.variables[var]
            if var in state.instance_variables:
                values = ncvar[pstart:pend]
            else:  # Particle variable
                values = ncvar[:pid_max]
            # Time type variable needs special treatment
            if "units" in ncvar.ncattrs() and "since" in ncvar.units:
                reftime = np.datetime64(ncvar.units.split("since")[1])
                values = reftime + values * np.timedelta64(1, ncvar.units[0])
        # Variables not on file, but with defaults
        elif var in state.default_values:
            if var in state.instance_variables:
                shape = (pcount,)
            else:
                shape = (pid_max,)
            values = np.full(shape, state.default_values[var])
        else:
            logger.error("Warm start: No value for variable %s", var)
            raise SystemExit(1)

        state.variables[var] = values

    # # Instance variables with default
    # if "alive" not in wvars:
    #     state.variables["alive"] = np.ones(pcount).astype("bool")
    # if "active" not in wvars:
    #     state.variables["active"] = np.ones(pcount).astype("bool")
