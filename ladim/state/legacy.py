"""
Class for the state of the model
"""

import sys
import os
import importlib
import logging
from typing import Any, Dict, Sized  # mypy

import numpy as np
from netCDF4 import Dataset, num2date

from ladim.tracker.legacy import Tracker
from ladim.gridforce.legacy import Grid, Forcing

# ------------------------

Config = Dict[str, Any]


class State(Sized):
    """The model variables at a given time"""

    def __init__(self, modules, **config):
        self.modules = modules

        logging.info("Initializing the model state")

        self.timestep = 0
        self.timestamp = config["start_time"].astype("datetime64[s]")
        self.dt = np.timedelta64(config["dt"], "s")
        self.position_variables = ["X", "Y", "Z"]
        if "ibm" in config and "variables" in config["ibm"]:
            self.ibm_variables = config["ibm"]["variables"]
        else:
            self.ibm_variables = config.get("ibm_variables", [])
        self.ibm_forcing = config["ibm_forcing"]
        self.particle_variables = config["particle_variables"]
        self.instance_variables = self.position_variables + [
            var for var in self.ibm_variables if var not in self.particle_variables
        ]

        self.pid = np.array([], dtype=int)
        for name in self.instance_variables:
            setattr(self, name, np.array([], dtype=float))

        for name in self.particle_variables:
            setattr(self, name, np.array([], dtype=config["release_dtype"][name]))

        self.dt = config["dt"]
        self.alive = []

        # self.num_particles = len(self.X)
        self.nnew = 0  # Modify with warm start?

        if config["warm_start_file"]:
            self.warm_start(config, self.modules['grid'])

    def __getitem__(self, name: str) -> None:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        return setattr(self, name, value)

    def __len__(self) -> int:
        return len(getattr(self, "X"))

    def append(self, new: Dict[str, Any], forcing: Forcing) -> None:
        """Append new particles to the model state"""
        nnew = len(new["pid"])
        self.pid = np.concatenate((self.pid, new["pid"]))
        for name in self.instance_variables:
            if name in new:
                self[name] = np.concatenate((self[name], new[name]))
            elif name in self.ibm_forcing:
                # Take values as Z must be a numpy array
                self[name] = np.concatenate(
                    (self[name], forcing.field(new["X"], new["Y"], new["Z"].values, name))
                )
            else:  # Initialize to zero
                self[name] = np.concatenate((self[name], np.zeros(nnew)))
        self.nnew = nnew

    def update(self):
        """Update the model state to the next timestep"""
        self.timestep += 1
        self.timestamp += np.timedelta64(self.dt, "s")

        if self.timestamp.astype("int") % 3600 == 0:  # New hour
            logging.info("Model time = {}".format(self.timestamp.astype("M8[h]")))

    def warm_start(self, config: Config, grid: Grid) -> None:
        """Perform a warm (re)start"""

        warm_start_file = config["warm_start_file"]
        try:
            f = Dataset(warm_start_file)
        except FileNotFoundError:
            logging.critical(f"Can not open warm start file: {warm_start_file}")
            raise SystemExit(1)

        logging.info("Reading warm start file")
        # Using last record in file
        tvar = f.variables["time"]
        warm_start_time = np.datetime64(num2date(tvar[-1], tvar.units))
        # Not needed anymore, explicitly set in configuration
        # if warm_start_time != config['start_time']:
        #    print("warm start time = ", warm_start_time)
        #    print("start time      = ", config['start_time'])
        #    logging.error("Warm start time and start time differ")
        #    raise SystemExit(1)

        pstart = f.variables["particle_count"][:-1].sum()
        pcount = f.variables["particle_count"][-1]
        self.pid = f.variables["pid"][pstart : pstart + pcount]
        # Give error if variable not in restart file
        for var in config["warm_start_variables"]:
            logging.debug(f"Reading {var} from warm start file")
            self[var] = f.variables[var][pstart : pstart + pcount]

        # Remove particles near edge of grid
        I = grid.ingrid(self["X"], self["Y"])
        self.pid = self.pid[I]
        for var in config["warm_start_variables"]:
            self[var] = self[var][I]
