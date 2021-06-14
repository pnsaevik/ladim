"""
Class for the state of the model
"""

# ----------------------------------
# Bjørn Ådlandsvik
# Institute of Marine Research
# October 2020
# ----------------------------------

import logging
from numbers import Number
from collections.abc import Sized
from typing import Dict, Union, Sequence, Optional, Any

import numpy as np  # type: ignore

DEBUG = False

logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)

# From State point-of-view the only difference between
# instance and particle variables is that instance variables
# are affected by the compactify operation

# ------------------------

# Define some types aliases
Scalar = Number
Arraylike = Union[np.ndarray, Sequence[Scalar], Scalar]
ParticleArray = np.ndarray
Dtype = Union[type, np.dtype]

# State has no internal difference between particle and instance variable.
# With append also particle variables needs a value
# Have an undef value?, nan for float types,
# Should not compactify particle variables


class State(Sized):
    """
    The model variables at a given time

    Mandatory state variables: pid, alive, active, X, Y, Z
    with predefined dtypes int, 2*bool, and 3*float

    attributes:
        npid: Number of particles released so far
        variables: Dictionary of state variables
        default_values: Default values for initializing state variables

    methods:
        set_default_values: Define default values for state variables
        append: Append new particles to the state
        compactify: Throw out dead particles

    """

    # The data are kept in the variables dictionary
    # instance_variables and particle_variables are sets of names

    # The length of the state should only be allowed to change by
    # append and compactify (rename the last to kill?)

    __slots__ = [
        "modules",
        "variables",
        "instance_variables",
        "particle_variables",
        "default_values",
        "dtypes",
        "npid",
    ]

    def __init__(
        self,
        instance_variables: Optional[Dict[str, type]] = None,
        particle_variables: Optional[Dict[str, type]] = None,
        default_values: Optional[Dict[str, Scalar]] = None,
        modules: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the state

        instance_variables: extra (non-mandatory) variables with dtype
        particle_variables: particle variables with dtype
        default_values: default initial values

        mandatory particle variables: pid, X, Y, Z, active, alive
        predefined defaults: active = True, alive = True

        """

        logger.info("Initiating the model state")
        self.modules = modules
        # Make dtypes dictionary
        mandatory_variables = dict(
            pid=int, X=float, Y=float, Z=float, active=bool, alive=bool
        )
        ivar = instance_variables if instance_variables else dict()
        # Union of dictionaries, python 3.9: mandatory_variables | ivar
        # Explicitly set instance variables override the defaults
        ivar = dict(mandatory_variables, **ivar)
        self.instance_variables = set(ivar)
        logger.info("  Instance variables: %s", list(ivar))

        pvar = particle_variables if particle_variables else dict()
        self.particle_variables = set(pvar)
        logger.info("  Particle variables: %s", list(pvar))

        # Raises TypeError if overlap
        self.dtypes: Dict[str, Dtype] = dict(**ivar, **pvar)
        # expand aliases (presently only "time")
        for var in self.dtypes:
            if self.dtypes[var] == "time":
                self.dtypes[var] = np.dtype("M8[s]")  # datetime64("s")

        # Data storage (initally empty)
        self.variables = {
            var: np.array([], dtype) for var, dtype in self.dtypes.items()
        }

        # Default values
        predef_default_values = dict(
            alive=np.array(True, dtype=bool), active=np.array(True, dtype=bool)
        )
        dvals = default_values if default_values else dict()
        # Some quality control
        if "pid" in dvals:
            raise ValueError("Can not set default for pid")
        for var, value in dvals.items():
            if var not in self.variables:
                raise ValueError("No such variable: ", var)
            if not np.isscalar(value):
                raise TypeError(f"Default value for {var} should be scalar")

        self.default_values = dict(predef_default_values, **dvals)

        for var in self.variables:
            if var not in self.default_values:
                self.default_values[var] = np.nan

        self.npid: int = 0  # Total number of pids used

    def append(self, **args: Arraylike) -> None:
        """Append particles to the State object"""

        # state_vars = instance_variables (without pid)
        state_vars = set(self.variables) - {"pid"}

        # Accept only state_vars
        for name in args:
            if name not in state_vars:
                raise ValueError(f"Invalid argument {name}")

        # Variables must have a value
        value_vars = dict(self.default_values, **args)
        for name in state_vars:
            if name not in value_vars:
                self.default_values[name] = np.nan

        # Broadcast all variables to 1D arrays
        #    Raise ValueError if not compatible
        b = np.broadcast(*value_vars.values())
        if b.ndim > 1:
            raise ValueError("Arguments must be 1D or scalar")
        # if b.ndim == 0:  # All arguments are scalar
        #    b = np.broadcast([0])  # Make b.size = 1
        num_new_particles = b.size
        logger.debug("Appending %d particles", num_new_particles)
        values = {
            var: np.broadcast_to(v, shape=(num_new_particles,))
            for var, v in value_vars.items()
        }

        # pid must be handles separately
        self.variables["pid"] = np.concatenate(
            (
                self.variables["pid"],
                np.arange(self.npid, self.npid + num_new_particles, dtype=int),
            )
        )
        self.npid = self.npid + num_new_particles

        # Concatenate the rest of the variables
        for var in state_vars:
            self.variables[var] = np.concatenate((self.variables[var], values[var]))

    def compactify(self) -> None:
        """Remove dead particles from the instance variables"""

        n_particles = len(self.variables["pid"])
        n_alive = sum(self.variables["alive"])
        n_remove = n_particles - n_alive
        if n_remove > 0:
            logger.debug("Compactifying: removing %d particles", n_remove)
            # Take a copy, so that variables["alive"] can be compactified
            alive = self.alive.copy()
            for var in self.instance_variables:
                self.variables[var] = self.variables[var][alive]

    def __len__(self) -> int:
        return len(self.pid)

    # Allow item notation, state[var]
    def __getitem__(self, var: str) -> ParticleArray:
        """Allow item style write access to variables"""
        return self.variables[var]

    def __setitem__(self, var: str, item: Arraylike) -> None:
        """Allow item style read access to variables"""
        value = np.array(item, dtype=self.dtypes[var])
        # The size of item should be unchanged
        if np.size(value) != len(self.variables[var]):
            raise KeyError("Size of data should be unchanged")
        self.variables[var] = value

    def __getattr__(self, var: str) -> ParticleArray:
        """Allow read access in attribute style"""
        return self.variables[var]

    def __setattr__(self, var: str, value: Any) -> None:
        """Follow pandas and xarray and disallow attribute assignment"""
        # Method taken from xarray
        try:  # allow asigning attributes in __slots__
            object.__setattr__(self, var, value)
        except AttributeError as e:
            raise AttributeError(
                f"Cannot assign attribute {var} on a State instance.\n"
                + f'Use state["{var}"] = ...  instead.'
            ) from e
