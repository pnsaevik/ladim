"""Abstract base class for LADiM forcing"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# November 2020
# ----------------------------------

import sys
import os
import importlib
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np

from ladim2.state import State


class BaseOutput(ABC):
    """Abstract base class for LADiM forcing"""

    output_period = np.timedelta64(0, "s")

    @abstractmethod
    def __init__(self, modules: Dict[str, Any], **kwargs) -> None:
        self.modules = modules

    @abstractmethod
    def update(self) -> None:
        """Write data from instance variables to output file"""

    @abstractmethod
    def write_particle_variables(self, state: State) -> None:
        """Write data from particle variables to output file"""

    @abstractmethod
    def close(self) -> None:
        """Close (last) output file"""


def init_output(module: str, **args) -> BaseOutput:
    """Initiates an Output class

    Args:
        module:
            Name of the module defining the Output class
        args:
            Keyword arguments passed on to the Output instance

    Returns:
        An Output instance

    The module should be in the LADiM source directory or in the working directory.
    The working directory takes priority.
    The Output class in the module should be named "Output".
    """
    # System path for ladim2.ladim2
    p = Path(__file__).parent
    sys.path.insert(0, str(p))
    # Working directory
    sys.path.insert(0, os.getcwd())

    # Import correct module
    output_module = importlib.import_module(module)
    return output_module.Output(**args)  # type: ignore
