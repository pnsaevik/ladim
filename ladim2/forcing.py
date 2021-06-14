"""Abstract base class for LADiM forcing"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# November 2020
# ----------------------------------

# import sys
# import os
# import importlib
# from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import numpy as np  # type: ignore

ParticleArray = np.ndarray  # 1D array, one element per particle


class BaseForce(ABC):
    """Abstract base class for LADiM forcing"""

    @abstractmethod
    def __init__(self, modules: Dict[str, str], **kwargs):
        self.modules = modules
        self.variables: Dict[str, np.ndarray]

    @abstractmethod
    def update(self) -> None:
        """Update the forcing to the next time step"""

    @abstractmethod
    def velocity(
        self,
        X: ParticleArray,
        Y: ParticleArray,
        Z: ParticleArray,
        fractional_step: float = 0,
        method: str = "bilinear",
    ) -> Tuple[ParticleArray, ParticleArray]:
        """Estimate velocity at particle positions"""

    @abstractmethod
    def close(self) -> None:
        """Close the (last) forcing file"""
