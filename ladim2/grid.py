"""Abstract base class for LADiM grid"""


# import sys
# import os
# from pathlib import Path
from abc import ABC, abstractmethod

# import importlib
from typing import Tuple

import numpy as np  # type: ignore

ParticleArray = np.ndarray  # 1D array, one element per particle


class BaseGrid(ABC):
    """Abstract Base Class for LADiM grid"""

    @abstractmethod
    def __init__(self, modules: dict, **kwargs):
        self.modules = modules
        self.xmin: float
        self.xmax: float
        self.ymin: float
        self.ymax: float

    @abstractmethod
    def depth(self, X: ParticleArray, Y: ParticleArray) -> ParticleArray:
        """Estimates bottom depth at particle positions"""

    @abstractmethod
    def metric(
        self, X: ParticleArray, Y: ParticleArray
    ) -> Tuple[ParticleArray, ParticleArray]:
        """Estimates grid spacing at particle positions"""

    @abstractmethod
    def ingrid(self, X: ParticleArray, Y: ParticleArray) -> ParticleArray:
        """Tests if particles are inside the grid"""

    @abstractmethod
    def atsea(self, X: ParticleArray, Y: ParticleArray) -> ParticleArray:
        """Tests if particles are at sea"""
