"""
zROMS forcing: velocity and scalar fields sampled at particle positions.

As for the ROMS forcing (see :mod:`ladim.gridforce.roms.forcing`), but for
files interpolated to z levels. The time variable is called ``time``, and
the first dimension of the 3-D variables is the vertical level.
"""

import logging

from ..roms.forcing import Forcing as RomsForcing
from .grid import Grid

logger = logging.getLogger(__name__)


class Forcing(RomsForcing):
    """
    Class for zROMS forcing
    """

    grid_class = Grid
    time_name = "time"

    def __init__(self, config, _=None):
        super().__init__(config, _)
        nz = self._u.shape[0]
        n = self._grid.N
        if n is None:
            raise ValueError(
                "The depths of the z levels are unknown. Specify them with "
                "gridforce.z_levels, or provide a variable 'zlevels' in the "
                "grid file or the forcing files."
            )
        if n > nz:
            raise ValueError(f"Number of z levels ({n}) exceeds the number of "
                             f"levels in the forcing files ({nz})")
        if n < nz:
            if config["gridforce"].get("z_levels") is None:
                raise ValueError(f"Number of z levels ({n}) is less than the number "
                                 f"of levels in the forcing files ({nz})")
            logger.info(f"Using the first {n} of {nz} levels in the forcing files")
