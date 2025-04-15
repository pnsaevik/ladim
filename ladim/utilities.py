"""
General utilities for LADiM
"""

from typing import Any, Dict, List
import numpy as np


def timestep2stamp(config: Dict[str, Any], n: int) -> np.datetime64:
    """Convert from time step number to timestamp"""
    timestamp = config["start_time"] + n * np.timedelta64(config["dt"], "s")
    return timestamp


def timestamp2step(config: Dict[str, Any], timestamp: np.datetime64) -> int:
    """Convert from timestamp to time step number"""
    # mtime = np.datetime64(timestamp)
    dtime = np.timedelta64(timestamp - config["start_time"], "s").astype(int)
    step = dtime // config["dt"]
    return step


# Utility function to test for position in grid
def ingrid(x: float, y: float, subgrid: List[int]) -> bool:
    """Check if position (x, y) is in a subgrid"""
    i0, i1, j0, j1 = subgrid
    return (i0 <= x) & (x <= i1 - 1) & (j0 <= y) & (y <= j1 - 1)


def read_timedelta(conf) -> np.timedelta64:
    time_value, time_unit = conf
    return np.timedelta64(time_value, time_unit)


def load_class(name):
    import importlib.util
    import sys
    from pathlib import Path

    pkg, cls = name.rsplit(sep='.', maxsplit=1)

    # Check if "pkg" is an existing file
    spec = None
    module_name = None
    file_name = pkg + '.py'
    if Path(file_name).exists():
        # This can return None if there were import errors
        module_name = pkg
        spec = importlib.util.spec_from_file_location(module_name, file_name)

    # If pkg can not be interpreted as a file, use regular import
    if spec is None:
        return getattr(importlib.import_module(pkg), cls)

    # File import
    else:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, cls)
