"""
General utilities for LADiM
"""

import numpy as np


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
        spec.loader.exec_module(module)  # type: ignore
        return getattr(module, cls)
