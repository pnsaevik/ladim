"""
Thread configuration shared by the gridforce subpackages.

The number of worker threads (numba kernels and chunk decoding) is taken from,
in order of precedence:

1. an explicit argument (e.g. ``gridforce.num_threads`` in the config),
2. the environment variable ``LADIM_NUM_THREADS``,
3. a default of 1 (single-threaded).
"""

import os

import numba

ENV_VAR = "LADIM_NUM_THREADS"


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Not available on all platforms
        return os.cpu_count() or 1


def num_threads(requested=None) -> int:
    """Resolve the number of threads to use (see module docstring)"""
    n = requested or os.environ.get(ENV_VAR) or 1
    return max(1, min(int(n), numba.config.NUMBA_NUM_THREADS))


def configure(requested=None) -> int:
    """Resolve the number of threads and apply it to numba"""
    n = num_threads(requested)
    numba.set_num_threads(n)
    return n
