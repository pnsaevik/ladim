"""
Chunk loader: fast, lazy, multithreaded access to gridded netCDF data.

* :class:`Dataset` joins the time frames of a series of files.
* :class:`Variable` interpolates a variable at arbitrary fractional,
  grid-native coordinates (:meth:`Variable.interp`), loading the required
  chunks on demand. Chunks can also be prefetched in the background.

HDF5 chunks are read and decoded directly (see :mod:`.codec`) by a pool of
threads; the interpolation kernels are numba-parallel (see :mod:`.kernels`).
The number of threads is set by ``LADIM_NUM_THREADS`` (see
:mod:`ladim.gridforce.parallel`).

This package knows nothing about ROMS; the model specific parts are in
:mod:`ladim.gridforce.roms`.
"""

from .dataset import Dataset, Variable, read_times, to_datetime64  # noqa: F401
