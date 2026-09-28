"""
ROMS forcing: velocity and scalar fields sampled at particle positions.

Combines the coordinate conversion of :mod:`.coords` (particle position ->
fractional grid indices) with the chunk loader (interpolation of the forcing
fields, which are never fully loaded).

Time interpolation
------------------
Model time step t lies in the forcing interval [t0, t1) of two consecutive
forcing frames. Velocities are linearly interpolated in time between the
frames.

Scalar fields (``field``) are not interpolated in time. They are taken from
the latest forcing frame at or before t (the first frame if t is before it).

Prefetching
-----------
At every update, the chunks that the particles currently use are requested
for the next forcing frame, so that they are decoded in the background.
"""

import glob
import logging

import numpy as np

from .. import parallel
from ..chunkloader import Dataset, read_times
from .grid import Grid

logger = logging.getLogger(__name__)


class Forcing:
    """
    Class for ROMS forcing
    """

    def __init__(self, config, _=None):
        logger.info("Initiating forcing")

        self._grid = grid = Grid(config)
        gconf = config["gridforce"]
        self.ibm_forcing = list(config.get("ibm_forcing", []) or [])
        self.has_been_initialized = False

        files = self.find_files(gconf)
        if not files:
            logger.error("No input file: {}".format(gconf["input_file"]))
            raise SystemExit(3)
        logger.info("Number of forcing files = {}".format(len(files)))
        num_threads = parallel.configure(gconf.get("num_threads"))

        # Times of all forcing frames
        times = []
        for fname in files:
            logger.info(f"Open forcing file {fname}")
            times.append(read_times(fname, "ocean_time"))
        all_frames = np.concatenate(times)
        _check_sorted(all_frames)
        logger.info(f"Number of available forcing times = {len(all_frames)}")

        self._dset = Dataset(files, "ocean_time", num_threads=num_threads, times=times)
        self._set_steps(config, all_frames)

        # Forcing variables
        self._u = self._dset["u"]
        self._v = self._dset["v"]
        self._fields = {}

        # Land masks at u- and v-points, global index space, zero outside
        # the subgrid (velocity is zero on land)
        self._mask_u = np.zeros(self._u.shape[-2:], dtype=np.uint8)
        self._mask_u[grid.Ju, grid.Iu] = grid.Mu
        self._mask_v = np.zeros(self._v.shape[-2:], dtype=np.uint8)
        self._mask_v[grid.Jv, grid.Iv] = grid.Mv

        self._t = None
        self._dense = {}

    def _set_steps(self, config, all_frames):
        """Model time step of each forcing frame, and coverage checks"""
        time0, time1 = all_frames[0], all_frames[-1]
        logger.info(f"First forcing time = {time0}")
        logger.info(f"Last forcing time = {time1}")
        start_time = np.datetime64(config["start_time"])
        stop_time = np.datetime64(config["stop_time"])
        if time0 > start_time:
            logger.error("No forcing at start time")
            raise SystemExit(3)
        if time1 < stop_time:
            logger.error("No forcing at stop time")
            raise SystemExit(3)

        seconds = (all_frames - start_time) / np.timedelta64(1, "s")
        steps = np.trunc(seconds / float(config["dt"])).astype(np.int64)
        self._steps = steps
        self.steps = steps.tolist()
        self.stepdiff = np.diff(steps)

    @staticmethod
    def find_files(force_config):
        """Find (and sort) the forcing file(s)"""
        # Use unix-style filenames to provide consistency between windows and linux
        first_file = force_config.get("first_file", "").replace("\\", "/")
        last_file = force_config.get("last_file", "").replace("\\", "/")
        files = sorted(f.replace("\\", "/") for f in glob.glob(force_config["input_file"]))
        if first_file:
            files = [f for f in files if f >= first_file]
        if last_file:
            files = [f for f in files if f <= last_file]
        return files

    # ---------- Time stepping ----------

    def _remaining_initialization(self):
        self._set_time(-1)
        self.has_been_initialized = True

    def update(self, t):
        """Update the fields to time step t"""
        logger.debug("Updating forcing, time step = {}".format(t))
        if not self.has_been_initialized:
            self._remaining_initialization()
        self._set_time(t)

    def _set_time(self, t):
        t = int(t)
        if t == self._t:
            return
        steps = self._steps
        n = int(np.searchsorted(steps, t, side="right")) - 1
        self._n0 = min(max(n, 0), len(steps) - 2)  # Velocity interval
        self._nf = min(max(n, 0), len(steps) - 1)  # Scalar field frame
        if n >= 0 and (self._t is None or n != self._n):
            logger.info("Forcing frame {} at time step {}".format(n, steps[n]))
        self._n = n
        self._t = t
        self._dense.clear()
        self._prefetch()

    def _velocity_frames(self, tstep=0.0):
        """(n0, n1, w1): Interpolation weight of the forcing frames"""
        n0 = self._n0
        t0, t1 = self._steps[n0], self._steps[n0 + 1]
        return n0, n0 + 1, (self._t + tstep - t0) / (t1 - t0)

    def _prefetch(self):
        """Request the chunks currently in use for the next forcing frames"""
        n0 = self._n0
        if n0 + 2 < len(self._steps):
            for var in (self._u, self._v):
                need = var.loaded(n0) | var.loaded(n0 + 1)
                if need.any():
                    var.prefetch(n0 + 2, need)
        nf = self._nf
        if nf + 1 < len(self._steps):
            for var in self._fields.values():
                need = var.loaded(nf)
                if need.any():
                    var.prefetch(nf + 1, need)

    # ---------- Sampling ----------

    def velocity(self, X, Y, Z, tstep=0, method="bilinear"):
        """Velocity components at particle positions and time t + tstep

        :param method: "bilinear" (linear interpolation in all dimensions) or
            "nearest" (values at the nearest u- and v-points, linear in time)
        """
        grid = self._grid
        frames = self._velocity_frames(tstep)
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if method == "bilinear":
            k = grid.scoord.level(X, Y, Z)
            iu, ju = X - 0.5, Y
            iv, jv = X, Y - 0.5
        else:
            k, J, I = grid.scoord.cell(X, Y, Z)
            # Nearest u- and v-points, rounded in subgrid coordinates
            iu = np.clip(np.rint(X - grid.i0 + 0.5), 0, grid.imax).astype(np.int64)
            iu += grid.i0 - 1
            jv = np.clip(np.rint(Y - grid.j0 + 0.5), 0, grid.jmax).astype(np.int64)
            jv += grid.j0 - 1
            ju, iv = J, I
        u = self._u.interp([k, ju, iu], frames, linear="tzyx", mask=self._mask_u)
        v = self._v.interp([k, jv, iv], frames, linear="tzyx", mask=self._mask_v)
        return u, v

    def field(self, X, Y, Z, name):
        """Scalar field in the grid cell of the particles

        The value at the nearest rho point, at the s-level just above the
        particle, without interpolation, from the latest forcing frame.
        """
        K, J, I = self._grid.scoord.cell(X, Y, Z)
        return self._field(name).interp([K, J, I], self._nf)

    def _field(self, name):
        var = self._fields.get(name)
        if var is None:
            var = self._fields[name] = self._dset[name]
        return var

    # ---------- Dense arrays (for code that uses the full fields) ----------

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails
        if name.startswith("_"):
            raise AttributeError(name)
        names = ["U", "V"] + self.ibm_forcing
        if name in names:
            base = name
        elif name[:1] == "d" and name[1:] in names:
            base = name[1:]
        elif name[-3:] == "new" and name[:-3] in names:
            base = name[:-3]
        else:
            raise AttributeError(name)
        if self._t is None:
            self._remaining_initialization()
        value = self._dense.get(name)
        if value is None:
            value = self._dense[name] = self._dense_array(name, base)
        return value

    def _dense_array(self, name, base):
        """Subgrid array of U, V, an IBM field, its next frame ('new') or its
        time derivative per time step ('d')"""
        grid = self._grid
        n0 = self._n0
        dt = self._steps[n0 + 1] - self._steps[n0]
        if base in ("U", "V"):
            var, region, mask = (
                (self._u, (slice(None), grid.Ju, grid.Iu), grid.Mu) if base == "U"
                else (self._v, (slice(None), grid.Jv, grid.Iv), grid.Mv)
            )
            now = self._velocity_frames()
        else:
            var = self._field(base)
            region = (slice(None),) * (len(var.shape) - 2) + (grid.J, grid.I)
            mask = None
            now = self._nf

        def frame(f):
            F = var.read(f, region)
            return F * mask.astype(F.dtype) if mask is not None else F

        if name == "d" + base:
            return (frame(n0 + 1) - frame(n0)) / dt
        if name == base + "new":
            if base in ("U", "V"):
                return frame(n0 + 1)
            return frame(min(self._nf + 1, len(self._steps) - 1))
        return frame(now)

    # Allow item notation
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def close(self):
        self._dset.close()


def _check_sorted(all_frames):
    """Check that time frames are strictly sorted"""
    bad = np.flatnonzero(all_frames[1:] <= all_frames[:-1])
    if bad.size:
        i = bad[0] + 1  # Index of first out-of-order frame
        oooframe = str(all_frames[i]).split(".")[0]  # Remove microseconds
        logger.info(f"Time frame {i} = {oooframe} out of order")
        logger.critical("Forcing time frames not strictly sorted")
        raise SystemExit(4)
