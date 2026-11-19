import numba
import numpy as np
from numba import prange

from .noise import noise_pair
from .vertical_diffusion import VerticalDiffusor
import typing
if typing.TYPE_CHECKING:
    from .model import Model


class Tracker:
    """The physical particle tracking kernel

    Horizontal advection with the Euler-Forward ("EF"), midpoint ("RK2") or
    fourth order Runge-Kutta ("RK4") method, and horizontal diffusion as a
    random walk with constant diffusion coefficient. The arithmetic runs in
    multithreaded numba kernels, the velocity is sampled by the forcing module.

    Optionally, vertical diffusion is applied after the horizontal movement
    by a :class:`~ladim.vertical_diffusion.VerticalDiffusor`, which is created
    from the specification ``vertical_diffusion`` (a dict with the key
    ``method`` and the fields of the method). Particles are reflected at the
    sea surface and at the sea bottom (at the bottom depth of the grid cell
    of the new horizontal position).

    The random walk uses a counter-based generator: the random numbers of a
    particle are a hash of its particle identifier and a key drawn from
    ``np.random`` once per time step. Results are reproducible when the numpy
    random seed is set (``numerics.seed``), independent of the number of
    threads and the order of the particles. The random numbers are uniform
    with variance 1, so the random walk is Gaussian after a few time steps.
    """

    METHODS = ("EF", "RK2", "RK4")

    def __init__(self, method, diffusion, vertical_diffusion=None):
        """
        :param method: Advection method, "EF", "RK2" or "RK4"
        :param diffusion: Horizontal diffusion coefficient [m2/s]
        :param vertical_diffusion: Specification of the vertical diffusion,
            given to :meth:`VerticalDiffusor.create`. Default is no vertical
            diffusion.
        """
        if method not in self.METHODS:
            raise NotImplementedError(f"Unknown integration method: {method}")
        if vertical_diffusion is None:
            vertical_diffusion = dict(method="constant", value=0)
        self.method = method
        self.diffusion = diffusion  # [m2.s-1]
        self.diffusor = VerticalDiffusor.create(**vertical_diffusion)

    @staticmethod
    def create(method, diffusion, vertical_diffusion=None):
        return Tracker(method, diffusion, vertical_diffusion)

    def update(self, model: "Model"):
        state = model.state
        grid = model.grid
        forcing = model.forcing
        dt = model.solver.step

        act = state['active']
        all_active = bool(np.all(act))
        if all_active:
            X, Y, Z = state['X'], state['Y'], state['Z']
            pid = state['pid']
        else:
            X, Y, Z = state['X'][act], state['Y'][act], state['Z'][act]
            pid = state['pid'][act]
        if X.size == 0:
            return
        X0 = np.ascontiguousarray(X, dtype=np.float64)
        Y0 = np.ascontiguousarray(Y, dtype=np.float64)
        dx, dy = (np.ascontiguousarray(a, dtype=np.float64)
                  for a in grid.sample_metric(X0, Y0))

        def velocity(x, y, tstep):
            u, v = forcing.velocity(x, y, Z, tstep=tstep)
            return (np.ascontiguousarray(u, dtype=np.float64),
                    np.ascontiguousarray(v, dtype=np.float64))

        pid = np.ascontiguousarray(pid, dtype=np.int64)
        X1, Y1 = self.integrate(velocity, X0, Y0, dx, dy, dt, pid)

        # Land, boundary treatment. Do not move the particles
        should_move = np.ascontiguousarray(grid.atsea(X1, Y1), dtype=np.bool_)
        _accept(X0, Y0, X1, Y1, should_move)

        if all_active:
            if X0 is not state['X']:
                state['X'][:] = X0
            if Y0 is not state['Y']:
                state['Y'][:] = Y0
        else:
            state['X'][act] = X0
            state['Y'][act] = Y0

        if self.diffusor.enabled:
            depth = grid.sample_depth(X0, Y0)
            Z1 = self.diffusor.diffuse(X0, Y0, Z, depth, pid, dt)
            if all_active:
                state['Z'][:] = Z1
            else:
                state['Z'][act] = Z1

    def integrate(self, velocity, X0, Y0, dx, dy, dt, pid):
        """New positions after one time step

        :param velocity: Function (x, y, tstep) -> (u, v), velocity [m/s] at
            positions x, y and time t0 + tstep * dt
        :param X0, Y0: Initial positions [grid units], float64 arrays
        :param dx, dy: Grid spacing [m] at the initial positions
        :param dt: Time step [s]
        :param pid: Particle identifiers (int64), for the random walk
        :return: X1, Y1
        """
        n = X0.size
        X1 = np.empty(n)
        Y1 = np.empty(n)
        AX = AY = np.empty(0)
        if self.method == "EF":
            U, V = velocity(X0, Y0, 0.0)
        elif self.method == "RK2":
            U, V = velocity(X0, Y0, 0.0)
            _stage(X0, Y0, U, V, dx, dy, 0.5 * dt, X1, Y1, AX, AY, 0.0, False, False)
            U, V = velocity(X1, Y1, 0.5)
        else:  # RK4
            AX = np.empty(n)
            AY = np.empty(n)
            U, V = velocity(X0, Y0, 0.0)
            _stage(X0, Y0, U, V, dx, dy, 0.5 * dt, X1, Y1, AX, AY, 1.0, True, True)
            U, V = velocity(X1, Y1, 0.5)
            _stage(X0, Y0, U, V, dx, dy, 0.5 * dt, X1, Y1, AX, AY, 2.0, True, False)
            U, V = velocity(X1, Y1, 0.5)
            _stage(X0, Y0, U, V, dx, dy, dt, X1, Y1, AX, AY, 2.0, True, False)
            U, V = velocity(X1, Y1, 1.0)

        # Random walk, with a new random key for each time step
        key = np.uint64(0)
        if self.diffusion:
            key = np.uint64(np.random.randint(0, 2**63, dtype=np.int64))
        stddev = (2 * self.diffusion) ** 0.5 * dt ** 0.5

        scale = dt / 6.0 if self.method == "RK4" else dt
        _final(X0, Y0, U, V, dx, dy, AX, AY, self.method == "RK4", scale,
               pid, key, stddev, bool(self.diffusion), X1, Y1)
        return X1, Y1


@numba.njit(parallel=True, nogil=True, cache=True)
def _stage(X0, Y0, U, V, dx, dy, h, X, Y, AX, AY, w, accumulate, first):
    """Intermediate position X = X0 + h * u, and AX (+)= w * u

    Here u = U / dx is the velocity in grid units.
    """
    for p in prange(X0.size):
        ux = U[p] / dx[p]
        vy = V[p] / dy[p]
        X[p] = X0[p] + h * ux
        Y[p] = Y0[p] + h * vy
        if accumulate:
            if first:
                AX[p] = w * ux
                AY[p] = w * vy
            else:
                AX[p] += w * ux
                AY[p] += w * vy


@numba.njit(parallel=True, nogil=True, cache=True)
def _final(X0, Y0, U, V, dx, dy, AX, AY, accumulate, scale, pid, key, stddev,
           diffuse, X, Y):
    """Final position X = X0 + scale * (AX + U / dx) + stddev / dx * noise"""
    for p in prange(X0.size):
        ux = U[p] / dx[p]
        vy = V[p] / dy[p]
        if accumulate:
            ux += AX[p]
            vy += AY[p]
        x = X0[p] + scale * ux
        y = Y0[p] + scale * vy
        if diffuse:
            nx, ny = noise_pair(key, pid[p])
            x += stddev / dx[p] * nx
            y += stddev / dy[p] * ny
        X[p] = x
        Y[p] = y


@numba.njit(parallel=True, nogil=True, cache=True)
def _accept(X, Y, X1, Y1, ok):
    """Move the particles where ok is True"""
    for p in prange(X.size):
        if ok[p]:
            X[p] = X1[p]
            Y[p] = Y1[p]
