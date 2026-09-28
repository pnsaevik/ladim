import numba
import numpy as np
from numba import prange
import typing
if typing.TYPE_CHECKING:
    from .model import Model


class Tracker:
    """The physical particle tracking kernel

    Horizontal advection with the Euler-Forward ("EF"), midpoint ("RK2") or
    fourth order Runge-Kutta ("RK4") method, and horizontal diffusion as a
    random walk with constant diffusion coefficient. The arithmetic runs in
    multithreaded numba kernels, the velocity is sampled by the forcing module.
    """

    METHODS = ("EF", "RK2", "RK4")

    def __init__(self, method, diffusion):
        if method not in self.METHODS:
            raise NotImplementedError(f"Unknown integration method: {method}")
        self.method = method
        self.diffusion = diffusion  # [m2.s-1]

    @staticmethod
    def create(method, diffusion):
        return Tracker(method, diffusion)

    def update(self, model: "Model"):
        state = model.state
        grid = model.grid
        forcing = model.forcing
        dt = model.solver.step

        act = state['active']
        all_active = bool(np.all(act))
        if all_active:
            X, Y, Z = state['X'], state['Y'], state['Z']
        else:
            X, Y, Z = state['X'][act], state['Y'][act], state['Z'][act]
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

        X1, Y1 = self.integrate(velocity, X0, Y0, dx, dy, dt)

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

    def integrate(self, velocity, X0, Y0, dx, dy, dt):
        """New positions after one time step

        :param velocity: Function (x, y, tstep) -> (u, v), velocity [m/s] at
            positions x, y and time t0 + tstep * dt
        :param X0, Y0: Initial positions [grid units], float64 arrays
        :param dx, dy: Grid spacing [m] at the initial positions
        :param dt: Time step [s]
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

        # Random walk
        if self.diffusion:
            noise = np.random.normal(size=2 * n)
            NX, NY = noise[:n], noise[n:]
        else:
            NX = NY = np.empty(0)
        stddev = (2 * self.diffusion) ** 0.5 * dt ** 0.5

        scale = dt / 6.0 if self.method == "RK4" else dt
        _final(X0, Y0, U, V, dx, dy, AX, AY, self.method == "RK4", scale,
               NX, NY, stddev, bool(self.diffusion), X1, Y1)
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
def _final(X0, Y0, U, V, dx, dy, AX, AY, accumulate, scale, NX, NY, stddev,
           diffuse, X, Y):
    """Final position X = X0 + scale * (AX + U / dx) + stddev / dx * NX"""
    for p in prange(X0.size):
        ux = U[p] / dx[p]
        vy = V[p] / dy[p]
        if accumulate:
            ux += AX[p]
            vy += AY[p]
        x = X0[p] + scale * ux
        y = Y0[p] + scale * vy
        if diffuse:
            x += stddev / dx[p] * NX[p]
            y += stddev / dy[p] * NY[p]
        X[p] = x
        Y[p] = y


@numba.njit(parallel=True, nogil=True, cache=True)
def _accept(X, Y, X1, Y1, ok):
    """Move the particles where ok is True"""
    for p in prange(X.size):
        if ok[p]:
            X[p] = X1[p]
            Y[p] = Y1[p]
