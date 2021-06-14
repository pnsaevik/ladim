# ------------------------------------
# tracker.py
# Part of the LADiM Model
#
# Bjørn Ådlandsvik, <bjorn@imr.no>
# Institute of Marine Research
#
# Licensed under the MIT license
# ------------------------------------

import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np  # type:ignore
import numba  # type: ignore

# from .state import State
from ladim2.forcing import BaseForce

# from .grid import BaseGrid


Velocity = Tuple[np.ndarray, np.ndarray]

PARALLEL = False
DEBUG = False

logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)


class Tracker:
    """The physical particle tracking kernel"""

    def __init__(
        self,
        advection: str,  # EF, RK2, or RK4
        diffusion: float = 0.0,
        vertdiff: float = 0.0,
        vertical_advection: bool = False,
        modules: Optional[Dict[str, Any]] = None,
    ) -> None:

        logger.info("Initiating the particle tracker")
        self.modules = modules
        self.dt = modules["time"].dt / np.timedelta64(1, "s")
        self.advection = advection  # Name of advection method
        logger.info("  Advection method: %s", advection)

        # advect <- requested advection method
        # advection = string "EF", "RK2", "RK4"
        # advect = the actual method
        if self.advection:
            self.advect = getattr(self, self.advection)

        self.vertical_advection = vertical_advection
        if vertical_advection:
            logger.info("  Vertical advection activated")

        self.diffusion = bool(diffusion)
        self.D = diffusion
        if self.diffusion:
            logger.info("  Horizontal diffusion: %s m²/s", diffusion)

        self.vertdiff = bool(vertdiff)
        self.Dz = vertdiff
        if self.vertdiff:
            logger.info("  Vertical diffusion: %s m²/s", vertdiff)

    def update(self) -> None:
        """Move the particles one time step"""

        state = self.modules["state"]
        grid = self.modules["grid"]
        force = self.modules["forcing"]

        X, Y, Z = state.X, state.Y, state.Z

        self.dx, self.dy = grid.metric(X, Y)

        self.xmin = grid.xmin + 0.01
        self.xmax = grid.xmax - 0.01
        self.ymin = grid.ymin + 0.01
        self.ymax = grid.ymax - 0.01

        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # --- Advection ---
        if self.advection:
            Uadv, Vadv = self.advect(X, Y, Z, force)
            U += Uadv
            V += Vadv

        # --- Diffusion ---
        if self.diffusion:
            Udiff, Vdiff = self.diffuse(num_particles=len(X))
            U += Udiff
            V += Vdiff

        # --- Move the particles

        # New position, if OK
        X1 = X + U * self.dt / self.dx
        Y1 = Y + V * self.dt / self.dy

        # Kill particles trying to move out of the grid
        out_of_grid = ~grid.ingrid(X1, Y1)
        state.alive[out_of_grid] = False
        state.active[out_of_grid] = False  # Not necessary if they are removed

        # Do not move inactive particles
        inactive = ~state.active
        X1[inactive] = X[inactive]
        Y1[inactive] = Y[inactive]

        # Land, boundary treatment. Do not move the particles onto land
        # Consider a sequence of different actions
        onland = ~grid.atsea(X1, Y1)
        X1[onland] = X[onland]
        Y1[onland] = Y[onland]

        # Update the particle positions
        state["X"] = X1
        state["Y"] = Y1

        # --- Vertical movement ---

        # Sample the depth level
        h = None
        if self.vertdiff or self.vertical_advection:
            # if hasattr(grid, "sample_depth") and callable(grid.sample_depth):
            #     h = grid.sample_depth(X, Y)
            # elif hasattr(grid, "depth") and callable(grid.depth):
            #     h = grid.depth(X, Y)
            h = grid.depth(X, Y)

            # Diffusion
            if self.vertdiff:
                W = self.diffuse_vert(num_particles=len(X))
                Z += W * self.dt

            # Advection
            if self.vertical_advection:
                W = force.variables["w"]
                Z += W * self.dt

            # Reflexive boundary conditions at surface
            Z[Z < 0] *= -1

            # Reflexive boundary conditions at bottom
            if h is not None:
                below_seabed = Z > h
                Z[below_seabed] = 2 * h[below_seabed] - Z[below_seabed]

            # Update particle positions
            state["Z"] = Z

    def EF(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, force: BaseForce
    ) -> Velocity:
        """Euler-Forward advective velocity"""

        U, V = force.velocity(X, Y, Z)

        return U, V

    def RK2(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, force: BaseForce
    ) -> Velocity:

        """Runge-Kutta second order = Heun scheme

        This version does not sample velocities outside the grid
        """

        dt = self.dt
        dtdx = dt / self.dx
        dtdy = dt / self.dy

        U, V = force.velocity(X, Y, Z)
        X1, Y1 = RKstep(X, Y, U, V, 0.5, dtdx, dtdy)
        clip(X, Y, self.xmin, self.xmax, self.ymin, self.ymax)

        return force.velocity(X1, Y1, Z, fractional_step=0.5)

    def RK4(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, force: BaseForce
    ) -> Velocity:
        """Runge-Kutta fourth order advection

        This version does not sample velocities outside the grid

        """

        dt = self.dt
        dtdx = dt / self.dx
        dtdy = dt / self.dy
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax

        U1, V1 = force.velocity(X, Y, Z, fractional_step=0.0)
        X1, Y1 = RKstep(X, Y, U1, V1, 0.5, dtdx, dtdy)
        clip(X1, Y1, xmin, xmax, ymin, ymax)

        U2, V2 = force.velocity(X1, Y1, Z, fractional_step=0.5)
        X2, Y2 = RKstep(X, Y, U2, V2, 0.5, dtdx, dtdy)
        clip(X2, Y2, xmin, xmax, ymin, ymax)

        U3, V3 = force.velocity(X2, Y2, Z, fractional_step=0.5)
        X3, Y3 = RKstep(X, Y, U3, V3, 1.0, dtdx, dtdy)
        clip(X3, Y3, xmin, xmax, ymin, ymax)

        U4, V4 = force.velocity(X3, Y3, Z, fractional_step=1.0)

        return RK4avg(U1, U2, U3, U4), RK4avg(V1, V2, V3, V4)

    def diffuse(self, num_particles: int) -> Velocity:
        """Random walk diffusion"""

        # Diffusive velocity
        stddev = (2 * self.D / self.dt) ** 0.5
        U = stddev * np.random.normal(size=num_particles)
        V = stddev * np.random.normal(size=num_particles)

        return U, V

    def diffuse_vert(self, num_particles: int):
        """Random walk diffusion"""

        # Diffusive velocity
        stddev = (2 * self.Dz / self.dt) ** 0.5
        W = stddev * np.random.normal(size=num_particles)

        return W


@numba.njit(parallel=PARALLEL)
def RKstep0(X, Y, U, V, frac, dtdx, dtdy):
    Xp = X + frac * U * dtdx
    Yp = Y + frac * V * dtdy
    return Xp, Yp


@numba.njit(parallel=False)
def RKstep1(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    frac: float,
    dtdx: np.ndarray,
    dtdy: np.ndarray,
) -> Velocity:
    """Do a forward Runge-Kutta (partial) step"""

    N = len(X)
    Xp = np.empty(N)
    Yp = np.empty(N)
    for i in numba.prange(N):
        Xp[i] = X[i] + frac * U[i] * dtdx[i]
        Yp[i] = Y[i] + frac * V[i] * dtdy[i]
    return X, Y


RKstep = RKstep1


@numba.njit(parallel=PARALLEL)
def clip(
    X: np.ndarray, Y: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float
) -> None:
    """Clip particle positions in place towards forcing domain boundary"""
    for p in numba.prange(len(X)):
        X[p] = max(min(X[p], xmax), xmin)
        Y[p] = max(min(Y[p], ymax), ymin)


@numba.njit(parallel=PARALLEL)
def RK4avg(
    U1: np.ndarray, U2: np.ndarray, U3: np.ndarray, U4: np.ndarray
) -> np.ndarray:
    """Average velocity component for Runge-Kutta 4-th order"""
    return (U1 + 2 * U2 + 2 * U3 + U4) / 6.0
