"""Useful functions for analytically defined current fields in LADiM"""

from typing import Optional, Tuple, Union, Callable

import numpy as np  # type: ignore

from ladim2.state import State

ParticleArray = Union[np.ndarray, float]  # 1D array of floats, one element per particle
Velocity = Tuple[ParticleArray, ParticleArray]


def get_velocity1(
    state: State,
    sample_func: Callable[[ParticleArray, ParticleArray], Velocity],
    dt: Optional[int] = None,
) -> Velocity:
    """Euler Forward velocity sampling

    Args:
        state:
            LADiM state instance
        sample_func:
            Function, providing velocity from position
        dt:
            Optional, not used, for consistent interface with the higher order functions
    Returns:
        velocity:
            Tuple of U- and V-components

    """
    x0, y0 = state.X, state.Y
    return sample_func(x0, y0)


def get_velocity2(
    state: State,
    sample_func: Callable[[ParticleArray, ParticleArray], Velocity],
    dt: int,
    s: float = 1.0,
) -> Velocity:
    """2nd order Runge Kutta velocity sampling

    Args:
        state:
            LADiM state instance
        sample_func:
            Function, providing velocity from position
        dt:
            integer, timestep in seconds
        s:
            Scheme defining parameter, optional, default = 1 = Heun scheme
    Returns:
        velocity:
            Tuple of U- and V-components

    Different values of `s` gives different schemes:

        s = 1/2  gives midpoint
        s = 2/3  gives Ralston
        s = 1    gives Heun

    For elliptical motion, s is eliminated, all schemes gives same result.

    """
    m = 1.0 / (2 * s)
    x0, y0 = state.X, state.Y
    u0, v0 = sample_func(x0, y0)
    x1, y1 = x0 + s * dt * u0, y0 + s * dt * v0
    u1, v1 = sample_func(x1, y1)
    return (1 - m) * u0 + m * u1, (1 - m) * v0 + m * v1


def get_velocity4(
    state: State,
    sample_func: Callable[[ParticleArray, ParticleArray], Velocity],
    dt: int,
) -> Velocity:

    """4th order Runge Kutta velocity sampling

    Args:
        state:
            LADiM state instance
        sample_func:
            Function, providing velocity from position
        dt:
            integer, timestep in seconds
    Returns:
        velocity:
            Tuple of U- and V-components

    """

    x0, y0 = state.X, state.Y
    u0, v0 = sample_func(x0, y0)
    x1, y1 = x0 + 0.5 * dt * u0, y0 + 0.5 * dt * v0
    u1, v1 = sample_func(x1, y1)
    x2, y2 = x0 + 0.5 * dt * u1, y0 + 0.5 * dt * v1
    u2, v2 = sample_func(x2, y2)
    x3, y3 = x0 + dt * u2, y0 + dt * v2
    u3, v3 = sample_func(x3, y3)
    return (u0 + 2 * u1 + 2 * u2 + u3) / 6, (v0 + 2 * v1 + 2 * v2 + v3) / 6
