"""
Vertical diffusion of particles

A vertical diffusor is created from a specification (the ``vertical_diffusion``
part of the tracker configuration) with :meth:`VerticalDiffusor.create`::

    diffusor = VerticalDiffusor.create(method="constant", value=0.001)
    Z1 = diffusor.diffuse(X, Y, Z, depth, pid, dt)

The random walk takes place in the depth Z (positive downwards). Particles are
reflected at the sea surface (Z = 0) and at the sea bottom (Z = depth).
"""

import inspect

import numba
import numpy as np
from numba import prange

from .noise import noise_pair


class VerticalDiffusor:
    """Base class for vertical random walks

    :ivar enabled: False if the diffusor never moves any particles, so that
        the caller can skip it.
    """

    enabled = True

    @staticmethod
    def create(method, **kwargs):
        """Create a vertical diffusor

        :param method: The name of the method, "constant" or "labolle"
        :param kwargs: The fields of the method. All fields are mandatory, there
            are no default values. Besides the fields of the method,
            ``vert_mix`` is the function ``vert_mix(X, Y, Z)`` which returns
            the vertical diffusivity [m2/s] at the particle positions (it is
            supplied automatically by the model, from the forcing). Fields
            which are numbers may also be given as strings like "inf".
        :raises ValueError: If the method is unknown, the fields are missing
            or unknown, or the field values are not valid
        """
        try:
            cls = _METHODS[method]
        except (KeyError, TypeError):
            raise ValueError(
                f"Unknown vertical diffusion method: {method!r}. "
                f"Available methods: {', '.join(_METHODS)}"
            ) from None
        try:
            inspect.signature(cls).bind(**kwargs)
        except TypeError as err:
            raise ValueError(f"Vertical diffusion method {method!r}: {err}") from None
        return cls(**kwargs)

    def diffuse(self, X, Y, Z, depth, pid, dt):
        """Vertical random walk for one time step

        :param X: Particle X positions [grid units]
        :param Y: Particle Y positions [grid units]
        :param Z: Particle depths [m, positive downwards]
        :param depth: Bottom depth [m] at the particle positions. Particles
            at positions without depth (zero or NaN) are not moved.
        :param pid: Particle identifiers (int64), for the random numbers
        :param dt: Time step [s]
        :return: New particle depths [m], a float64 array
        """
        raise NotImplementedError


class ConstantDiffusor(VerticalDiffusor):
    """Random walk with constant diffusivity

    :param value: The diffusivity [m2/s]. Zero means no vertical diffusion.
    :param vert_mix: Not used
    """

    def __init__(self, value, vert_mix=None):
        self.value = _number(value, "value")
        if not 0 <= self.value < np.inf:
            raise ValueError(f"The diffusivity must be non-negative and finite: "
                             f"{self.value}")
        self.enabled = self.value > 0

    def diffuse(self, X, Y, Z, depth, pid, dt):
        Z0 = np.array(Z, dtype=np.float64)
        if not self.enabled:
            return Z0
        dW = _increments(pid, dt)
        K = np.full(Z0.size, self.value)
        return _step(Z0, K, dW, depth)


class LaBolleDiffusor(VerticalDiffusor):
    """Random walk with diffusivity from the forcing

    Uses the two-step scheme by LaBolle et al. (2000),
    doi: 10.1029/1999WR900224, which is consistent with a depth-dependent
    diffusivity: The diffusivity is first evaluated at the initial depth to
    make an intermediate step. The final step starts from the initial depth,
    with the same random increment, and uses the diffusivity at the
    intermediate depth.

    The diffusivity is clipped to [min, max] (invalid values are replaced by
    min), and sampled at the midpoint of the depth bin of the particle, to
    avoid numerical errors from rapidly varying diffusivities.

    :param min: Lower limit of the diffusivity [m2/s]
    :param max: Upper limit of the diffusivity [m2/s]
    :param resolution: The height [m] of the depth bins. The diffusivity is
        sampled at the midpoints of the bins, which are at depths
        resolution / 2, 3 * resolution / 2, ... Infinity means no bins, so
        that the diffusivity is sampled at the particle depth.
    :param vert_mix: Function ``vert_mix(X, Y, Z)`` returning the diffusivity
        [m2/s] at the particle positions
    """

    def __init__(self, min, max, resolution, vert_mix):
        self.min = _number(min, "min")
        self.max = _number(max, "max")
        self.resolution = _number(resolution, "resolution")
        if not 0 <= self.min <= self.max:
            raise ValueError(f"The diffusivity limits must satisfy "
                             f"0 <= min <= max: {self.min}, {self.max}")
        if not self.resolution > 0:
            raise ValueError(f"The resolution must be positive (or infinite): "
                             f"{self.resolution}")
        if not callable(vert_mix):
            raise ValueError("vert_mix must be a function of X, Y, Z")
        self.vert_mix = vert_mix

    def diffuse(self, X, Y, Z, depth, pid, dt):
        Z0 = np.array(Z, dtype=np.float64)
        dW = _increments(pid, dt)
        Z1 = _step(Z0, self._diffusivity(X, Y, Z0), dW, depth)  # Intermediate
        return _step(Z0, self._diffusivity(X, Y, Z1), dW, depth)  # Final

    def _diffusivity(self, X, Y, Z):
        """Clipped diffusivity, sampled at the midpoints of the depth bins"""
        if np.isfinite(self.resolution):
            Z = (Z // self.resolution) * self.resolution + 0.5 * self.resolution
        K = np.asarray(self.vert_mix(X, Y, Z), dtype=np.float64)
        K = np.where(np.isfinite(K), K, self.min)
        return np.ascontiguousarray(np.clip(K, self.min, self.max))


_METHODS = dict(constant=ConstantDiffusor, labolle=LaBolleDiffusor)


def _number(value, name):
    """Convert a field to float (also accepts strings like "inf")"""
    try:
        if isinstance(value, bool):
            raise TypeError
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"The field {name!r} must be a number: {value!r}") from None
    if np.isnan(x):
        raise ValueError(f"The field {name!r} must be a number: {value!r}")
    return x


def _increments(pid, dt):
    """Random increments with zero mean and variance dt, one per particle

    The random numbers are a hash of the particle identifier and a key drawn
    from ``np.random`` for each time step, so that results are reproducible
    when the numpy random seed is set, independent of the number of threads
    and the order of the particles.
    """
    pid = np.ascontiguousarray(pid, dtype=np.int64)
    key = np.uint64(np.random.randint(0, 2**63, dtype=np.int64))
    dW = np.empty(pid.size)
    _noise(pid, key, dt ** 0.5, dW)
    return dW


def _step(Z0, K, dW, depth):
    """New depth Z0 + sqrt(2 K) * dW, reflected at the surface and the bottom"""
    depth = np.ascontiguousarray(depth, dtype=np.float64)
    Z = np.empty(Z0.size)
    _reflected_step(Z0, K, dW, depth, Z)
    return Z


@numba.njit(parallel=True, nogil=True, cache=True)
def _noise(pid, key, scale, out):
    """out = scale * noise, with unit variance noise"""
    for p in prange(out.size):
        noise, _ = noise_pair(key, pid[p])
        out[p] = scale * noise


@numba.njit(parallel=True, nogil=True, cache=True)
def _reflected_step(Z0, K, dW, depth, Z):
    """Z = Z0 + sqrt(2 K) * dW, folded onto [0, depth]

    The reflection at the surface and the bottom is a folding of the real
    line onto [0, depth] (a triangle wave), which works for any step length.
    Particles where the depth is not positive (or NaN) are not moved.
    """
    for p in prange(Z.size):
        d = depth[p]
        if d > 0:  # False for NaN
            z = Z0[p] + np.sqrt(2.0 * K[p]) * dW[p]
            Z[p] = d - abs(z % (2.0 * d) - d)
        else:
            Z[p] = Z0[p]
