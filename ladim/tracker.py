import numba
import numpy as np
from numba import prange
import typing
from collections import namedtuple
if typing.TYPE_CHECKING:
    from .model import Model


VertMix = namedtuple("VertMix", ["method", "value", "min", "max", "resolution"])
VertMix.__doc__ = """Specification of the vertical diffusion

``method`` is "constant" (diffusivity ``value`` [m2/s]) or "labolle"
(diffusivity from the forcing, clipped to [``min``, ``max``] and sampled at
the midpoints of depth bins of height ``resolution`` [m], or at the particle
depth if ``resolution`` is infinite). Fields that do not apply to the method
are None."""

VERT_MIX_METHODS = ("constant", "labolle")


def parse_vert_mix(spec):
    """Vertical diffusion specification from the ``numerics.vert_mix`` config

    :param spec: One of

        * None or a number: Constant diffusivity [m2/s], same as
          ``{method: constant, value: <number>}``. Zero means no vertical
          diffusion.
        * "labolle": Diffusivity from the forcing, same as
          ``{method: labolle, min: 0, max: inf, resolution: inf}``.
        * A dict with the keys ``method`` and the fields of the method, with
          no default values. The "constant" method has the field ``value``.
          The "labolle" method has the fields ``min`` and ``max`` (the limits
          of the diffusivity [m2/s]) and ``resolution`` (the height [m] of
          the depth bins where the diffusivity is sampled, at the midpoint of
          the bin; infinite means no binning).

    :return: A VertMix tuple
    """
    if spec is None:
        spec = 0
    if isinstance(spec, str):
        if spec != "labolle":
            raise ValueError(f"Unknown vertical diffusion method: {spec!r}. "
                             f"Available strings: 'labolle'")
        spec = dict(method="labolle", min=0, max=float("inf"), resolution=float("inf"))
    elif isinstance(spec, dict):
        spec = dict(spec)
    elif isinstance(spec, (int, float)) and not isinstance(spec, bool):
        spec = dict(method="constant", value=spec)
    else:
        raise ValueError(f"Invalid vertical diffusion: {spec!r}")

    method = spec.pop("method", None)
    if method not in VERT_MIX_METHODS:
        raise ValueError(f"Unknown vertical diffusion method: {method!r}. "
                         f"Available methods: {', '.join(VERT_MIX_METHODS)}")
    fields = dict(constant=["value"], labolle=["min", "max", "resolution"])[method]
    missing = [f for f in fields if f not in spec]
    unknown = [f for f in spec if f not in fields]
    if missing:
        raise ValueError(f"Vertical diffusion method {method!r} requires "
                         f"the fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"Unknown fields for vertical diffusion method "
                         f"{method!r}: {', '.join(unknown)}")
    try:
        values = {f: float(spec[f]) for f in fields}  # Also accepts "inf"
    except (TypeError, ValueError):
        raise ValueError(f"Vertical diffusion fields must be numbers: {spec}")
    if any(np.isnan(v) for v in values.values()):
        raise ValueError(f"Vertical diffusion fields must be numbers: {spec}")

    out = dict(value=None, min=None, max=None, resolution=None)
    out.update(values)
    if method == "constant":
        if not 0 <= out["value"] < float("inf"):
            raise ValueError(f"Vertical diffusion must be non-negative and "
                             f"finite: {out['value']}")
    else:
        if not 0 <= out["min"] <= out["max"]:
            raise ValueError(f"Vertical diffusion limits must satisfy "
                             f"0 <= min <= max: {out['min']}, {out['max']}")
        if not out["resolution"] > 0:
            raise ValueError(f"Vertical diffusion resolution must be positive "
                             f"(or infinite): {out['resolution']}")
    return VertMix(method=method, **out)


class Tracker:
    """The physical particle tracking kernel

    Horizontal advection with the Euler-Forward ("EF"), midpoint ("RK2") or
    fourth order Runge-Kutta ("RK4") method, and horizontal diffusion as a
    random walk with constant diffusion coefficient. The arithmetic runs in
    multithreaded numba kernels, the velocity is sampled by the forcing module.

    Optionally, vertical diffusion is applied after the horizontal movement,
    as a random walk in the depth Z (see :func:`parse_vert_mix` for the
    options). Particles are reflected at the sea surface and at the sea bottom
    (at the bottom depth of the grid cell of the new horizontal position).

    With the "constant" method, the diffusivity is constant. With the "labolle"
    method, the diffusivity is read from the forcing (``forcing.vert_mix``),
    and the random walk uses the two-step scheme by LaBolle et al. (2000),
    doi: 10.1029/1999WR900224, which is consistent with a depth-dependent
    diffusivity: The diffusivity is first evaluated at the initial depth to
    make an intermediate step, and the final step uses the diffusivity at the
    intermediate depth. The diffusivity is clipped to [min, max], and sampled
    at the midpoints of depth bins of height ``resolution``, to avoid
    numerical errors from rapidly varying diffusivities.

    The random walk uses a counter-based generator: the random numbers of a
    particle are a hash of its particle identifier and a key drawn from
    ``np.random`` once per time step. Results are reproducible when the numpy
    random seed is set (``numerics.seed``), independent of the number of
    threads and the order of the particles. The random numbers are uniform
    with variance 1, so the random walk is Gaussian after a few time steps.
    """

    METHODS = ("EF", "RK2", "RK4")

    def __init__(self, method, diffusion, vert_mix=0):
        if method not in self.METHODS:
            raise NotImplementedError(f"Unknown integration method: {method}")
        self.method = method
        self.diffusion = diffusion  # [m2.s-1]
        self.vert_mix = parse_vert_mix(vert_mix)
        self._vertical = self.vert_mix.method != "constant" or self.vert_mix.value > 0

    @staticmethod
    def create(method, diffusion, vert_mix=0):
        return Tracker(method, diffusion, vert_mix)

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

        if self._vertical:
            Z0 = np.ascontiguousarray(Z, dtype=np.float64)
            self.diffuse_vertically(Z0, grid.sample_depth(X0, Y0), pid, dt,
                                    forcing=forcing, X=X0, Y=Y0)
            if all_active:
                state['Z'][:] = Z0
            else:
                state['Z'][act] = Z0

    def diffuse_vertically(self, Z, depth, pid, dt, forcing=None, X=None, Y=None):
        """Vertical random walk, in place

        :param Z: Particle depths [m, positive downwards], float64 array
        :param depth: Bottom depth [m] at the particle positions. Particles
            at positions without depth (zero or NaN) are not moved.
        :param pid: Particle identifiers (int64), for the random numbers
        :param dt: Time step [s]
        :param forcing: Forcing module with a function ``vert_mix(X, Y, Z)``.
            Only used by the "labolle" method.
        :param X, Y: Particle positions [grid units]. Only used by the
            "labolle" method.
        """
        # A new random key for each time step
        key = np.uint64(np.random.randint(0, 2**63, dtype=np.int64))
        depth = np.ascontiguousarray(depth, dtype=np.float64)
        vm = self.vert_mix

        if vm.method == "constant":
            _vertical(Z, depth, pid, key, (2 * vm.value * dt) ** 0.5)
            return

        # LaBolle et al. (2000): The diffusivity at the intermediate position
        # is used to compute the final step, with the same random increment
        if not hasattr(forcing, "vert_mix"):
            raise ValueError(f"The forcing module {type(forcing).__name__} has no "
                             f"vertical diffusivity, required by the vertical "
                             f"diffusion method {vm.method!r}")
        dW = np.empty(Z.size)
        _noise_z(pid, key, dt ** 0.5, dW)
        Z0 = Z.copy()

        def mixing(z):
            if np.isfinite(vm.resolution):
                z = (z // vm.resolution) * vm.resolution + 0.5 * vm.resolution
            k = np.asarray(forcing.vert_mix(X, Y, z), dtype=np.float64)
            k = np.where(np.isfinite(k), k, vm.min)
            return np.ascontiguousarray(np.clip(k, vm.min, vm.max))

        Z1 = np.empty(Z.size)
        _vertical_step(Z0, mixing(Z0), dW, depth, Z1)  # Intermediate step
        _vertical_step(Z0, mixing(Z1), dW, depth, Z)  # Final step

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


@numba.njit(inline="always")
def _splitmix64(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@numba.njit(inline="always")
def _noise_pair(key, pid):
    """Two independent random numbers from (key, pid), mean 0 and variance 1

    Uniform on [-sqrt(3), sqrt(3)), about 10 times cheaper than normal
    deviates (Box-Muller).
    """
    h1 = _splitmix64(key ^ _splitmix64(np.uint64(pid)))
    h2 = _splitmix64(h1)
    scale = 2.0 * np.sqrt(3.0) / 9007199254740992.0  # 2 * sqrt(3) / 2**53
    return (np.float64(h1 >> np.uint64(11)) * scale - np.sqrt(3.0),
            np.float64(h2 >> np.uint64(11)) * scale - np.sqrt(3.0))


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
            nx, ny = _noise_pair(key, pid[p])
            x += stddev / dx[p] * nx
            y += stddev / dy[p] * ny
        X[p] = x
        Y[p] = y


@numba.njit(parallel=True, nogil=True, cache=True)
def _vertical(Z, depth, pid, key, stddev):
    """Z += stddev * noise, reflected at the surface (0) and the bottom (depth)

    The reflection is a folding of the real line onto [0, depth], and works
    for any step length.
    """
    for p in prange(Z.size):
        d = depth[p]
        if d > 0:  # False for NaN
            noise, _ = _noise_pair(key, pid[p])
            z = Z[p] + stddev * noise
            Z[p] = d - abs(z % (2.0 * d) - d)  # Triangle wave


@numba.njit(parallel=True, nogil=True, cache=True)
def _noise_z(pid, key, scale, out):
    """out = scale * noise, with unit variance noise"""
    for p in prange(out.size):
        noise, _ = _noise_pair(key, pid[p])
        out[p] = scale * noise


@numba.njit(parallel=True, nogil=True, cache=True)
def _vertical_step(Z0, K, dW, depth, Z):
    """Z = Z0 + sqrt(2 K) * dW, reflected at the surface and the bottom

    Particles where the depth is not positive (or NaN) are not moved.
    """
    for p in prange(Z.size):
        d = depth[p]
        if d > 0:  # False for NaN
            z = Z0[p] + np.sqrt(2.0 * K[p]) * dW[p]
            Z[p] = d - abs(z % (2.0 * d) - d)  # Triangle wave
        else:
            Z[p] = Z0[p]


@numba.njit(parallel=True, nogil=True, cache=True)
def _accept(X, Y, X1, Y1, ok):
    """Move the particles where ok is True"""
    for p in prange(X.size):
        if ok[p]:
            X[p] = X1[p]
            Y[p] = Y1[p]
