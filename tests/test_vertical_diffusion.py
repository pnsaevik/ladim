import numpy as np
import pytest

from ladim import vertical_diffusion
from ladim.vertical_diffusion import VerticalDiffusor

DT = 600.0

# Particle set: (depth Z, bottom depth, random increment in units of sqrt(dt))
#
# Covers particles near the surface and the bottom, which fold back (also
# several times, in the shallow columns), particles in the middle of the water
# column, particles not at the centre of a 5 m depth bin (or exactly at the
# edge of one), no random step, and positions without bottom depth (not moved).
PARTICLES = np.array([
    # Z       depth  increment
    (0.00,    50.0,  +0.5),
    (0.30,    50.0,  -1.0),  # Folds at the surface
    (2.50,    50.0,  -1.7),  # Folds at the surface, at a bin center
    (5.00,    50.0,  -0.2),  # At a bin edge
    (7.10,    50.0,  +1.2),
    (12.00,   50.0,  +0.0),  # No random step
    (25.37,   50.0,  +0.9),
    (37.90,   50.0,  -1.3),
    (49.80,   50.0,  +0.3),  # Folds at the bottom
    (49.99,   50.0,  -0.4),
    (7.00,     8.0,  +1.7),  # Folds at the bottom, shallow column
    (3.30,    12.0,  -1.7),
    (1.00,     2.0,  +1.7),  # Folds several times
    (1.40,     2.0,  -1.1),  # Folds several times
    (30.00,  np.nan, +1.0),  # No bottom depth
    (4.00,     0.0,  +1.0),  # No bottom depth
])
Z0, DEPTH, C = PARTICLES.T
DW = C * DT ** 0.5


@pytest.fixture
def fixed_increments(monkeypatch):
    """Use the increments above instead of random numbers"""
    monkeypatch.setattr(vertical_diffusion, "_increments", lambda pid, dt: DW)


def diffuse(diffusor, **kwargs):
    n = len(Z0)
    return diffusor.diffuse(np.zeros(n), np.zeros(n), Z0, DEPTH,
                            np.arange(n), DT, **kwargs)


def test_constant_diffusion(fixed_increments):
    # Diffusivity 0.01 m2/s: A step of one increment is sqrt(2 * 0.01 * 600) = 3.464 m
    expected = [
        1.732051,   # 0.0 + 0.5 * 3.464
        3.164102,   # 0.3 - 3.464 = -3.164, reflected at the surface
        3.388973,   # 2.5 - 1.7 * 3.464 = -3.389, reflected at the surface
        4.307180,   # 5.0 - 0.2 * 3.464
        11.256922,
        12.000000,  # No random step
        28.487691,
        33.396668,
        49.160770,  # 49.8 + 1.039 = 50.839, reflected at the bottom (50 m)
        48.604359,
        3.111027,   # 7.0 + 5.889 = 12.889, reflected at the bottom (8 m)
        2.588973,
        1.111027,   # 1.0 + 5.889 = 6.889, reflected at the bottom (2 m), the
                    # surface, and the bottom again
        1.589488,
        30.000000,  # No bottom depth, not moved
        4.000000,   # No bottom depth, not moved
    ]
    result = diffuse(VerticalDiffusor.create(0.01))
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_labolle_diffusion(fixed_increments):
    # The diffusivity of the ocean model decreases with depth, 0.1 * exp(-z / 5)
    # m2/s, and is sampled at the midpoints of 5 m bins (2.5 m, 7.5 m, ...).
    # It is clipped to the interval [1e-4, 3e-2] m2/s, so the diffusivity is
    # 0.03 m2/s in the upper 5 m (a step of one increment is 6 m), and
    # 0.0001 m2/s below 35 m (a step of one increment is 0.346 m).
    expected = [
        3.000000,   # 0.0 + 0.5 * 6, in the upper bin before and after
        4.874516,
        6.296678,
        3.800000,   # At a bin edge: First sampled at 7.5 m (0.0223 m2/s) which
                    # gives 3.965, then at 2.5 m (0.03 m2/s): 5.0 - 0.2 * 6
        10.866203,
        12.000000,  # No random step
        26.000265,  # Very low diffusivity (0.0004 m2/s), small step
        37.449667,  # Lowest diffusivity (0.0001 m2/s), step of -1.3 * 0.346
        49.903923,  # Lowest diffusivity, so it stays close to the bottom
        49.851436,
        1.200000,   # 7.0 + 1.7 * 6 = 17.2, reflected at the bottom (8 m) and the surface
        5.496678,
        0.800000,
        1.200000,
        30.000000,  # No bottom depth, not moved
        4.000000,   # No bottom depth, not moved
    ]
    spec = dict(method="labolle", min=1e-4, max=3e-2, resolution=5)
    result = diffuse(VerticalDiffusor.create(spec),
                     vert_mix=lambda x, y, z: 0.1 * np.exp(-z / 5))
    np.testing.assert_allclose(result, expected, atol=1e-5)
