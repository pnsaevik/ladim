import numpy as np
import pytest

from ladim.vertical_diffusion import VerticalDiffusor

DT = 600.0  # Time step [s]
DEPTH = 50.0  # Bottom depth [m]

# Initial particle depths [m]. Covers particles near the surface and the
# bottom, which fold back, particles in the middle of the water column, and
# particles not at the centre of a 5 m depth bin (or exactly at the edge of one).
Z0 = np.array([0.00, 0.30, 2.50, 3.00, 4.00, 5.00, 7.10,
               12.00, 25.37, 37.90, 47.50, 49.80, 49.99])

# With the random seed below, the random increments (in units of sqrt(dt)) are
#   0.32, -1.055, -1.675, 1.028, -1.204, -0.13, 1.053,
#   -0.317, -0.322, 0.358, 1.402, 1.174, 0.729
SEED = 5


def profile(x, y, z):
    """Vertical diffusivity of the ocean model [m2/s]"""
    return 0.1 * np.exp(-z / 5)


@pytest.mark.parametrize("spec, expected", [
    # No diffusion: Particles stay where they are
    pytest.param(0, [
        0.00, 0.30, 2.50, 3.00, 4.00, 5.00, 7.10,
        12.00, 25.37, 37.90, 47.50, 49.80, 49.99,
    ], id="zero"),

    # Diffusivity 0.01 m2/s: One increment is a step of sqrt(2 * 0.01 * 600) =
    # 3.464 m. Particle 2: 0.3 - 1.055 * 3.464 = -3.356 is reflected at the
    # surface. Particle 11: 47.5 + 1.402 * 3.464 = 52.355 is reflected at the
    # bottom (50 m).
    pytest.param(0.01, [
        1.109878, 3.355798, 3.303655, 6.560168, 0.171425, 4.550723, 10.747532,
        10.902282, 24.255494, 39.138476, 47.645049, 46.132141, 47.486104,
    ], id="constant"),

    # The diffusivity is sampled at the midpoints of 5 m bins and clipped to
    # [1e-4, 3e-2]: It is 0.03 m2/s in the upper 5 m (one increment is a step
    # of 6 m), and 0.0001 m2/s below 35 m (a step of 0.346 m). Particle 1:
    # 0.32 * 6 = 1.92 m. Particle 12 (49.8 m) is at very low diffusivity:
    # 49.8 + 1.174 * 0.346 = 50.206 is reflected at the bottom. Particle 6 (at
    # a bin edge) is first sampled in the bin below (0.0223 m2/s), then in the
    # bin above (0.03 m2/s): 5.0 - 0.13 * 6 = 4.22. Particle 4 (3.0 m) steps to
    # 9.17 m, where the diffusivity is lower (0.0223 m2/s, one increment is
    # 5.17 m), so that the final step is 3.0 + 1.028 * 5.17 = 8.32 m.
    pytest.param(dict(method="labolle", min=1e-4, max=3e-2, resolution=5), [
        1.922365, 5.160864, 6.169234, 8.318016, 3.225119, 4.221830, 10.404692,
        11.005459, 25.144694, 38.023848, 47.985495, 49.793214, 49.757610,
    ], id="labolle"),
])
def test_diffusors(spec, expected):
    n = len(Z0)
    np.random.seed(SEED)
    diffusor = VerticalDiffusor.create(spec)
    result = diffusor.diffuse(
        X=np.zeros(n), Y=np.zeros(n), Z=Z0, depth=np.full(n, DEPTH),
        pid=np.arange(n), dt=DT, vert_mix=profile,
    )
    np.testing.assert_allclose(result, expected, atol=1e-5)
