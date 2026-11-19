"""Counter-based random numbers, shared by the random walks of the tracker"""

import numba
import numpy as np


@numba.njit(inline="always")
def splitmix64(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@numba.njit(inline="always")
def noise_pair(key, pid):
    """Two independent random numbers from (key, pid), mean 0 and variance 1

    Uniform on [-sqrt(3), sqrt(3)), about 10 times cheaper than normal
    deviates (Box-Muller).
    """
    h1 = splitmix64(key ^ splitmix64(np.uint64(pid)))
    h2 = splitmix64(h1)
    scale = 2.0 * np.sqrt(3.0) / 9007199254740992.0  # 2 * sqrt(3) / 2**53
    return (np.float64(h1 >> np.uint64(11)) * scale - np.sqrt(3.0),
            np.float64(h2 >> np.uint64(11)) * scale - np.sqrt(3.0))
