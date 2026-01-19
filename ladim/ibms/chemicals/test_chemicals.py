import numpy as np
from ladim_plugins.chemicals import gridforce, IBM


def is_legacy():
    try:
        import ladim.timestepper
    except ModuleNotFoundError:
        return True
    return False


class Test_nearest_unmasked:
    def test_correct_when_all_unmasked(self):
        mask = np.zeros((4, 3))
        i = np.array([1, 1, 2])
        j = np.array([2, 3, 3])
        ii, jj = gridforce.nearest_unmasked(mask, i, j)
        assert ii.tolist() == i.tolist()
        assert jj.tolist() == j.tolist()

    def test_correct_when_south_edge(self):
        mask_south = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
        i = np.array([0, 1, 2.51])
        j = np.array([0, 1, 1.49])
        ii, jj = gridforce.nearest_unmasked(mask_south, i, j)
        assert ii.tolist() == [0, 1, 3]
        assert jj.tolist() == [0, 0, 0]

    def test_correct_when_corner(self):
        mask = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 0]])
        i = np.array([0.51, 0.51, 0.99, 1.49, 1.51, 2.00, 3.00])
        j = np.array([0.52, 0.98, 0.52, 1.01, 1.01, 1.01, 1.01])
        ii, jj = gridforce.nearest_unmasked(mask, i, j)
        assert ii.tolist() == [0, 0, 1, 1, 2, 2, 3]
        assert jj.tolist() == [1, 1, 0, 0, 0, 0, 2]


class Test_is_close_to_land:
    def test_correct_when_all_land(self):
        mask = np.zeros((4, 3))
        i = np.array([1, 1, 2])
        j = np.array([2, 3, 3])
        isclose = gridforce.is_close_to_land(mask, i, j)
        assert isclose.tolist() == [True, True, True]

    def test_correct_when_all_sea(self):
        mask = np.ones((4, 3))
        i = np.array([1, 1, 2])
        j = np.array([2, 3, 3])
        isclose = gridforce.is_close_to_land(mask, i, j)
        assert isclose.tolist() == [False, False, False]

    def test_correct_when_south_edge(self):
        mask_south = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
        i = np.array([0, 1, 2])
        j = np.array([0, 1, 2])
        isclose = gridforce.is_close_to_land(mask_south, i, j)
        assert isclose.tolist() == [True, True, False]

    def test_correct_when_west_edge(self):
        mask_west = np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])
        i = np.array([0, 1, 2, 3])
        j = np.array([0, 1, 2, 2])
        isclose = gridforce.is_close_to_land(mask_west, i, j)
        assert isclose.tolist() == [True, True, False, False]


class Test_compute_w:
    def test_requires_correct_shape(self):
        pn = np.ones((10, 15))
        pm = pn
        u = np.zeros((1, 20, 10, 14))
        v = np.zeros((1, 20, 9, 15))
        z_w = np.zeros((1, 21, 10, 15))
        for i in range(21):
            z_w[:, i, :, :] = i
        z_r = z_w[:, :-1, :, :] + 0.5
        w = gridforce.compute_w(pn, pm, u, v, z_w, z_r)
        assert w.shape == z_w.shape

    def test_zero_when_divergence_free_horizontal_velocity(self):
        t, z, eta, xi = np.meshgrid(
            range(1), range(3), range(4), range(5), indexing='ij')

        eta_u = 0.5 * (eta[:, :, :, :-1] + eta[:, :, :, 1:])
        xi_u = 0.5 * (xi[:, :, :, :-1] + xi[:, :, :, 1:])
        eta_v = 0.5 * (eta[:, :, :-1, :] + eta[:, :, 1:, :])
        # xi_v = 0.5 * (xi[:, :, :-1, :] + xi[:, :, 1:, :])

        z_r = z + 0.5
        z_w = np.concatenate((z, 1 + z[:, -1:, :, :]), axis=1)
        pn = np.ones(xi.shape[-2:])
        pm = pn

        # Define divergence-free field
        u = eta_u * xi_u
        v = - eta_v * eta_v

        w = gridforce.compute_w(pn, pm, u, v, z_w, z_r)
        assert np.max(np.abs(w)) < 1e-7

    def test_computes_positive_velocity_when_downward(self):
        t, z, eta, xi = np.meshgrid(
            range(1), [-2, -1, 0], range(4), range(5), indexing='ij')

        z_r = z + 0.5
        z_w = np.concatenate((1 + z[:, :1, :, :], z), axis=1)

        eta_u = 0.5 * (eta[:, :, :, :-1] + eta[:, :, :, 1:])
        xi_u = 0.5 * (xi[:, :, :, :-1] + xi[:, :, :, 1:])
        z_u = 0.5 * (z_r[:, :, :, :-1] + z_r[:, :, :, 1:])
        eta_v = 0.5 * (eta[:, :, :-1, :] + eta[:, :, 1:, :])
        xi_v = 0.5 * (xi[:, :, :-1, :] + xi[:, :, 1:, :])
        z_v = 0.5 * (z_r[:, :, :-1, :] + z_r[:, :, 1:, :])

        pn = np.ones(xi.shape[-2:])
        pm = pn

        # Define convergence on top, divergence on bottom
        # This gives downward movement
        u = (z_u - 1.5) * (eta_u + xi_u)
        v = (z_v - 1.5) * (eta_v + xi_v)

        w = gridforce.compute_w(pn, pm, u, v, z_w, z_r)

        # Outer edges and top+bottom is zero, so we check internal velocity
        w_internal = w[0, 1:-1, 1:-1, 1:-1]
        assert np.all(w_internal > 0), 'Downward velocity should be positive'


class Test_vertdiff:
    def test_stable_distribution_when_discontinuous_vertdiff(self):
        np.random.seed(0)
        num_particles = 10000
        num_updates = 100
        depth = 10
        dx = 1
        AKs = 0.001
        dt = 100
        vertdiff = lambda z: AKs/100 + AKs*99/100 * ((depth/2 < z) & (z < depth/2 + dx))

        ibm = IBM(
            dict(
                dt=dt,
                ibm=dict(
                    land_collision='freeze',
                    vertical_mixing='AKs',
                ),
            )
        )

        state = Stub()

        forcing = Stub()
        forcing.forcing = Stub()
        forcing.forcing.wvel = lambda x, y, z: x*0
        forcing.forcing.vertdiff = lambda x, y, z, n: vertdiff(z)

        grid = Stub()
        grid.sample_depth = lambda x, y: x*0 + depth

        state.X = np.ones(num_particles)
        state.Y = np.ones(num_particles)
        state.Z = np.arange(num_particles) * depth / num_particles

        bins = np.linspace(0, 1, 11) * depth
        pre_distribution = np.histogram(state.Z, bins=bins)[0]

        for i in range(num_updates):
            ibm.update_ibm(grid, state, forcing)

        post_distribution = np.histogram(state.Z, bins=bins)[0]
        deviation = np.linalg.norm(np.divide(post_distribution, pre_distribution) - 1)
        assert deviation < 0.1

    def test_unstable_distribution_if_big_vertdiff_gradient(self):
        np.random.seed(0)
        num_particles = 10000
        num_updates = 1
        depth = 10
        dx = 1
        AKs = 0.01
        dt = 100
        vertdiff = lambda z: AKs/100 + AKs*99/100 * ((depth/2 < z) & (z < depth/2 + dx))

        ibm = IBM(
            dict(
                dt=dt,
                ibm=dict(
                    land_collision='freeze',
                    vertical_mixing='AKs',
                ),
            )
        )

        state = Stub()

        forcing = Stub()
        forcing.forcing = Stub()
        forcing.forcing.wvel = lambda x, y, z: x*0
        forcing.forcing.vertdiff = lambda x, y, z, n: vertdiff(z)

        grid = Stub()
        grid.sample_depth = lambda x, y: x*0 + depth

        state.X = np.ones(num_particles)
        state.Y = np.ones(num_particles)
        state.Z = np.arange(num_particles) * depth / num_particles

        bins = np.linspace(0, 1, 11) * depth
        pre_distribution = np.histogram(state.Z, bins=bins)[0]

        for i in range(num_updates):
            ibm.update_ibm(grid, state, forcing)

        post_distribution = np.histogram(state.Z, bins=bins)[0]
        deviation = np.linalg.norm(np.divide(post_distribution, pre_distribution) - 1)
        assert deviation > 0.1

    def test_stable_distribution_if_big_vertdiff_gradient_and_small_dt(self):
        np.random.seed(0)
        num_particles = 10000
        num_updates = 1
        depth = 10
        dx = 1
        AKs = 0.01
        dt = 100
        vertdiff = lambda z: AKs/100 + AKs*99/100 * ((depth/2 < z) & (z < depth/2 + dx))

        ibm = IBM(
            dict(
                dt=dt,
                ibm=dict(
                    land_collision='freeze',
                    vertical_mixing='AKs',
                    vertdiff_dt=1,
                ),
            )
        )

        state = Stub()

        forcing = Stub()
        forcing.forcing = Stub()
        forcing.forcing.wvel = lambda x, y, z: x*0
        forcing.forcing.vertdiff = lambda x, y, z, n: vertdiff(z)

        grid = Stub()
        grid.sample_depth = lambda x, y: x*0 + depth

        state.X = np.ones(num_particles)
        state.Y = np.ones(num_particles)
        state.Z = np.arange(num_particles) * depth / num_particles

        bins = np.linspace(0, 1, 11) * depth
        pre_distribution = np.histogram(state.Z, bins=bins)[0]

        for i in range(num_updates):
            ibm.update_ibm(grid, state, forcing)

        post_distribution = np.histogram(state.Z, bins=bins)[0]
        deviation = np.linalg.norm(np.divide(post_distribution, pre_distribution) - 1)
        assert deviation < 0.1


class Test_update:
    def test_kills_old_particles(self):
        ibm_conf = {
            'land_collision': 'freeze',
            'vertical_advection': False,
            'lifespan': 20,
        }
        conf = {'dt': 10, 'ibm': ibm_conf}
        state = Stub()
        state['age'] = np.array([0, 10, 20])
        state['alive'] = np.array([False, True, True])
        ibm = IBM(conf)
        ibm.update_ibm(grid=None, state=state, forcing=None)
        assert state['alive'].tolist() == [False, True, False]


class Stub:
    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        setattr(self, item, value)
