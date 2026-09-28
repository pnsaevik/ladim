import numpy as np
import pytest

from ladim.state import State
from ladim.tracker import Tracker


class MockObj:
    pass


def rotation(x, y, t, omega=1e-4, x0=500.0, y0=500.0, dx=100.0):
    """Solid body rotation [m/s] around (x0, y0), grid spacing dx"""
    return -omega * (y - y0) * dx, omega * (x - x0) * dx


def make_model(X, Y, active=None, velocity=rotation, dx=100.0, dt=600,
               atsea=None):
    model = MockObj()
    model.state = State()
    particles = dict(X=np.array(X, dtype=float), Y=np.array(Y, dtype=float),
                     Z=np.zeros(len(X)))
    if active is not None:
        particles['active'] = active
    model.state.append(particles)

    model.solver = MockObj()
    model.solver.time = 0
    model.solver.step = dt

    model.forcing = MockObj()
    model.forcing.velocity = lambda x, y, z, tstep=0.0: velocity(x, y, tstep * dt)

    model.grid = MockObj()
    model.grid.sample_metric = lambda x, y: (np.full(len(x), dx), np.full(len(x), dx))
    model.grid.atsea = atsea or (lambda x, y: np.ones(len(x), dtype=bool))
    return model


def reference(method, velocity, X, Y, dx, dt, diffusion):
    """Numpy implementation of the original ladim integrators"""
    r0 = np.stack([X, Y])

    def vel(t, r):
        x, y = r
        u, v = velocity(x, y, t)
        return np.stack([u / dx, v / dx])

    if method == "EF":
        r_adv = r0 + vel(0, r0) * dt
    elif method == "RK2":
        u1 = vel(0, r0)
        r_adv = r0 + vel(0.5 * dt, r0 + 0.5 * u1 * dt) * dt
    else:
        u1 = vel(0, r0)
        u2 = vel(0.5 * dt, r0 + 0.5 * u1 * dt)
        u3 = vel(0.5 * dt, r0 + 0.5 * u2 * dt)
        u4 = vel(dt, r0 + u3 * dt)
        r_adv = r0 + (u1 + 2 * u2 + 2 * u3 + u4) / 6.0 * dt
    if diffusion:
        dw = np.random.normal(size=r0.size).reshape(r0.shape) * np.sqrt(dt)
        r_adv = r_adv + (2 * diffusion) ** 0.5 / dx * dw
    return r_adv


@pytest.mark.parametrize("method", ["EF", "RK2", "RK4"])
def test_matches_reference_implementation(method):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1000, 1000)
    Y = rng.uniform(0, 1000, 1000)
    model = make_model(X, Y)

    Tracker.create(method, 0).update(model)
    expected = reference(method, rotation, X, Y, 100.0, 600, 0)

    assert np.allclose(model.state['X'], expected[0], rtol=0, atol=1e-10)
    assert np.allclose(model.state['Y'], expected[1], rtol=0, atol=1e-10)


@pytest.mark.parametrize("method, order", [("EF", 1), ("RK2", 2), ("RK4", 4)])
def test_convergence_order(method, order):
    # Solid body rotation: exact solution is known
    omega = 1e-4
    errors = []
    for dt in (2000, 1000):
        model = make_model([700.0], [500.0], dt=dt)
        tracker = Tracker.create(method, 0)
        for _ in range(8000 // dt):
            tracker.update(model)
        angle = omega * 8000
        x_exact = 500 + 200 * np.cos(angle)
        y_exact = 500 + 200 * np.sin(angle)
        errors.append(np.hypot(model.state['X'][0] - x_exact,
                               model.state['Y'][0] - y_exact))
    observed = np.log2(errors[0] / errors[1])
    assert observed == pytest.approx(order, abs=0.3)


def test_does_not_move_inactive_particles_or_onto_land():
    def uniform(x, y, t):
        return np.full(len(x), 10.0), np.zeros(len(x))

    def atsea(x, y):
        return x < 1000

    model = make_model(
        X=[100.0, 200.0, 995.0], Y=[0.0, 0.0, 0.0],
        active=[True, False, True], velocity=uniform, atsea=atsea,
    )
    Tracker.create("RK4", 0).update(model)
    # 10 m/s * 600 s / 100 m = 60 grid units
    assert model.state['X'].tolist() == [160.0, 200.0, 995.0]


def test_keeps_float32_state():
    model = make_model([100.0], [100.0])
    model.state['X'] = model.state['X'].astype(np.float32)
    Tracker.create("EF", 0).update(model)
    assert model.state['X'].dtype == np.float32


def still(x, y, t):
    return np.zeros(len(x)), np.zeros(len(x))


def random_walk(n=200_000, seed=1, steps=1, pid=None):
    """Displacements (in standard deviations) of a pure random walk"""
    diffusion, dt, dx = 2.0, 600, 100.0
    model = make_model(np.zeros(n), np.zeros(n), velocity=still, dx=dx, dt=dt)
    if pid is not None:
        model.state['pid'] = pid
    np.random.seed(seed)
    tracker = Tracker.create("RK4", diffusion)
    for _ in range(steps):
        tracker.update(model)
    sigma = (2 * diffusion * dt) ** 0.5 / dx
    return model.state['X'] / sigma, model.state['Y'] / sigma


def test_diffusion_is_standard_normal():
    for d in random_walk():
        assert d.mean() == pytest.approx(0, abs=0.01)
        assert d.var() == pytest.approx(1, rel=0.02)
        # Normal distribution: fraction within 1 and 2 std, kurtosis
        assert np.mean(np.abs(d) < 1) == pytest.approx(0.6827, abs=0.005)
        assert np.mean(np.abs(d) < 2) == pytest.approx(0.9545, abs=0.003)
        assert np.mean(d ** 4) == pytest.approx(3, rel=0.05)


def test_diffusion_is_uncorrelated():
    dx, dy = random_walk()
    assert np.corrcoef(dx, dy)[0, 1] == pytest.approx(0, abs=0.01)
    # Neighbouring particle identifiers
    assert np.corrcoef(dx[1:], dx[:-1])[0, 1] == pytest.approx(0, abs=0.01)
    # Consecutive time steps: variance grows linearly
    dx2, _ = random_walk(steps=2)
    assert dx2.var() == pytest.approx(2, rel=0.02)


def test_diffusion_is_reproducible():
    a = random_walk(n=1000, seed=3)
    b = random_walk(n=1000, seed=3)
    c = random_walk(n=1000, seed=4)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.allclose(a[0], c[0])


def test_diffusion_follows_particle_identifier():
    pid = np.arange(1000)
    a, _ = random_walk(n=1000, pid=pid)
    b, _ = random_walk(n=1000, pid=pid[::-1].copy())
    assert np.array_equal(a, b[::-1])


def test_unknown_method():
    with pytest.raises(NotImplementedError):
        Tracker.create("RK3", 0)
