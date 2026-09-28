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
@pytest.mark.parametrize("diffusion", [0.0, 1.0])
def test_matches_reference_implementation(method, diffusion):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1000, 1000)
    Y = rng.uniform(0, 1000, 1000)
    model = make_model(X, Y)

    np.random.seed(1)
    Tracker.create(method, diffusion).update(model)
    np.random.seed(1)
    expected = reference(method, rotation, X, Y, 100.0, 600, diffusion)

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


def test_diffusion_variance():
    n = 200_000
    diffusion, dt, dx = 2.0, 600, 100.0

    def still(x, y, t):
        return np.zeros(len(x)), np.zeros(len(x))

    model = make_model(np.zeros(n), np.zeros(n), velocity=still, dx=dx, dt=dt)
    Tracker.create("RK4", diffusion).update(model)
    expected_var = 2 * diffusion * dt / dx ** 2
    for k in ('X', 'Y'):
        assert model.state[k].mean() == pytest.approx(0, abs=0.01)
        assert model.state[k].var() == pytest.approx(expected_var, rel=0.02)


def test_unknown_method():
    with pytest.raises(NotImplementedError):
        Tracker.create("RK3", 0)
