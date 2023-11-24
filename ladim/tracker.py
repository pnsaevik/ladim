from .model import Model, Module
import numpy as np


class Tracker(Module):
    def __init__(self, model: Model):
        super().__init__(model)


class HorizontalTracker:
    """The physical particle tracking kernel"""

    def __init__(self, model: Model, **config) -> None:
        self.model = model

        self.integrator = StochasticDifferentialEquationIntegrator.from_keyword(None)
        self.D = config["diffusion_coefficient"]  # [m2.s-1]
        self.active_check = 'active' in config['ibm_variables']

    def update(self):
        state = self.model.state
        grid = self.model.grid
        forcing = self.model.forcing

        dx, dy = grid.sample_metric(state['X'], state['Y'])
        t0 = self.model.solver.time
        dt = self.model.solver.step

        X, Y, z = state['X'], state['Y'], state['Z']
        r0 = np.stack([state['X'], state['Y']])

        # Set diffusion function
        def mixing(t, r):
            _ = t
            stddev = (2 * self.D) ** 0.5
            u_diff = stddev / dx
            return np.broadcast_to(u_diff, r.shape)

        # Set advection function
        def velocity(t, r):
            x, y = r.reshape([2, -1])
            u, v = forcing.velocity(x, y, state['Z'], tstep=(t - t0) / dt)
            return np.concatenate([u / dx, v / dy]).reshape(r.shape)

        X1, Y1 = self.integrator(velocity, mixing, t0, r0, dt)

        # Land, boundary treatment. Do not move the particles
        # Consider a sequence of different actions
        # I = (grid.ingrid(X1, Y1)) & (grid.atsea(X1, Y1))
        I = grid.atsea(X1, Y1)
        # I = True
        X[I] = X1[I]
        Y[I] = Y1[I]

        state['X'] = X
        state['Y'] = Y


class StochasticDifferentialEquationIntegrator:
    @staticmethod
    def from_keyword(kw):
        return RK4Integrator()

    def __call__(self, vel, mix, t0, r0, dt):
        """
        Integrate a stochastic differential equation, one time step

        For both ``vel`` and ``mix``, the calling signature is fun(t, r) where t is
        scalar time and r is an array of particle positions. The function should
        return an array of the same shape as r.

        Both ``vel`` and ``mix`` should  assume the same coordinate system and units as
        the input particle positions and time. For instance, if time is given in
        seconds and positions are given in grid coordinates, velocity should have units
        of grid cells per second and mixing should have units of grid cells squared per
        second.

        :param vel: Velocity function. Calling signature is fun(t, r) where t is scalar
        and r is an array of particle positions. Should return an array of the
        same shape as r.

        :param mix: Mixing function. Calling signature is fun(t, r) where t is scalar
        and r is an array of particle positions. Should return an array of the
        same shape as r.

        :param t0: Initial time

        :param r0: Initial positions

        :param dt: Time step size

        :return: Final particle positions (same shape as r0)
        """

        return r0


class RK4Integrator(StochasticDifferentialEquationIntegrator):
    def __call__(self, velocity, mixing, t0, r0, dt):
        u1 = velocity(t0, r0)
        r1 = r0 + 0.5 * u1 * dt

        u2 = velocity(t0 + 0.5 * dt, r1)
        r2 = r0 + 0.5 * u2 * dt

        u3 = velocity(t0 + 0.5 * dt, r2)
        r3 = r0 + u3 * dt

        u4 = velocity(t0 + dt, r3)

        u_adv = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0
        r_adv = r0 + u_adv * dt

        # Diffusive velocity
        u_diff = mixing(t0, r_adv)
        dw = np.random.normal(size=np.size(r0)).reshape(r0.shape) * np.sqrt(dt)

        return r_adv + u_diff * dw
