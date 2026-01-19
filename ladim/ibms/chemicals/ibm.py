import numpy as np


class IBM:
    def __init__(self, config):
        ibmconf = config.get('ibm', dict())

        # Time before a particle is taken out of the simulation [seconds]
        self.lifespan = ibmconf.get('lifespan', None)

        self.D = ibmconf.get('vertical_mixing', 0)  # Vertical mixing [m*2/s]
        self.dt = config['dt']
        self.vertdiff_dt = ibmconf.get('vertdiff_dt', self.dt)  # Vertical diffusion timestep [s]
        self.vertdiff_dz = ibmconf.get('vertdiff_dz', 0)  # Spacing of vertical diffusion sampling [m]
        self.vertdiff_max = ibmconf.get('vertdiff_max', np.inf)  # Maximal vertical diffusion [m2/s]
        self.horzdiff_type = ibmconf.get('horzdiff_type', None)
        self.horzdiff_max = ibmconf.get('horzdiff_max', np.inf)
        self.horzdiff_min = ibmconf.get('horzdiff_min', 0)
        self.vertadv = ibmconf.get('vertical_advection', True)
        self.x = np.array([])
        self.y = np.array([])
        self.pid = np.array([])
        self.land_collision = ibmconf.get('land_collision', 'reposition')
        self.grid = None
        self.state = None
        self.forcing = None

        # Issue warning if parameters for vertical diffusion indicates numerical instability
        if self.vertdiff_max < np.inf and self.vertdiff_dz > 0:
            instability = 6 * self.vertdiff_max * self.vertdiff_dt / (self.vertdiff_dz ** 2)
            if instability > 1:
                import logging
                logging.warning('Possible unstable vertical diffusion scheme')
                logging.warning('Reduce time step, increase sampling distance or limit the diffusion coefficient')

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.state = state
        self.forcing = forcing

        if self.land_collision == "reposition":
            self.reposition()
        elif self.land_collision == "coastal_diffusion":
            self.coastal_diffusion()

        if self.vertadv:
            self.advect()

        if isinstance(self.D, str):
            self.diffuse_labolle()
        elif self.D:
            self.diffuse_const()

        if self.horzdiff_type == 'smagorinsky':
            self.horzdiff()

        if self.land_collision == "reposition":
            self.store_position()

        if self.lifespan is not None:
            self.kill_old()

    def advect(self):
        # Vertical advection
        x = self.state.X
        y = self.state.Y
        z = self.state.Z
        self.state['Z'] += self.dt * self.forcing.forcing.wvel(x, y, z)
        self.reflect()

    def kill_old(self):
        state = self.state
        state['age'] += self.dt
        state['alive'] = state.alive & (state.age <= self.lifespan)

    def horzdiff(self):
        # Itô backwards scheme (LaBolle et al. 2000) for horizontal diffusion
        x = self.state.X
        y = self.state.Y
        z = self.state.Z
        dt = self.dt
        dx, dy = self.grid.sample_metric(x, y)

        def compute_diff(xx, yy, zz):
            K = self.forcing.forcing.horzdiff(xx, yy, zz)
            K = np.maximum(self.horzdiff_min, np.minimum(self.horzdiff_max, K))
            return np.sqrt(2 * K)

        # X direction. Uniform stochastic differential. Predictor-corrector.
        dWx = (np.random.rand(len(z)) * 2 - 1) * np.sqrt(3 * dt) / dx
        diff1x = compute_diff(x, y, z)
        x1 = x + diff1x * dWx
        diff2x = compute_diff(x1, y, z)
        x2 = x + diff2x * dWx

        # Y direction. Uniform stochastic differential. Predictor-corrector.
        dWy = (np.random.rand(len(z)) * 2 - 1) * np.sqrt(3 * dt) / dy
        diff1y = compute_diff(x2, y, z)
        y1 = y + diff1y * dWy
        diff2y = compute_diff(x2, y1, z)
        y2 = y + diff2y * dWy

        # Kill particles trying to move out of grid
        in_grid = self.grid.ingrid(x2, y2)
        self.state['X'][in_grid] = x2[in_grid]
        self.state['Y'][in_grid] = y2[in_grid]
        self.state.alive[~in_grid] = False

    # Itô backwards scheme (LaBolle et al. 2000) for vertical diffusion
    def diffuse_labolle(self):
        x = self.state.X
        y = self.state.Y
        H = self.grid.sample_depth(x, y)

        if self.vertdiff_dz:
            def z_coarse(zz):
                dz = self.vertdiff_dz
                return np.maximum(0.25 * dz, ((zz - 0.5 * dz) // dz) * dz + dz)
        else:
            def z_coarse(zz):
                return zz

        def sample_K(xx, yy, zz):
            kk = self.forcing.forcing.vertdiff(xx, yy, z_coarse(zz), self.D)
            return np.minimum(kk, self.vertdiff_max)

        current_time = 0
        while current_time < self.dt:
            old_time = current_time
            current_time = np.minimum(self.dt, current_time + self.vertdiff_dt)
            ddt = current_time - old_time
            z = self.state.Z

            # Uniform stochastic differential
            dW = (np.random.rand(len(z)) * 2 - 1) * np.sqrt(3 * ddt)

            # Vertical diffusion, intermediate step
            Z1 = z + np.sqrt(2 * sample_K(x, y, z)) * dW  # Diffusive step
            Z1[Z1 < 0] *= -1                    # Reflexive boundary at top
            below_seabed = Z1 > H
            Z1[below_seabed] = 2*H[below_seabed] - Z1[below_seabed]  # Reflexive bottom

            # Diffusive step and reflective boundary conditions
            self.state['Z'] += np.sqrt(2 * sample_K(x, y, Z1)) * dW  # Diffusive step
            self.reflect()

    def diffuse_const(self):
        # Uniform stochastic differential
        dW = (np.random.rand(len(self.state.Z)) * 2 - 1) * np.sqrt(3 * self.dt)
        self.state['Z'] += np.sqrt(2 * self.D) * dW
        self.reflect()

    def reflect(self):
        x = self.state.X
        y = self.state.Y
        z = self.state.Z
        H = self.grid.sample_depth(x, y)
        below_seabed = z > H
        z[z < 0] *= -1  # Reflexive boundary at top
        z[below_seabed] = 2 * H[below_seabed] - z[below_seabed]  # Reflexive bottom
        self.state['Z'] = z

    def reposition(self):
        # If particles have not moved: Assume they ended up on land.
        # If that is the case, reposition them within the cell.
        pid, pidx_old, pidx_new = np.intersect1d(self.pid, self.state.pid, return_indices=True)
        onland = ((self.x[pidx_old] == self.state.X[pidx_new]) &
                  (self.y[pidx_old] == self.state.Y[pidx_new]))
        num_onland = np.count_nonzero(onland)
        pidx_new_onland = pidx_new[onland]
        x_new = np.round(self.state.X[pidx_new_onland]) - 0.5 + np.random.rand(num_onland)
        y_new = np.round(self.state.Y[pidx_new_onland]) - 0.5 + np.random.rand(num_onland)
        self.state.X[pidx_new_onland] = x_new
        self.state.Y[pidx_new_onland] = y_new

    def store_position(self):
        self.x = self.state.X
        self.y = self.state.Y
        self.pid = self.state.pid

    def coastal_diffusion(self):
        # If particles are close to coast, reposition them within the cell
        x, y, pid = self.state.X, self.state.Y, self.state.pid
        is_coastal = self.grid.grid.is_close_to_land(x, y)
        num_coastal = np.count_nonzero(is_coastal)
        x_new = np.round(x[is_coastal]) - 0.5 + np.random.rand(num_coastal)
        y_new = np.round(y[is_coastal]) - 0.5 + np.random.rand(num_coastal)
        self.state['X'][is_coastal] = x_new
        self.state['Y'][is_coastal] = y_new
