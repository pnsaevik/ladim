import numpy as np


class IBM:
    def __init__(self, config):
        # Time before a particle is taken out of the simulation [seconds]
        self.lifespan = config['ibm']['lifespan']

        # Vertical mixing [m*2/s]
        self.vdiff = config['ibm'].get('vertical_mixing', 0)
        self.taucrit_fn = get_taucrit_fn(config['ibm'].get('taucrit', 1000))

        # Vertical advection on/off
        self.vadv = config['ibm'].get('vertical_advection', False)

        # Store time step value to calculate age
        self.dt = config['dt']

        # Possible separate output file to record time (and place) of death
        self.output_file = config["ibm"].get('output_file', None)
        self.output_vars = {
            varname: config['nc_attributes'][varname]
            for varname in config['output_instance']
        }
        if self.output_file:
            create_outfile(self.output_file, self.output_vars)

        # Record positions to know if the particles are stuck near land
        self.land_collision = config["ibm"].get('land_collision', 'reposition')
        self.x = np.array([])
        self.y = np.array([])
        self.pid = np.array([])

        # Reference to other modules
        self.grid = None
        self.forcing = None
        self.state = None

        # Variables for lazy evaluation
        self._ustar = None
        self._ustar_tstep = -1

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.forcing = forcing
        self.state = state

        if self.has_active():
            has_been_buried_before = (self.state.active != 1)
        else:
            has_been_buried_before = None

        self.reposition()
        self.resuspend()
        self.diffuse()
        self.sink()
        self.bury()
        self.kill_old()
        self.store()

        if self.has_active():
            is_active = (self.state.active != 0)
            self.state.active[is_active & has_been_buried_before] = 2

    def has_active(self):
        try:
            _ = self.state['active']
            return True
        except KeyError:
            return False

    def active(self):
        if self.has_active():
            return self.state['active']
        else:
            return np.broadcast_to(1, self.state.X.shape)

    def reposition(self):
        if self.land_collision == "reposition":
            state = self.state
            a = self.active()
            X, Y = state['X'], state['Y']

            # If particles have not moved: Assume they ended up on land.
            # If that is the case, reposition them within the cell.
            pid, pidx_old, pidx_new = np.intersect1d(self.pid, state.pid, return_indices=True)
            onland = ((self.x[pidx_old] == X[pidx_new]) &
                      (self.y[pidx_old] == Y[pidx_new]) &
                      np.bool_(a[pidx_new])
                      )
            num_onland = np.count_nonzero(onland)
            pidx_new_onland = pidx_new[onland]
            x_new = np.round(X[pidx_new_onland]) - 0.5 + np.random.rand(num_onland)
            y_new = np.round(Y[pidx_new_onland]) - 0.5 + np.random.rand(num_onland)
            X[pidx_new_onland] = x_new
            Y[pidx_new_onland] = y_new

            state['X'] = X
            state['Y'] = Y
            self.x = np.copy(state.X)
            self.y = np.copy(state.Y)
            self.pid = state.pid

    def resuspend(self):
        if self.taucrit_fn is None:
            return

        ustar = self.shear_velocity_btm()
        tau = shear_stress_btm(ustar)
        lon, lat = self.grid.lonlat(self.state.X, self.state.Y)
        taucrit = self.taucrit_fn(lon, lat)
        resusp = tau >= taucrit
        self.state.active[resusp] = True

    def bury(self):
        grid = self.grid
        a = self.active() != 0
        X, Y, Z = self.state.X[a], self.state.Y[a], self.state.Z[a]

        # Define which particles have settled to the bottom and which have not
        H = grid.sample_depth(X, Y)  # Water depth
        at_seabed = Z > H
        Z[at_seabed] = H[at_seabed]

        # Store new vertical position
        self.state.Z[a] = Z
        if self.has_active():
            self.state.active[a] = ~at_seabed

        # Kill buried particles if no resuspension
        if not self.taucrit_fn:
            self.state['alive'][a] &= ~at_seabed

    def diffuse(self):
        # Get parameters
        a = self.active() != 0
        x, y, z = self.state.X[a], self.state.Y[a], self.state.Z[a]
        dt = self.dt

        # Diffusion
        b0 = np.sqrt(2 * self.vdiff)
        dw = np.random.randn(z.size).reshape(z.shape) * np.sqrt(dt)
        z1 = z + b0 * dw

        # Reflexive boundary condition at the top
        z1[z1 < 0] *= -1  # Surface

        # Update vertical position
        self.state.Z[a] = z1

    def sink(self):
        # Get parameters
        a = self.active() != 0
        z = self.state.Z[a]
        w = self.state.sink_vel[a]  # Sink velocity
        if self.vadv:
            w += self.forcing.forcing.wvel(self.state.X[a], self.state.Y[a], z)

        # Euler scheme, no boundary conditions
        self.state.Z[a] = z + self.dt * w

    def kill_old(self):
        state = self.state
        state['age'] += state.dt
        state['alive'] &= state['age'] <= self.lifespan

    def store(self):
        if not self.output_file:
            return

        dead = ~self.state.alive
        new_values = {
            k: self.state[k][dead]
            for k in self.output_vars.keys()
            if k not in ['lon', 'lat']
        }
        if 'lon' in self.output_vars.keys() and 'lat' in self.output_vars.keys():
            new_values['lon'], new_values['lat'] = self.grid.xy2ll(
                new_values['X'], new_values['Y'])

        update_outfile(self.output_file, new_values)

    def shear_velocity_btm(self):
        if self._ustar_tstep < self.state.timestep:
            # Calculate bottom shear velocity from last computational layer
            # velocity
            # returns: Ustar at bottom cell
            x = self.state.X
            y = self.state.Y
            h = self.grid.sample_depth(x, y)

            u_btm, v_btm = self.forcing.velocity(x, y, h, tstep=0)
            U2 = u_btm*u_btm + v_btm*v_btm
            c = 0.003
            self._ustar = np.sqrt(c * U2)
            self._ustar_tstep = self.state.timestep

        return self._ustar


def shear_stress_btm(ustar):
    rho = 1000
    return ustar * ustar * rho


def get_taucrit_fn(value):
    if value >= 1000:  # No resuspension if high value
        return None
    else:
        return lambda lon, lat: np.zeros_like(lon) + value


def get_vdiff_constant_fn(value):
    def fn(z, h, dt, _):
        # Diffusion
        b0 = np.sqrt(2 * value)
        dw = np.random.randn(z.size).reshape(z.shape) * np.sqrt(dt)
        z1 = z + b0 * dw

        # Reflexive boundary conditions
        z1[z1 < 0] *= -1  # Surface
        below_seabed = z1 > h
        z1[below_seabed] = 2*h[below_seabed] - z1[below_seabed]

        # Return new vertical position
        return z1

    return fn


def create_outfile(fname, variables):
    import netCDF4 as nc
    with nc.Dataset(fname, 'w') as dset:
        dset.createDimension('particle', None)

        for k, v in variables.items():
            var = dset.createVariable(k, v['ncformat'], 'particle')
            for attr_name, attr_val in v.items():
                if attr_name != 'ncformat':
                    var.setncattr(attr_name, attr_val)


def update_outfile(fname, new_values):
    import netCDF4 as nc
    with nc.Dataset(fname, 'a') as dset:
        num_old = dset.dimensions['particle'].size
        num_new = len(next(v for v in new_values.values()))

        for k, v in new_values.items():
            dset.variables[k][num_old:num_old + num_new] = v
