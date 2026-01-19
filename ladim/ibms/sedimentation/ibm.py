import numpy as np
import typing


class IBM:
    def __init__(self, config):
        # Time before a particle is taken out of the simulation [seconds]
        self.lifespan = config['ibm']['lifespan']

        # Vertical mixing [m*2/s]
        self.vdiff_fn = get_vdiff_fn(config['ibm'].get('vertical_mixing', None))
        self.taucrit_fn = get_taucrit_fn(config['ibm'].get('taucrit', None))

        # Store time step value to calculate age
        self.dt = config['dt']

        # Reference to other modules
        self.grid = None        # type: typing.Any
        self.forcing = None     # type: typing.Any
        self.state = None       # type: typing.Any

        # Variables for lazy evaluation
        self._ustar = None
        self._ustar_tstep = -1

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.forcing = forcing
        self.state = state

        has_been_buried_before = (self.state.active != 1)

        self.initialize()
        self.resuspend()
        self.diffuse()
        self.sink()
        self.bury()
        self.kill_old()

        is_active = (self.state.active != 0)
        self.state.active[is_active & has_been_buried_before] = 2

    def initialize(self):
        state = self.state
        idx_new_particles = state['sink_vel'][:] == 0
        num_new_particles = np.count_nonzero(idx_new_particles)

        if num_new_particles:
            state['sink_vel'][idx_new_particles] = sinkvel(num_new_particles)

        if "jitter" in state:
            idx_new_jitter_particles = state['jitter'][:] >= 0
            num_new_jitter_particles = np.count_nonzero(idx_new_jitter_particles)

            if num_new_jitter_particles:
                jitter_r = state['jitter'][idx_new_jitter_particles]
                jitter_x, jitter_y = circle_rand(num_new_jitter_particles)
                x = state['X'][idx_new_jitter_particles]
                y = state['Y'][idx_new_jitter_particles]                
                grid_x, grid_y = self.grid.sample_metric(x, y)
                dx_coord = jitter_x * jitter_r / grid_x
                dy_coord = jitter_y * jitter_r / grid_y
                state['X'][idx_new_jitter_particles] = x + dx_coord
                state['Y'][idx_new_jitter_particles] = y + dy_coord
                state['jitter'][idx_new_jitter_particles] = -1

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
        a = self.state.active != 0
        X, Y, Z = self.state.X[a], self.state.Y[a], self.state.Z[a]

        # Define which particles have settled to the bottom and which have not
        H = grid.sample_depth(X, Y)  # Water depth
        at_seabed = Z > H
        Z[at_seabed] = H[at_seabed]

        # Store new vertical position
        self.state.Z[a] = Z
        self.state.active[a] = ~at_seabed

    def diffuse(self):
        if self.vdiff_fn is None:
            return

        # Get parameters
        a = self.state.active != 0
        x, y, z = self.state.X[a], self.state.Y[a], self.state.Z[a]
        h = self.grid.sample_depth(x, y)
        ustar = self.shear_velocity_btm()[a]

        self.state.Z[a] = self.vdiff_fn(z, h, self.dt, ustar)

    def sink(self):
        # Get parameters
        a = self.state.active != 0
        z = self.state.Z[a]
        w = self.state.sink_vel[a]  # Sink velocity

        # Euler scheme, no boundary conditions
        self.state.Z[a] = z + self.dt * w

    def kill_old(self):
        state = self.state
        state['age'] += state.dt
        state['alive'] = state.alive & (state.age <= self.lifespan)

    def shear_velocity_btm(self):
        if self._ustar_tstep < self.state.timestep:
            # Calculate bottom shear velocity from last computational layer
            # velocity
            # returns: Ustar at bottom cell
            x = self.state.X
            y = self.state.Y
            h = self.grid.sample_depth(x, y)

            u_btm, v_btm = self.forcing.velocity(x, y, h)
            U2 = u_btm*u_btm + v_btm*v_btm
            c = 0.003
            self._ustar = np.sqrt(c * U2)
            self._ustar_tstep = self.state.timestep

        assert self._ustar is not None
        return self._ustar


def shear_stress_btm(ustar):
    rho = 1000
    return ustar * ustar * rho


def ladis(x0, t0, t1, v, K):
    """
    Lagrangian Advection and DIffusion Solver.

    Solve the diffusion equation in a Lagrangian framework. The equation is

    dc/dt = - grad (vc) + div (K grad c),

    where c is concentration, v is 3-dimensional velocity, K is a diagonal
    tensor (i.e. main axes oriented along coordinate axes) of diffusion.

    This is translated to a stochastic differential equation of the form

    dx = (v_x + d/dx K_xx) * dt + sqrt(2*K_xx) * dw_x,
    dy = (v_y + d/dy K_yy) * dt + sqrt(2*K_yy) * dw_y,
    dz = (v_z + d/dz K_zz) * dt + sqrt(2*K_zz) * dw_z,

    where x, y, z is the spatial position, K_xx, K_yy, K_zz are the diagonal
    elements of K, and dw_x, dw_y, dw_z are Wiener process differential elements
    with zero mean and stdev = sqrt(dt).

    Algorithm:

    Operator splitting: Diffusion first, then advection. Diffusion is solved
    using the gradient-free backwards Itô scheme, according to LaBolle
    (2000, 10.1029/1999WR900224).

    :param x0: An N x M vector of initial values, where N is the number of
               particles and M is the number of coordinates.
    :param t0: The initial time.
    :param t1: The end time.
    :param v:  The velocity. A function (x, t) --> x-like.
    :param K:  The diagonal elements of the diffusion tensor.
               A function (x, t) --> x-like.
    :return:   An x0-like array of the new particle positions.
    """

    dt = t1 - t0

    # --- Diffusion, LaBolle scheme ---

    # First diffusion step (predictor)
    b0 = np.sqrt(2 * K(x0, t0))
    dw = np.random.randn(x0.size).reshape(x0.shape) * np.sqrt(dt)
    x1 = x0 + b0 * dw

    # Second diffusion step (corrector)
    b1 = np.sqrt(2 * K(x1, t0))
    x2 = x0 + b1 * dw

    # --- Advection, forward Euler ---

    a3 = v(x2, t0)
    x3 = x2 + a3 * dt

    return x3


def get_taucrit_fn(subconf):
    if subconf is None:
        return None

    if not isinstance(subconf, dict):
        subconf = dict(method='constant', value=subconf)

    method = subconf['method']
    if method == 'constant':
        value = subconf['value']
        return lambda lon, lat: np.zeros_like(lon) + value
    elif method == 'grain_size_bin':
        return get_taucrit_fn_grain_size(
            source=subconf['source'],
            varname=subconf['varname'],
            method='bin',
        )
    elif method == 'grain_size_poly':
        return get_taucrit_fn_grain_size(
            source=subconf['source'],
            varname=subconf['varname'],
            method='poly',
        )
    else:
        raise ValueError(f'Unknown method: {method}')


def get_taucrit_fn_grain_size(source, varname, method):
    import xarray as xr
    with xr.open_dataset(source) as dset:
        grain_size_var = dset.data_vars[varname]
        grain_size = grain_size_var.transpose('latitude', 'longitude').values
        clat = dset.latitude.values
        clon = dset.longitude.values

    difflat = clat[1] - clat[0]
    difflon = clon[1] - clon[0]
    lat0 = clat[0]
    lon0 = clon[0]
    imax = grain_size.shape[1] - 1
    jmax = grain_size.shape[0] - 1

    def sedvalue(lon, lat):
        i = np.clip(np.int32(.5 + (lon - lon0) / difflon), 0, imax)
        j = np.clip(np.int32(.5 + (lat - lat0) / difflat), 0, jmax)
        sed = grain_size[j, i]
        sed[np.isnan(sed)] = 0
        return sed

    def taucrit_bin(lon, lat):
        sed = sedvalue(lon, lat)
        tauc = np.empty(lon.shape, dtype=np.float32)
        tauc[:] = 0.12  # Applies both to sed == 0 (default value) and 70 <= sed < 180
        tauc[(sed > 0) & (sed < 70)] = 0.06
        tauc[sed > 180] = 0.32
        return tauc

    def taucrit_poly(lon, lat):
        sed = sedvalue(lon, lat)
        tauc = 6e-6*sed**2+3e-5*sed+0.0591
        tauc[sed == 0] = 0.12  # Default value
        return tauc

    taucrit_fn = dict(
        bin=taucrit_bin,
        poly=taucrit_poly,
    )

    return taucrit_fn[method]


def get_vdiff_fn(subconf):
    if subconf is None:
        return None

    if not isinstance(subconf, dict):
        subconf = dict(method='constant', value=subconf)

    method = subconf['method']
    if method == 'constant':
        return get_vdiff_constant_fn(subconf['value'])

    elif method == 'bounded_linear':
        return get_vdiff_bounded_linear_fn(subconf['max_diff'])

    else:
        raise ValueError(f'Unknown method: {method}')


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


def get_vdiff_bounded_linear_fn(max_diff):
    def get_turbulence(ustar, meters_from_seafloor, max_mixing):
        # Extracts vertical diffusion and vertical diffusion gradient following a linear
        # function from the bottom to the base of the model where we asume BBL finishes
        # and Az becomes constant and max
        kappa = 0.41
        dA_dz = kappa * ustar  # Alternative formulation: kappa * ustar * w * exp(-w/w0), where w = z - h
        A = dA_dz * meters_from_seafloor
        cutoff = (A > max_mixing)
        A[cutoff] = max_mixing
        dA_dz[cutoff] = 0
        return A, dA_dz

    def fn(z, h, dt, ustar):
        A, dA_dZ = get_turbulence(ustar, np.maximum(h - z, 0), max_diff)

        # Diffusion velocity, adding a pseudovelocity and evaluating the diffusion
        # in an upstream point
        rand = np.random.normal(size=len(A))
        diff_upstream = A + 0.5 * dA_dZ ** 2 * dt
        w = -dA_dZ + rand * np.sqrt(2 * diff_upstream / dt)

        # Vertical position update
        z_new = z + dt * w  # Euler forward timestepping
        benthic = z_new >= h  # Newly settled particles
        z_new[benthic] = h[benthic]  # Absorbant boundary at bottom
        z_new[z_new < 0] *= -1  # Reflective boundary at surface

        return z_new

    return fn


def sinkvel(n):
    from scipy.interpolate import InterpolatedUnivariateSpline
    sinkvel_tab = np.array([.100, .050, .025, .015, .010, .005, 0])
    cumprob_tab = np.array([.000, .662, .851, .883, .909, .937, 1])
    fn = InterpolatedUnivariateSpline(cumprob_tab, sinkvel_tab, k=2)
    return fn(np.random.rand(n))


def get_settled_particles(dset):
    # Find indices of last recorded timestep per particle
    pid, right_index = np.unique(np.flip(dset['pid'].values), return_index=True)
    pinst = len(dset['pid']) - right_index - 1

    # Construct xarray
    import xarray as xr
    pid_vars = {k: xr.Variable('pid', v[pid])
                for k, v in dset.variables.items()
                if v.dims == ('particle',)}
    pinst_vars = {k: xr.Variable('pid', v[pinst])
                  for k, v in dset.variables.items()
                  if v.dims == ('particle_instance',)}
    all_vars = {**dict(pid=pid), **pid_vars, **pinst_vars}
    return xr.Dataset(all_vars).assign_coords(pid=pid)


def circle_rand(n):
    """
    Return a set of random positions within the unit circle
    """
    r = np.sqrt(np.random.rand(n))       # radius (sqrt ensures uniform density)
    theta = np.random.rand(n) * 2 * np.pi  # angle in radians
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y