import numpy as np


class IBM:
    """Adding a constant horizontal velocity to the particle tracking"""

    def __init__(self, config):
        # Can not initialize grid here, as grid is not available

        # Azimuthal direction, 0 = N, 90 = E, 180 = S, 270 = W
        self.direction = 180   # [clockwise degree from North]

        self.speed = config['ibm']['speed']
        moon_lat, moon_lon = config['ibm']['lunar_latlon']
        self.D = config['ibm']['vertical_mixing']
        self.vertical_limits = config['ibm']['vertical_limits']

        self.dt = config['dt']
        self.xs_dx = None
        self.ys_dy = None
        self.state = None
        self.grid = None

        self.moonfunc = get_moon_function(lat=moon_lat, lon=moon_lon)

    def update_ibm(self, grid, state, _):
        self.state = state
        self.grid = grid

        if self.xs_dx is None:
            self.init_grid()

        self.horizontal_advect()
        self.vertical_diffuse()

    def init_grid(self):
        defgrid = self.grid.grid
        angle = defgrid.angle
        azim = self.direction * np.pi / 180.0
        # (xs, ys) is unit vector in the direction
        xs = np.sin(azim + angle)
        ys = np.cos(azim + angle)
        self.xs_dx = xs / defgrid.dx
        self.ys_dy = ys / defgrid.dy

    def horizontal_advect(self):
        state = self.state

        if self.moonfunc(state.timestamp):
            # Update position
            i = np.round(state.X).astype('int')
            j = np.round(state.Y).astype('int')
            x1 = state.X + self.speed * self.dt * self.xs_dx[j, i]
            y1 = state.Y + self.speed * self.dt * self.ys_dy[j, i]

            # Do not move out of grid or on land
            idx = self.grid.ingrid(x1, y1) & self.grid.atsea(x1, y1)
            state.X[idx] = x1[idx]
            state.Y[idx] = y1[idx]

    def vertical_diffuse(self):
        state = self.state

        # Random diffusion velocity
        rand = np.random.normal(size=len(state.Z))
        state['Z'] += rand * np.sqrt(2 * self.D * self.dt)

        # Keep within vertical limits, reflexive condition
        state['Z'] = reflexive(state.Z, *self.vertical_limits)


def reflexive(r, rmin=-np.inf, rmax=np.inf):
    r = r.copy()
    r[r < rmin] = 2*rmin - r[r < rmin]
    r[r > rmax] = 2*rmax - r[r > rmax]
    return np.clip(r, rmin, rmax)


def _load_ephemeris():
    from skyfield.api import load_file
    pkname = 'ladim_plugins.lunar_eel'

    try:
        from importlib.resources import files, as_file
        with as_file(files(pkname).joinpath('de421.bsp')) as fname:
            return load_file(fname)
    except ImportError:
        import pkg_resources
        try:
            fname = pkg_resources.resource_filename(pkname, 'de421.bsp')
            return load_file(fname)
        finally:
            pkg_resources.cleanup_resources()


def get_moon_function(lat, lon):
    from skyfield.api import load, Topos
    import datetime
    from skyfield.timelib import utc

    # Load ephemeris
    ts = load.timescale(builtin=True)
    eph = _load_ephemeris()
    sun, moon, earth = eph['sun'], eph['moon'], eph['earth']
    obs = earth + Topos(latitude_degrees=lat, longitude_degrees=lon)

    def moonfunc(npdate):
        pydate = npdate.astype(datetime.datetime).astimezone(utc)
        t = ts.utc(pydate)
        e = earth.at(t)

        _, slon, _ = e.observe(sun).apparent().ecliptic_latlon()
        _, mlon, _ = e.observe(moon).apparent().ecliptic_latlon()
        phase = mlon.degrees - slon.degrees  # 0 = new moon, 180 = full moon
        correct_phase = not (45.0 < (phase % 180.0) < 135.0)  # correct = close to new or full moon

        alt, _, _ = obs.at(t).observe(moon).apparent().altaz()
        above_horizon = alt.degrees > 0

        return correct_phase and above_horizon

    return moonfunc
