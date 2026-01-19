import numpy as np
from ladim.ibms.light import light
from ladim.ibms.eos import viscosity, density
import typing


class IBM:

    def __init__(self, config):
        # --- Adjustable parameters ---

        # Light extinction coefficient [1/m], exponential decay rate of daylight by depth
        # coeff = 0  ==>  Crystal clear water
        # coeff = k  ==>  Daylight reduced by a factor exp(-k) after 1 meter
        self.k = config['ibm'].get('extinction_coeff', 0.2)

        # Vertical mixing [m^2/s]
        # diff = 0  ==>  No vertical diffusion
        # diff = 1e-2  ==> Wave-induced mixing
        # diff = 1e-5  ==> Stratified water
        self.D = config['ibm'].get('vertical_mixing', 0)

        # --- Species-specific parameters ---
        #
        # New species can be added to the list below. In the ladim config file,
        # users can set, e.g., `ibm.species = "cod"` to get the default values
        # for cod. Users can change the default values by setting individual
        # parameters directly. If the user specifies `ibm.species = "cod"` and
        # `ibm.hatch_day = 90`, the explicitly defined hatch day takes precedence.

        species_defaults = dict(
            cod=dict(
                egg_diam=0.0014,  # Egg diameter [m]
                hatch_day=93.7,   # Hatch day [degree days]
                swim_speed=0.1,   # Vertical swimming speed [body lengths/second]
                light=1,          # Desired light level [µmol photons/s]
                min_depth=0,      # Minimum depth [m]
                max_depth=1000,   # Maximal depth [m]
                init_larvae_weight=9.3e-2,  # Larvae weight at hatching [mg]
                growth=growth_cod_larvae,   # Larvae growth function
                length=weight_to_length,    # Larvae weight to length function
            ),
            saithe=dict(
                egg_diam=0.0011,
                hatch_day=60,
                swim_speed=0.2,
                light=1,
                init_larvae_weight=9.3e-2,
                min_depth=30,
                max_depth=60,
                growth=growth_cod_larvae,
                length=weight_to_length,
            ),
        )

        self.species = config['ibm'].get('species', 'unknown')

        def read_species_param(p):
            if p in config['ibm']:
                return config['ibm'][p]
            else:
                return species_defaults[self.species][p]

        self.egg_diam = read_species_param('egg_diam')
        self.hatch_day = read_species_param('hatch_day')
        self.init_larvae_weight = read_species_param('init_larvae_weight')  # type: float
        self.swim_speed = read_species_param('swim_speed')
        self.desired_light = read_species_param('light')
        self.min_depth = read_species_param('min_depth')  # type: float
        self.max_depth = read_species_param('max_depth')  # type: float
        self.growth = read_species_param('growth')  # type: typing.Any
        self.length = read_species_param('length')  # type: typing.Any

        self.dt = config['dt']

    def update_ibm(self, grid, state, forcing):
        # --- Update forcing ---
        state['temp'] = forcing.field(state.X, state.Y, state.Z, 'temp')
        state['salt'] = forcing.field(state.X, state.Y, state.Z, 'salt')

        # --- Ageing ---
        is_egg = state.age <= self.hatch_day
        state['age'] += state.temp * state.dt / 86400

        # --- Larvae growth ---
        temp_larvae = state.temp[~is_egg]
        weight = np.maximum(state['weight'][~is_egg], self.init_larvae_weight)
        state['weight'][~is_egg] = weight + self.growth(temp_larvae, weight, self.dt)

        # --- Egg sinking velocity ---
        W = np.zeros(np.shape(is_egg), dtype=np.float32)
        T, S = state.temp[is_egg], state.salt[is_egg]
        W[is_egg] = sinkvel_egg(
            mu_w=viscosity(T, S),
            dens_w=density(T, S),
            dens_egg=density(T, state.egg_buoy[is_egg]),
            diam_egg=self.egg_diam,
        )

        # --- Larvae swimming velocity ---
        lon, lat = grid.lonlat(state.X[~is_egg], state.Y[~is_egg])
        Z_larvae = state.Z[~is_egg]
        Eb = light(state.timestamp, lon, lat, depth=Z_larvae, extinction_coef=self.k)
        length = 0.001 * self.length(state.weight[~is_egg])
        W[~is_egg] = self.swim_speed * length * np.sign(Eb - self.desired_light)

        # --- Vertical turbulent mixing ---
        if self.D:
            W += np.random.normal(size=len(W)) * np.sqrt(2 * self.D / self.dt)

        # --- Execute vertical movement ---
        Z = state.Z + W * self.dt
        Z = np.clip(Z, self.min_depth, self.max_depth)
        state['Z'] = Z


def growth_cod_larvae(temp, weight, dt):
    """
    Incremental growth of NA cod larvae as reported by Folkvord (2005):
    "Comparison of size-at-age of larval Atlantic cod (Gadus morhua) from different
    populations based on size- and temperature-dependent growth models". Canadian Journal
    of Fisheries and Aquatic Sciences 62(5). https://doi.org/10.1139/f05-008

    :param temp: Temperature [degrees Celcius]
    :param weight: Larvae dry weight [mg]
    :param dt: Growth time [s]
    :return: Dry weight increase [mg]
    """

    w = np.log(weight)
    sec2day = 1 / 86400
    GR_percent = (1.08 + temp * (1.79 + w * (- 0.074 + w * (-0.0965 + w * 0.0112))))
    GR = np.log(1 + 0.01 * GR_percent)
    return (np.exp(GR * sec2day * dt) - 1) * weight


def weight_to_length(weight):
    """
    Compute larvae length from dry weight using equation (6) of Folkvord (2005):
    "Comparison of size-at-age of larval Atlantic cod (Gadus morhua) from different
    populations based on size- and temperature-dependent growth models". Canadian Journal
    of Fisheries and Aquatic Sciences 62(5). https://doi.org/10.1139/f05-008

    :param weight: Dry weight [mg]
    :return: Body length of larvae [mm]
    """

    w = np.log(weight)
    return np.exp(2.296 + w * (0.277 - w * 0.005128))


def sinkvel_egg(mu_w, dens_w, dens_egg, diam_egg):
    # Calculate maximum diameter in Stokes formula
    dens_diff = dens_w - dens_egg
    dmax = (
        (9.0 * mu_w * mu_w) / (1025.0 * 9.81 * (np.abs(dens_diff) + 1e-16))
    ) ** (1 / 3)

    small_W = -(1 / 18) * (1 / mu_w) * 9.81 * (diam_egg ** 2) * dens_diff
    large_W = (
        -0.08825 * (diam_egg - 0.4 * dmax) * np.abs(dens_diff) ** (2 / 3)
        * mu_w ** (-1 / 3) * np.sign(dens_diff)
    )

    return np.where(diam_egg <= dmax, small_W, large_W)
