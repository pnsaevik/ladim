import numpy as np

from ladim.ibms import light

# from ladim2.ibm import BaseIBM
# from ladim2.timekeeper import TimeKeeper
# from ladim2.state import State
# from ladim2.forcing import BaseForce
# from ladim2.grid import BaseGrid


# class IBM(BaseIBM):
class IBM:
    #def __init__(self, modules, **kwargs):
    def __init__(self, modules, vertical_mixing=0, salinity_model="new"):
    #     self,
    #     timer: TimeKeeper,
    #     state: State,
    #     forcing: BaseForce,
    #     grid: BaseGrid,
    #     vertical_mixing: float = 0,
    #     salinity_model: str = "new",
    # ):

        # Modules
        self.timer = modules["time"]
        self.state = modules["state"]
        self.forcing = modules["forcing"]
        self.grid = modules["grid"]

        # Keyword arguments
        self.vertical_mixing = vertical_mixing
        self.new_salinity_model = salinity_model == "True"

        # Constants
        mortality = 0.17  # [days-1]

        # mm2m = 0.001
        # g = 9.81
        # tempB = 7.0  # set default temperature

        self.k = 0.2  # Light extinction coefficient
        self.swim_vel = 5e-4  # m/s
        self.vertical_diffusion = self.vertical_mixing > 0

        self.dt = self.timer.dt / np.timedelta64(1, 'D')  # Timestep in days
        self.mortality_factor = np.exp(-mortality * self.dt / 86400)

        # salinity_model = config["ibm"].get('salinity_model', 'new')
        # self.new_salinity_model = (salinity_model == 'new')
        #
    def update(self):

        state = self.state
        forcing = self.forcing

        # Mortality
        state["super"] *= self.mortality_factor

        # Update forcing
        # forcing.force_particles(state.X, state.Y, state.Z)
        forcing.force_particles(state.X, state.Y)
        state["temp"] = forcing.variables["temp"]
        state["salt"] = forcing.variables["salt"]

        # Age in degree-days
        state["age"] += state.temp * self.dt / 86400
        state["days"] += 1.0 * (self.dt / 86400)

        # Light at depth
        lon, lat = self.grid.lonlat(state.X, state.Y)
        light0 = light.surface_light(self.timer.time, lon, lat)
        Eb = light0 * np.exp(-self.k * state.Z)
        # Eb = 0.3

        # Swimming velocity
        W = np.zeros_like(state.X)
        # Upwards if light enough (decreasing depth)
        W[Eb >= 0.01] = -self.swim_vel

        if self.new_salinity_model:
            # Mixture of down/up if salinity between 23 and 31
            # Downwards if salinity < 31
            salt_limit = np.random.uniform(23, 31, W.shape)
        else:
            # Downwards if salinity < 20
            salt_limit = 20

        W[state.salt < salt_limit] = self.swim_vel

        # Random diffusion velocity
        if self.vertical_diffusion:
            rand = np.random.normal(size=len(W))
            W += rand * (2 * self.D / self.dt) ** 0.5

        # Update vertical position, using reflexive boundary condition at the top
        state["Z"] += W * self.dt
        state["Z"][state.Z < 0] *= -1

        # For z-version, do not go below 20 m
        state["Z"][state.Z >= 20.0] = 19.0

        # Mark particles older than 200 degree days as dead
        state["alive"] &= state.age < 200
