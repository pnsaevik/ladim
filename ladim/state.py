import pandas as pd
import numpy as np
from .model import Model, Module


class State(Module):
    def __init__(self, model: Model):
        super().__init__(model)

    @property
    def size(self):
        raise NotImplementedError

    @property
    def num_released(self):
        raise NotImplementedError

    def append(self, particles):
        raise NotImplementedError

    def kill(self, particles):
        raise NotImplementedError

    def variables(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __contains__(self, item):
        return item in self.variables()

    def __len__(self):
        return self.size


class DynamicState(State):
    def __init__(self, model: Model, **conf):
        super().__init__(model)
        from .legacy.state import State as LegacyState
        self._state = LegacyState(model, **conf)
        self._num_released = 0
        self._varnames = set()

    @property
    def num_released(self):
        return self._num_released

    def update(self):
        self._state.update()

    def variables(self):
        return self._varnames

    def append(self, particles):
        num_new_particles = next(len(v) for v in particles.values())
        particles['pid'] = np.arange(num_new_particles) + self._num_released

        df = pd.DataFrame(data=particles)
        self._state.append(df, self.model.forcing)
        self._num_released += num_new_particles
        self._varnames.update(particles.keys())

        # Set alive status of new particles
        self._state.alive = self.model.grid.ingrid(self['X'], self['Y'])

    def kill(self, particles):
        if not np.any(particles):
            return

        keep = ~particles
        for v in self._state.instance_variables:
            self._state[v] = self._state[v][keep]

    def warm_start(self, config, grid):
        self._state.warm_start(self, config, grid)

    @property
    def size(self):
        return len(self._state)

    def __getitem__(self, item):
        return self._state[item]

    def __setitem__(self, item, value):
        self._state[item] = value
