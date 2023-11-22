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
    def __init__(self, model: Model):
        super().__init__(model)

        self._num_released = 0
        self._varnames = set()

        self._data = pd.DataFrame()

    @property
    def num_released(self):
        return self._num_released

    def update(self):
        pass

    def variables(self):
        return self._varnames

    def append(self, particles):
        num_new_particles = next(len(v) for v in particles.values())
        particles['pid'] = np.arange(num_new_particles) + self._num_released
        particles['alive'] = np.ones(num_new_particles, dtype=bool)

        new_particles = pd.DataFrame(data=particles)
        self._data = pd.concat(
            objs=[self._data, new_particles],
            axis='index',
            ignore_index=True,
            join='outer',
        )

        self._num_released += num_new_particles
        self._varnames.update(particles.keys())

    def kill(self, particles):
        if not np.any(particles):
            return

        keep = ~particles
        self._data = self._data.iloc[keep]

    @property
    def size(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item].values

    def __setitem__(self, item, value):
        self._data[item] = value
