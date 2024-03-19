import pandas as pd
import numpy as np
from .model import Model, Module


class State(Module):
    def __init__(self, model: Model):
        """
        The state module contains static and dynamic particle properties

        The other modules interact with the state module mostly through
        the getitem and setitem methods. For instance, to increase the
        depth of all particles by 1, use

        >>> model.state['Z'] += 1

        :param model: Parent model
        """
        super().__init__(model)

    @property
    def size(self):
        """
        Current number of particles
        """
        raise NotImplementedError

    @property
    def released(self):
        """
        Total number of released particles
        """
        raise NotImplementedError

    def append(self, particles: dict):
        """
        Add new particles

        Missing variables are assigned a default value of 0.

        :param particles: A mapping from variable names to values
        """
        raise NotImplementedError

    def remove(self, particles):
        """
        Remove particles

        :param particles: Boolean index of particles to remove
        :return:
        """
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __len__(self):
        return self.size

    def __contains__(self, item):
        raise NotImplementedError


class DynamicState(State):
    def __init__(self, model: Model):
        super().__init__(model)

        self._num_released = 0
        self._varnames = set()

        self._data = pd.DataFrame()

    @property
    def released(self):
        return self._num_released

    def append(self, particles: dict):
        # If there are no new particles, do nothing
        if not particles:
            return

        num_new_particles = next(len(v) for v in particles.values())
        particles['pid'] = np.arange(num_new_particles) + self._num_released
        particles['alive'] = np.ones(num_new_particles, dtype=bool)
        if 'active' in particles:
            particles['active'] = np.array(particles['active'], dtype=bool)
        else:
            particles['active'] = np.ones(num_new_particles, dtype=bool)

        new_particles = pd.DataFrame(data=particles)
        self._data = pd.concat(
            objs=[self._data, new_particles],
            axis='index',
            ignore_index=True,
            join='outer',
        )

        self._num_released += num_new_particles

    def remove(self, particles):
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

    def __getattr__(self, item):
        if item not in self:
            raise AttributeError(f'Attribute not defined: {item}')
        return self[item]

    def __contains__(self, item):
        return item in self._data

    @property
    def dt(self):
        """Backwards-compatibility function for returning model.solver.step"""
        return self.model.solver.step

    @property
    def timestamp(self):
        """Backwards-compatibility function for returning solver time as numpy datetime"""
        return np.int64(self.model.solver.time).astype('datetime64[s]')
