import pandas as pd
import numpy as np


class State:
    """
    The state module contains static and dynamic particle properties

    The other modules interact with the state module mostly through
    the getitem and setitem methods. For instance, to increase the
    depth of all particles by 1, use state['Z'] += 1
    """

    def __init__(self):
        self._num_released = 0
        self._varnames = set()
        self._data = pd.DataFrame()

    @property
    def size(self):
        """
        Current number of particles
        """
        return len(self._data)

    @property
    def released(self):
        """
        Total number of released particles
        """
        return self._num_released

    def append(self, particles: dict):
        """
        Add new particles

        Missing variables are assigned a default value of 0.

        :param particles: A mapping from variable names to values
        """
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
        """
        Remove particles

        :param particles: Boolean index of particles to remove
        :return:
        """
        if not np.any(particles):
            return

        keep = ~particles
        self._data = self._data.iloc[keep]

    def __getitem__(self, item):
        return self._data[item].values

    def __setitem__(self, item, value):
        self._data[item] = value

    def __len__(self):
        return self.size

    def __contains__(self, item):
        return item in self._data

    def __getattr__(self, item):
        if item not in self:
            raise AttributeError(f'Attribute not defined: {item}')
        return self[item]

    def __setattr__(self, item, value):
        excepted_values = [
            '_data', '_model', '_num_released', '_varnames', 'dt', 'timestep',
            'timestamp'
        ]
        if item in list(self.__dict__.keys()) + excepted_values:
            super().__setattr__(item, value)
        elif item in self._data:
            self._data[item] = value
        else:
            raise AttributeError(f"Attribute not defined: '{item}'")
