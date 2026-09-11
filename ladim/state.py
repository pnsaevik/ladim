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
        self._data = dict()  # type: dict[str, np.ndarray]

    @property
    def size(self):
        """
        Current number of particles
        """
        keys = list(self._data.keys())
        if len(keys) == 0:
            return 0
        return len(self._data[keys[0]])

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

        # Check that input has correct format
        num_new = next(len(v) for v in particles.values())
        fields = {}  # type: dict[str, np.ndarray]
        for k, v in particles.items():
            fields[k] = np.asarray(v)
            if not np.shape(fields[k]) == (num_new, ):
                raise ValueError('Unequal number of array elements in input')

        _add_standard_variables(fields, first_pid=self._num_released)

        self._data = _append_fields(self._data, fields)
        self._num_released += num_new

    def remove(self, particles):
        """
        Remove particles

        :param particles: Boolean index of particles to remove
        :return:
        """
        if not np.any(particles):
            return

        keep = ~particles
        for k in self._data.keys():
            self._data[k] = self._data[k][keep]

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value):
        v = np.asarray(value)
        assert v.shape == (self.size, )
        self._data[item] = v

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
            'timestamp', 'released'
        ]
        if item in list(self.__dict__.keys()) + excepted_values:
            super().__setattr__(item, value)
        elif item in self._data:
            self._data[item] = value
        else:
            raise AttributeError(f"Attribute not defined: '{item}'")


def _add_standard_variables(fields: dict[str, np.ndarray], first_pid: int):
    """
    Adds standard variables pid, alive and active to a set of particles.

    Modifies the input dictionary.
    """
    num_new = next(len(v) for v in fields.values())

    # Add standard variables
    fields['pid'] = np.arange(num_new) + first_pid
    fields['alive'] = np.ones(num_new, dtype=bool)
    if 'active' in fields:
        fields['active'] = np.array(fields['active'], dtype=bool)
    else:
        fields['active'] = np.ones(num_new, dtype=bool)


def _append_fields(
        oldf: dict[str, np.ndarray],
        newf: dict[str, np.ndarray]):
    """
    Append a new set of fields to an old set.

    In this context, "fields" is a dict of one-dimensional numpy arrays, each
    having the same length. The new field set has the same keys as the union
    of the two old ones.
    """

    # Find columns of the new field set
    old_cols = list(oldf)
    extra_cols = [c for c in newf if c not in old_cols]
    cols = old_cols + extra_cols

    num_old = 0 if len(oldf) == 0 else next(len(v) for v in oldf.values())
    num_new = 0 if len(newf) == 0 else next(len(v) for v in newf.values())

    # Concatenate old and new particles
    newdata = {}  # type: dict[str, np.ndarray]
    for c in cols:
        dtype = oldf[c].dtype if c in oldf else newf[c].dtype
        oldv = oldf[c] if c in oldf else np.zeros(num_old, dtype=dtype)
        newv = newf[c] if c in newf else np.zeros(num_new, dtype=dtype)
        newdata[c] = np.concatenate((oldv, newv), dtype=dtype)

    return newdata
