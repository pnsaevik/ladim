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
    Add the standard particle metadata fields ``pid``, ``alive`` and
    ``active`` in-place to a mapping of one-dimensional arrays.

    The input dictionary maps variable names to NumPy arrays, all of which are
    assumed to have the same length ``num_new``. The helper mutates that
    dictionary by adding the missing standard variables and normalizing the
    ``active`` field to a boolean array.

    :param fields: Mapping from field names to one-dimensional NumPy arrays.
        The arrays store values for the same set of newly released particles.
    :param first_pid: The first particle identifier to assign. The new
        ``pid`` field is created as ``np.arange(num_new) + first_pid``.
    :returns: The same dictionary object, updated in-place with the new
        standard variables ``pid``, ``alive`` and ``active``.

    .. note::
       ``alive`` is always created as a boolean array of ones, and ``active``
       is either converted from an existing array to boolean or created as an
       array of ones if the input dictionary does not already provide it.
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
    Append two field dictionaries by concatenating their per-variable arrays.

    The input dictionaries map a field name to a one-dimensional NumPy array.
    Each array stores particles for that variable, and the key sets are allowed
    to differ: the output contains the union of keys from ``oldf`` and ``newf``.
    For any variable present only in one input mapping, the missing side is
    padded with a zero-filled array of the appropriate dtype and length before
    the concatenation.

    :param oldf: Existing field mapping. The keys are variable names, and each
        value is a one-dimensional array of shape ``(n_old,)``.
    :param newf: New field mapping. The keys are variable names, and each
        value is a one-dimensional array of shape ``(n_new,)``.
    :returns: A new dictionary whose keys are the union of ``oldf`` and
        ``newf`` keys, and whose arrays have the shape ``(n_old + n_new,)``.
        Fields that only exist in either input are appended as zeros on the
        missing side before concatenation.

    .. note::
       This helper is used to build the particle state ``State._data`` by
       appending newly released particles to the previously stored particle
       fields.
    """

    # Find columns of the new field set
    old_cols = list(oldf)
    extra_cols = [c for c in newf if c not in old_cols]
    cols = old_cols + extra_cols

    num_old = 0 if len(oldf) == 0 else next(len(v) for v in oldf.values())
    num_new = 0 if len(newf) == 0 else next(len(v) for v in newf.values())
    num_total = num_old + num_new

    # Concatenate old and new particles
    newdata = {}  # type: dict[str, np.ndarray]
    for c in cols:
        dtype = oldf[c].dtype if c in oldf else newf[c].dtype
        newdata_c = np.empty(num_total, dtype=dtype)
        if c in oldf:
            newdata_c[:num_old] = oldf[c]
        else:
            newdata_c[:num_old] = 0
        if c in newf:
            newdata_c[num_old:] = newf[c]
        else:
            newdata_c[num_old:] = 0

        newdata[c] = newdata_c

    return newdata
