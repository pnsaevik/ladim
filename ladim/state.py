from .model import Model, Module


class State(Module):
    def __init__(self, model: Model):
        super().__init__(model)

    @property
    def size(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __len__(self):
        return self.size


class WarmStartState(State):
    def __init__(self, model: Model, **conf):
        super().__init__(model)
        from .legacy.state import State as LegacyState
        self._state = LegacyState(model, **conf)

    def update(self):
        self._state.update()

    def append(self, *args):
        self._state.append(*args)

    def warm_start(self, config, grid):
        self._state.warm_start(self, config, grid)

    @property
    def size(self):
        return len(self._state)

    @property
    def timestep(self):
        return self._state.timestep

    @property
    def timestamp(self):
        return self._state.timestamp

    @property
    def ibm_variables(self):
        return self._state.ibm_variables

    @property
    def instance_variables(self):
        return self._state.instance_variables

    def __getitem__(self, item):
        return self._state[item]

    def __setitem__(self, item, value):
        self._state[item] = value

    @property
    def pid(self):
        return self._state.pid

    @pid.setter
    def pid(self, value):
        self._state.pid = value

    @property
    def X(self):
        return self._state.X

    @X.setter
    def X(self, value):
        self._state.X = value

    @property
    def Y(self):
        return self._state.Y

    @Y.setter
    def Y(self, value):
        self._state.Y = value

    @property
    def Z(self):
        return self._state.Z

    @Z.setter
    def Z(self, value):
        self._state.Z = value

    @property
    def alive(self):
        return self._state.alive

    @alive.setter
    def alive(self, value):
        self._state.alive = value
