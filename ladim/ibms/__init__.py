from ..model import Model, Module
import numpy as np


class IBM(Module):
    pass


class LegacyIBM(IBM):
    def __init__(self, legacy_module, conf):
        from ..model import load_class
        LegacyIbmClass = load_class(legacy_module + '.IBM')
        self._ibm = LegacyIbmClass(conf)

    def update(self, model: Model):
        grid = model.grid
        state = model.state

        state.dt = model.solver.step
        state.timestamp = np.int64(model.solver.time).astype('datetime64[s]')
        state.timestep = (
                (model.solver.time - model.solver.start) // model.solver.step
        )

        forcing = model.forcing
        self._ibm.update_ibm(grid, state, forcing)
