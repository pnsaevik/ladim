import numpy as np
import typing

if typing.TYPE_CHECKING:
    from ..model import Model


class IBM:
    def __init__(self, legacy_module=None, conf: dict = None):
        from ..utilities import load_class

        if legacy_module is None:
            UserIbmClass = EmptyIBM
        else:
            UserIbmClass = load_class(legacy_module + '.IBM')

        self.user_ibm = UserIbmClass(conf or {})

    def update(self, model: "Model"):
        grid = model.grid
        state = model.state

        state.dt = model.solver.step
        state.timestamp = np.int64(model.solver.time).astype('datetime64[s]')
        state.timestep = (
                (model.solver.time - model.solver.start) // model.solver.step
        )

        forcing = model.forcing
        self.user_ibm.update_ibm(grid, state, forcing)


class EmptyIBM:
    def __init__(self, _):
        pass

    def update_ibm(self, grid, state, forcing):
        return
