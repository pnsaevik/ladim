from .model import Model, Module


class Forcing(Module):
    def __init__(self, model: Model):
        super().__init__(model)

    def velocity(self, X, Y, Z, tstep=0.0):
        raise NotImplementedError


class RomsForcing(Forcing):
    def __init__(self, model: Model, **conf):
        super().__init__(model)

        from ladim.gridforce.ROMS import Forcing as LegacyForcing

        grid_ref = GridReference(model)
        legacy_conf = dict(
            gridforce=dict(
                input_file=conf['input_file'],
            ),
            ibm_forcing=conf.get('ibm_forcing', []),
            start_time=conf.get('start_time', None),
            stop_time=conf.get('stop_time', None),
            dt=conf.get('dt', None),
        )

        # Allow gridforce module in current directory
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        # Import correct gridforce_module
        self.forcing = LegacyForcing(legacy_conf, grid_ref)
        # self.steps = self.forcing.steps
        # self.U = self.forcing.U
        # self.V = self.forcing.V

    def update(self):
        elapsed = self.model.solver.time - self.model.solver.start
        t = elapsed // self.model.solver.step

        return self.forcing.update(t)

    def velocity(self, X, Y, Z, tstep=0.0):
        return self.forcing.velocity(X, Y, Z, tstep=tstep)

    def field(self, X, Y, Z, name):
        return self.forcing.field(X, Y, Z, name)

    def close(self):
        return self.forcing.close()


class GridReference:
    def __init__(self, modules: Model):
        self.modules = modules

    def __getattr__(self, item):
        return getattr(self.modules.grid.grid, item)
