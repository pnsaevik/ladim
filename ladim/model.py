import importlib

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ladim.grid import Grid
    from ladim.forcing import Forcing
    import ladim.ibms
    import ladim.output
    import ladim.release
    import ladim.state
    import ladim.tracker
    import ladim.solver

DEFAULT_MODULES = dict(
    grid='ladim.gridforce.Grid',
    forcing='ladim.gridforce.Forcing',
    release='ladim.release.Releaser',
    state='ladim.state.State',
    output='ladim.output.Output',
    ibm='ladim.ibms.IBM',
    tracker='ladim.tracker.Tracker',
    solver='ladim.solver.Solver',
)


class Model:
    def __init__(self, config):
        module_names = (
            'grid', 'forcing', 'release', 'state', 'output', 'ibm', 'tracker',
            'solver',
        )

        self.modules = dict()
        for name in module_names:
            self.add_module(name, config[name])

    def add_module(self, name, conf):
        module_name = conf.get('module', DEFAULT_MODULES[name])
        conf_without_module = {
            k: v for k, v in conf.items()
            if k != 'module'
        }

        cls = load_class(module_name)
        self.modules[name] = cls(self, **conf_without_module)

    @property
    def grid(self) -> "Grid":
        return self.modules.get('grid', None)

    @property
    def forcing(self) -> "Forcing":
        return self.modules.get('forcing', None)

    @property
    def release(self) -> "ladim.release.Releaser":
        return self.modules.get('release', None)

    @property
    def state(self) -> "ladim.state.State":
        return self.modules.get('state', None)

    @property
    def output(self) -> "ladim.output.Output":
        return self.modules.get('output', None)

    @property
    def ibm(self) -> "ladim.ibms.IBM":
        return self.modules.get('ibm', None)

    @property
    def tracker(self) -> "ladim.tracker.Tracker":
        return self.modules.get('tracker', None)

    @property
    def solver(self) -> "ladim.solver.Solver":
        return self.modules.get('solver', None)

    def __getitem__(self, item):
        return self.modules[item]

    def __contains__(self, item):
        return item in self.modules

    def run(self):
        self.solver.run()

    def close(self):
        for m in self.modules.values():
            if hasattr(m, 'close') and callable(m.close):
                m.close()


def load_class(name):
    pkg, cls = name.rsplit(sep='.', maxsplit=1)
    return getattr(importlib.import_module(pkg), cls)


class Module:
    def __init__(self, model: Model):
        self._model = model

    @property
    def model(self):
        return self._model
