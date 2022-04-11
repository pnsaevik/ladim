import importlib

import ladim.gridforce
import ladim.ibms
import ladim.output
import ladim.release
import ladim.state
import ladim.tracker
import ladim.timestepper

DEFAULT_MODULES = dict(
    grid='ladim.gridforce.Grid',
    forcing='ladim.gridforce.Forcing',
    release='ladim.release.Releaser',
    state='ladim.state.State',
    output='ladim.output.Output',
    ibm='ladim.ibms.IBM',
    tracker='ladim.tracker.Tracker',
    timestepper='ladim.timestepper.TimeStepper',
)


class Model:
    def __init__(self, config):
        module_names = (
            'grid', 'forcing', 'release', 'state', 'output', 'ibm', 'tracker',
            'timestepper',
        )

        self.modules = dict()
        for name in module_names:
            self.add_module(name, config[name])

    def add_module(self, name, conf):
        module_name = conf.get('module', DEFAULT_MODULES[name])
        Module = load_module(module_name)
        self.modules[name] = Module(self, **conf)

    @property
    def grid(self) -> ladim.gridforce.Grid:
        return self.modules.get('grid', None)

    @property
    def forcing(self) -> ladim.gridforce.Forcing:
        return self.modules.get('forcing', None)

    @property
    def release(self) -> ladim.release.Releaser:
        return self.modules.get('release', None)

    @property
    def state(self) -> ladim.state.State:
        return self.modules.get('state', None)

    @property
    def output(self) -> ladim.output.Output:
        return self.modules.get('output', None)

    @property
    def ibm(self) -> ladim.ibms.IBM:
        return self.modules.get('ibm', None)

    @property
    def tracker(self) -> ladim.tracker.Tracker:
        return self.modules.get('tracker', None)

    @property
    def timestepper(self) -> ladim.timestepper.TimeStepper:
        return self.modules.get('timestepper', None)

    def __getitem__(self, item):
        return self.modules[item]

    def __contains__(self, item):
        return item in self.modules

    def run(self):
        self.timestepper.run()

    def close(self):
        for m in self.modules.values():
            if hasattr(m, 'close') and callable(m.close):
                m.close()


def load_module(name):
    pkg, cls = name.rsplit(sep='.', maxsplit=1)
    return getattr(importlib.import_module(pkg), cls)
