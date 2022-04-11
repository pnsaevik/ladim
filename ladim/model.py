import logging
import importlib


DEFAULT_MODULES = dict(
    grid='ladim.gridforce.Grid',
    forcing='ladim.gridforce.Forcing',
    release='ladim.release.Releaser',
    state='ladim.state.State',
    output='ladim.output.Output',
    ibm='ladim.ibms.IBM',
    tracker='ladim.tracker.Tracker',
)


class Model:
    def __init__(self, config):
        module_names = ('grid', 'forcing', 'release', 'state', 'output', 'ibm', 'tracker')

        self.numsteps = config['timestepper']['numsteps']
        self.timestep_order = ('release', 'forcing', 'output', 'tracker', 'ibm', 'state')
        self.modules = dict()
        for name in module_names:
            self.add_module(name, config[name])

    def add_module(self, name, conf):
        module_name = conf.get('module', DEFAULT_MODULES[name])
        Module = load_module(module_name)
        self.modules[name] = Module(self.modules, **conf)

    def run(self):
        modules = self.modules
        logging.info("Starting time loop")
        for step in range(self.numsteps + 1):
            for name in self.timestep_order:
                modules[name].update()

    def close(self):
        for m in self.modules.values():
            if hasattr(m, 'close') and callable(m.close):
                m.close()


def load_module(name):
    pkg, cls = name.rsplit(sep='.', maxsplit=1)
    return getattr(importlib.import_module(pkg), cls)
