import importlib
import importlib.util
import sys
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ladim.grid import Grid
    from ladim.forcing import Forcing
    from ladim.ibms import IBM
    from ladim.output import Output
    from ladim.release import Releaser
    from ladim.state import State
    from ladim.tracker import Tracker
    from ladim.solver import Solver

DEFAULT_MODULES = dict(
    grid='ladim.grid.RomsGrid',
    forcing='ladim.forcing.RomsForcing',
    release='ladim.release.TextFileReleaser',
    state='ladim.state.DynamicState',
    output='ladim.output.RaggedOutput',
    ibm='ladim.ibms.IBM',
    tracker='ladim.tracker.HorizontalTracker',
    solver='ladim.solver.Solver',
)


class Model:
    """
    The Model class represents the entire simulation model. The different
    submodules control the simulation behaviour. In particular, the solver
    submodule controls the execution flow while the other submodules are
    called once every time step within the main simulation loop.
    """

    def __init__(
            self, grid: "Grid", forcing: "Forcing", release: "Releaser",
            state: "State", output: "Output", ibm: "IBM", tracker: "Tracker",
            solver: "Solver",
    ):
        self.grid = grid
        self.forcing = forcing
        self.release = release
        self.state = state
        self.output = output
        self.ibm = ibm
        self.tracker = tracker
        self.solver = solver

    @staticmethod
    def from_config(config: dict) -> "Model":
        """
        Initialize a model class by supplying the configuration parameters
        of each submodule.

        :param config: Configuration parameters for each submodule
        :return: An initialized Model class
        """

        # Create new version of the config dict without the 'model' keyword
        def remove_module_key(d: dict):
            return {k: v for k, v in d.items() if k != 'module'}

        # Initialize modules
        module_names = (
            'grid', 'forcing', 'release', 'state', 'output', 'ibm', 'tracker',
            'solver',
        )
        modules = dict()
        for name in module_names:
            subconf = config.get(name, dict())
            modules[name] = Module.from_config(
                conf=remove_module_key(subconf),
                module=subconf.get('module', DEFAULT_MODULES[name]),
            )

        # Initialize model
        return Model(**modules)

    @property
    def modules(self) -> dict:
        return dict(
            grid=self.grid,
            forcing=self.forcing,
            release=self.release,
            state=self.state,
            output=self.output,
            ibm=self.ibm,
            tracker=self.tracker,
            solver=self.solver,
        )

    def run(self):
        self.solver.run(self)

    def close(self):
        for m in self.modules.values():
            if hasattr(m, 'close') and callable(m.close):
                m.close()


def load_class(name):
    pkg, cls = name.rsplit(sep='.', maxsplit=1)

    # Check if "pkg" is an existing file
    spec = None
    module_name = None
    file_name = pkg + '.py'
    if Path(file_name).exists():
        # This can return None if there were import errors
        module_name = pkg
        spec = importlib.util.spec_from_file_location(module_name, file_name)

    # If pkg can not be interpreted as a file, use regular import
    if spec is None:
        return getattr(importlib.import_module(pkg), cls)

    # File import
    else:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, cls)


class Module:
    @staticmethod
    def from_config(conf: dict, module: str) -> "Module":
        """
        Initialize a module using a configuration dict.

        :param conf: The configuration parameters of the module
        :param module: The fully qualified name of the module
        :return: An initialized module
        """
        cls = load_class(module)
        return cls(**conf)

    def update(self, model: Model):
        pass

    def close(self):
        pass
