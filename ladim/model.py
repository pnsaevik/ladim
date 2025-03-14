import importlib
import importlib.util
import sys
from pathlib import Path

from typing import TYPE_CHECKING, Hashable, Any
if TYPE_CHECKING:
    from ladim.ibms import IBM
    from ladim.output import RaggedOutput as Output
    from ladim.solver import Solver


from ladim.release import Releaser
from ladim.grid import Grid
from ladim.forcing import Forcing
from ladim.state import State
from ladim.tracker import Tracker


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

        grid = Grid.from_roms(**config['grid'])
        forcing = Forcing.from_roms(**config['forcing'])

        release = Releaser.from_textfile(
            lonlat_converter=grid.ll2xy, **config['release']
        )
        tracker = Tracker.from_config(**config['tracker'])

        output = Module.from_config(config['output'])
        ibm = Module.from_config(config['ibm'])
        solver = Module.from_config(config['solver'])

        state = State()

        # noinspection PyTypeChecker
        return Model(grid, forcing, release, state, output, ibm, tracker, solver)

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
    def from_config(conf: dict) -> "Module":
        """
        Initialize a module using a configuration dict.

        The configuration dict should contain the keyword ``module``,
        which is the fully qualified name of the module. The other
        keys should be the named parameters of the module's init
        method.

        :param conf: The configuration parameters of the module
        :return: An initialized module
        """
        conf2, module = dict_pop_pure(conf, 'module')
        cls = load_class(module)
        return cls(**conf2)

    def update(self, model: Model):
        pass

    def close(self):
        pass


def dict_pop_pure(d: dict, key: Hashable) -> tuple[dict, Any]:
    """
    Same as dict.pop, but does not modify the input dict

    :param d: Input dict
    :param key: Key to pop
    :return: A tuple (d2, val) where d2 is the dict without the key,
        and val is d[key]
    """
    d2 = {k: v for k, v in d.items() if k != key}
    val = d[key]
    return d2, val
