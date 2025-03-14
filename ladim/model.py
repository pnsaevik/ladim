from ladim.ibms import IBM
from ladim.solver import Solver
from ladim.release import Releaser
from ladim.grid import Grid
from ladim.forcing import Forcing
from ladim.state import State
from ladim.tracker import Tracker
from ladim.output import Output


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

        output = Output(**config['output'])
        ibm = IBM(**config['ibm'])
        solver = Solver(**config['solver'])

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
