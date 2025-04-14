import typing
if typing.TYPE_CHECKING:
    from ladim.model import Model


class Forcing:
    @staticmethod
    def from_roms(**conf):
        return RomsForcing(**conf)

    def velocity(self, X, Y, Z, tstep=0.0):
        raise NotImplementedError

    def update(self, model: "Model"):
        raise NotImplementedError


class RomsForcing(Forcing):
    def __init__(self, file, variables=None, **conf):
        """
        Forcing module which uses output data from the ROMS ocean model

        :param file: Glob pattern for the input files
        :param variables: A mapping of variable names to interpolation
        specifications. Each interpolaction specification consists of 0-4
        of the letters "xyzt". Coordinates that are listed in the string are
        interpolated linearly, while the remaining ones use nearest-neighbor
        interpolation. Some default configurations are defined:

        .. code-block:: json
            {
                "temp": "xyzt",
                "salt": "xyzt",
                "u": "xt",
                "v": "yt",
                "w": "zt",
            }


        :param conf: Legacy config dict
        """
        # Apply default interpolation configs
        variables = variables or dict()
        default_vars = dict(u="xt", v="yt", w="zt", temp="xyzt", salt="xyzt")
        self.variables = {**default_vars, **variables}

        grid_ref = GridReference()
        legacy_conf = dict(
            gridforce=dict(input_file=file, **conf),
            ibm_forcing=conf.get('ibm_forcing', []),
            start_time=conf.get('start_time', None),
            stop_time=conf.get('stop_time', None),
            dt=conf.get('dt', None),
        )
        if conf.get('subgrid', None) is not None:
            legacy_conf['gridforce']['subgrid'] = conf['subgrid']

        from .utilities import load_class
        LegacyForcing = load_class(conf.get('legacy_module', 'ladim.gridforce.ROMS.Forcing'))

        # Allow gridforce module in current directory
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        # Import correct gridforce_module
        self.forcing = LegacyForcing(legacy_conf, grid_ref)
        # self.steps = self.forcing.steps
        # self.U = self.forcing.U
        # self.V = self.forcing.V

    def update(self, model: "Model"):
        elapsed = model.solver.time - model.solver.start
        t = elapsed // model.solver.step

        # noinspection PyProtectedMember
        self.forcing._grid.modules = model
        self.forcing.update(t)

        # Update state variables by sampling the field
        x, y, z = model.state['X'], model.state['Y'], model.state['Z']
        for v in self.variables:
            if v in model.state:
                model.state[v] = self.field(x, y, z, v)

    def velocity(self, X, Y, Z, tstep=0.0):
        return self.forcing.velocity(X, Y, Z, tstep=tstep)

    def field(self, X, Y, Z, name):
        return self.forcing.field(X, Y, Z, name)

    def close(self):
        return self.forcing.close()


class GridReference:
    def __init__(self):
        self.modules = None

    def __getattr__(self, item):
        return getattr(self.modules.grid.grid, item)
