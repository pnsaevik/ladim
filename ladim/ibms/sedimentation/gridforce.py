import ladim.gridforce.ROMS


class Grid(ladim.gridforce.ROMS.Grid):
    def __init__(self, config):
        super().__init__(config)

    def sample_depth(self, X, Y):
        from scipy.ndimage import map_coordinates
        i = X - self.i0
        j = Y - self.j0
        return map_coordinates(self.H, [j, i], order=1)


class Forcing(ladim.gridforce.ROMS.Forcing):
    def __init__(self, config, grid):
        super().__init__(config, grid)
