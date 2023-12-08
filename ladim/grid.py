from .model import Model, Module


class Grid(Module):
    def __init__(self, model: Model):
        super().__init__(model)

    def ingrid(self, X, Y):
        raise NotImplementedError

    def sample_metric(self, X, Y):
        raise NotImplementedError

    def atsea(self, X, Y):
        raise NotImplementedError

    def ll2xy(self, lon, lat):
        raise NotImplementedError


class RomsGrid(Grid):
    def __init__(self, model: Model, **conf):
        super().__init__(model)

        legacy_conf = dict(
            gridforce=dict(
                input_file=conf['dataset'],
            ),
            start_time=conf.get('start_time', None),
        )

        from ladim.gridforce.ROMS import Grid as LegacyGrid

        # Allow gridforce module in current directory
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        # Import correct gridforce_module
        # gridforce_module = importlib.import_module(config["gridforce_module"])
        self.grid = LegacyGrid(legacy_conf)
        self.xmin = self.grid.xmin
        self.xmax = self.grid.xmax
        self.ymin = self.grid.ymin
        self.ymax = self.grid.ymax

    def sample_metric(self, X, Y):
        """Sample the metric coefficients"""
        return self.grid.sample_metric(X, Y)

    def sample_depth(self, X, Y):
        """Return the depth of grid cells"""
        return self.grid.sample_depth(X, Y)

    def lonlat(self, X, Y, method=None):
        """Return the longitude and latitude from grid coordinates"""
        return self.grid.lonlat(X, Y, method=method)

    def ingrid(self, X, Y):
        """Returns True for points inside the subgrid"""
        return self.grid.ingrid(X, Y)

    def onland(self, X, Y):
        """Returns True for points on land"""
        return self.grid.onland(X, Y)

    # Error if point outside
    def atsea(self, X, Y):
        """Returns True for points at sea"""
        return self.grid.atsea(X, Y)

    def ll2xy(self, lon, lat):
        return self.grid.ll2xy(lon, lat)

    def xy2ll(self, X, Y):
        return self.grid.xy2ll(X, Y)
