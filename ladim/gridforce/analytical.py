import numpy as np


class Grid:
    def __init__(self, config):
        self.xmin = -np.inf
        self.xmax = np.inf
        self.ymin = -np.inf
        self.ymax = np.inf

    def sample_depth(self, X, Y):
        return np.zeros(np.shape(X), dtype='i2')

    def sample_metric(self, X, Y):
        m = np.ones(np.shape(X), dtype='i2')
        return m, m

    def lonlat(self, X, Y, method="bilinear"):
        return X, Y

    def ingrid(self, X, Y):
        return np.ones(np.shape(X), dtype=bool)

    def onland(self, X, Y):
        return np.zeros(np.shape(X), dtype=bool)

    def atsea(self, X, Y):
        return np.ones(np.shape(X), dtype=bool)

    def xy2ll(self, X, Y):
        return X, Y

    def ll2xy(self, lon, lat):
        return lon, lat


class Forcing:
    def __init__(self, config, grid):
        pass

    def update(self, t):
        pass

    def close(self):
        pass

    def velocity(self, X, Y, Z, tstep=0, method="bilinear"):
        u = np.ones(np.shape(X))
        return u, u

    def field(self, X, Y, Z, name):
        return np.zeros(np.shape(X))
