from ladim2.grid import BaseGrid

import numpy as np


class Grid(BaseGrid):
    def __init__(self, **args):
        pass

    def depth(self, X, Y):
        return 100 + np.zeros_like(X)

    def metric(self, X, Y):
        return np.ones_like(X)

    def ingrid(self, X, Y):
        return (0 < X) and (X < 100) and (0 < Y) and (Y < 100)

    def atsea(self, X, Y):
        return X == X  # True of correct shape
