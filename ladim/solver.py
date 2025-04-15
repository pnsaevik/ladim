import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ladim.model import Model


class Solver:
    def __init__(self, start, stop, step, seed=None):
        self.start = np.datetime64(start, 's').astype('int64')
        self.stop = np.datetime64(stop, 's').astype('int64')
        self.step = np.timedelta64(step, 's').astype('int64')
        self.time = None

        if seed is not None:
            np.random.seed(seed)

    def run(self, model: "Model"):
        self.time = self.start
        while self.time <= self.stop:
            model.release.update(model)
            model.forcing.update(model)
            model.output.update(model)
            model.tracker.update(model)
            model.ibm.update(model)

            self.time += self.step
