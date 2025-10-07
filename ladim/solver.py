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

    @staticmethod
    def create(start, stop, step, seed=None):
        return Solver(start, stop, step, seed)

    def run(self, model: "Model"):
        self.time = self.start

        if model.release.warm_start_file is not None:
            self.time = model.release.warm_start_time()
            model.release.from_warm_start_file(model)
            model.forcing.prepare_warm_start(model)
            model.output.writer.prepare_warm_start(model.release.warm_start_file)
            model.tracker.update(model)
            model.ibm.update(model)
            self.time += self.step

        while self.time <= self.stop:
            model.release.update(model)
            model.forcing.update(model)
            model.output.update(model)
            model.tracker.update(model)
            model.ibm.update(model)

            self.time += self.step
