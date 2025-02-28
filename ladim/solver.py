import numpy as np


class Solver:
    def __init__(self, start, stop, step, order=None, seed=None):
        self.order = order or ('release', 'forcing', 'tracker', 'ibm', 'output')
        self.start = np.datetime64(start, 's').astype('int64')
        self.stop = np.datetime64(stop, 's').astype('int64')
        self.step = np.timedelta64(step, 's').astype('int64')
        self.time = None

        if seed is not None:
            np.random.seed(seed)

    def run(self, model):
        modules = model.modules
        ordered_modules = [modules[k] for k in self.order if k in modules]

        self.time = self.start
        while self.time <= self.stop:
            for m in ordered_modules:
                m.update(model)
            self.time += self.step
