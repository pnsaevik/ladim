import numpy as np


class Solver:
    def __init__(self, modules, start, stop, step, order=None, seed=None):
        self.order = order or ('release', 'forcing', 'tracker', 'ibm', 'output')
        self.modules = modules
        self.start = np.datetime64(start, 's')
        self.stop = np.datetime64(stop, 's')
        self.step = np.timedelta64(step, 's')
        self.time = None

        if seed is not None:
            np.random.seed(seed)

    def run(self):
        modules = [self.modules[k] for k in self.order if k in self.modules]

        self.time = self.start
        while self.time <= self.stop:
            for m in modules:
                m.update()
            self.time += self.step
