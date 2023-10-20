import numpy as np


class TimeStepper:
    def __init__(self, modules, start, stop, step, order=None, seed=None):
        self.order = order or ('forcing', 'release', 'output', 'ibm', 'tracker')
        self.modules = modules
        self.start = np.datetime64(start)
        self.stop = np.datetime64(stop)
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
