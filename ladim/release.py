from .model import Model, Module
import numpy as np


class Releaser(Module):
    def __init__(self, model: Model):
        super().__init__(model)


class TextFileReleaser(Releaser):
    def __init__(self, model: Model, **conf):
        super().__init__(model)

        from .legacy.release import ParticleReleaser as LegacyReleaser
        self._releaser = LegacyReleaser(model, **conf)

    def update(self):
        self._releaser.update()

    @property
    def total_particle_count(self):
        return self._releaser.total_particle_count

    @property
    def particle_variables(self):
        return self._releaser.particle_variables


def sorted_interval(v, a, b):
    """
    Searches for an interval in a sorted array

    Returns the start (inclusive) and stop (exclusive) indices of
    elements in *v* that are greater than or equal to *a* and
    less than *b*. In other words, returns *start* and *stop* such
    that v[start:stop] == v[(v >= a) & (v < b)]

    :param v: Sorted input array
    :param a: Lower bound of array values (inclusive)
    :param b: Upper bound of array values (exclusive)
    :returns: A tuple (start, stop) defining the output interval
    """
    start = np.searchsorted(v, a, side='left')
    stop = np.searchsorted(v, b, side='left')
    return start, stop
