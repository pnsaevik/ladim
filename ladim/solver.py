import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ladim.model import Model


class Solver:
    def __init__(self, start, stop, step, skip_forwards_to_next_release_on_empty_state, seed=None):
        self.start = np.datetime64(start, 's').astype('int64')
        self.stop = np.datetime64(stop, 's').astype('int64')
        self.step = np.timedelta64(step, 's').astype('int64')
        self.time = None
        self.skip_forwards_to_next_release_on_empty_state = skip_forwards_to_next_release_on_empty_state

        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def create(start, stop, step, skip_forwards_to_next_release_on_empty_state=False, seed=None):
        return Solver(start, stop, step, skip_forwards_to_next_release_on_empty_state, seed)

    def run(self, model: "Model"):
        # Skip to first release time if possible
        first_release_time = model.release.first_release_time()
        seconds_to_skip = max(0, first_release_time - self.start)
        timesteps_to_skip = seconds_to_skip // self.step
        self.time = self.start + self.step * timesteps_to_skip

        while self.time <= self.stop:
            model.release.update(model)
            if model.state.size == 0 and self.skip_forwards_to_next_release_on_empty_state:
                self.time = model.release.get_next_release_time(self.time)
                if self.time is None:
                    # end of input -> exit while loop
                    break
                continue
            model.forcing.update(model)
            model.output.update(model)
            model.tracker.update(model)
            model.ibm.update(model)

            self.time += self.step

