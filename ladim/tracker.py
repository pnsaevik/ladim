from .model import Model, Module


class Tracker(Module):
    def __init__(self, model: Model):
        super().__init__(model)


class HorizontalTracker(Tracker):
    def __init__(self, model: Model, **conf):
        super().__init__(model)
        from .legacy.tracker import Tracker as LegacyTracker
        self._tracker = LegacyTracker(model, **conf)

    def update(self):
        self._tracker.update()
