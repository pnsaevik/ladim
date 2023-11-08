from .model import Model, Module


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
