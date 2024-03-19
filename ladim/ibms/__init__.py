from ..model import Model, Module


class IBM(Module):
    def __init__(self, model: Model):
        super().__init__(model)


class LegacyIBM(IBM):
    def __init__(self, model: Model, legacy_module, conf):
        super().__init__(model)

        from ..model import load_class
        LegacyIbmClass = load_class(legacy_module + '.IBM')
        self._ibm = LegacyIbmClass(conf)

    def update(self):
        grid = self.model['grid']
        state = self.model['state']
        forcing = self.model['forcing']
        self._ibm.update_ibm(grid, state, forcing)
