from .model import Model, Module


class Output(Module):
    def __init__(self, model: Model):
        super().__init__(model)


class NetCDFOutput(Output):
    def __init__(self, model: Model, **conf):
        super().__init__(model)
        from .legacy.output import OutPut as LegacyOutput
        self._output = LegacyOutput(model, **conf)

    def update(self):
        self._output.update()
