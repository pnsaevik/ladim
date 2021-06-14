class IBM:
    """Base class for Individual Based Model"""

    def __init__(self, modules: dict, **kwargs):
        self.modules = modules
        self.opts = kwargs

    def update(self) -> None:
        """Updates the IBM to the next time step"""

    def close(self) -> None:
        """Perform cleanup procedures after simulation is finished"""
