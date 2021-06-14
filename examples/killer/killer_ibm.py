# Minimal IBM to kill old particles

import numpy as np

class IBM:
    def __init__(self, modules, **kwargs) -> None:

        print("Initializing killer feature")
        self.state = modules["state"]
        self.dt = modules["time"].dt / np.timedelta64(1, "D")  # Unit = days
        self.lifetime = kwargs["lifetime"]


    def update(self) -> None:

        # Update the particle age
        self.state["age"] += self.dt

        # Add particles older than prescribed lifetime to dead
        self.state["alive"] &= self.state["age"] < self.lifetime
