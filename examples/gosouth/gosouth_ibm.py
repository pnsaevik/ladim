"""IBM module for the gosouth example in LADiM version 2"""

# ---------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# ----------------------------------

# 2021-01-18 Modified for LADiM v.2

import numpy as np


class IBM:
    """Adding a constant horizontal velocity to the particle tracking"""

    def __init__(
        self,
        modules: dict,
        direction: float,  # clockwise degree from North
        speed: float,  # swimming speed [m/s]
    ):
        self.dt = modules['time'].dtsec
        self.state = modules['state']
        self.grid = modules['grid']

        # Compute swimming velocity in grid coordinates
        azimuth = direction * np.pi / 180
        angle = self.grid.angle  # type: ignore
        self.Xs = speed * np.sin(azimuth + angle)
        self.Ys = speed * np.cos(azimuth + angle)

    def update(self) -> None:

        state = self.state
        grid = self.grid

        # Compute new position
        I = np.round(state.X).astype("int")
        J = np.round(state.Y).astype("int")
        X1 = state.X + self.Xs[J, I] * self.dt / grid.dx[J, I]  # type: ignore
        Y1 = state.Y + self.Ys[J, I] * self.dt / grid.dy[J, I]  # type: ignore

        # Only move particles to sea positions inside the grid
        move = grid.ingrid(X1, Y1) & grid.atsea(X1, Y1)
        state.X[move] = X1[move]
        state.Y[move] = Y1[move]
