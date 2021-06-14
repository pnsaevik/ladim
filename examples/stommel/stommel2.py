import numpy as np
from numpy import pi, exp, sin, cos
import matplotlib.pyplot as plt

from ladim2.state import State
from ladim2.tracker import Tracker


def main():

    # --- Simulation ---
    Nsteps = 1736

    # --- Setup ---
    g = Grid()
    f = Forcing(grid=g)
    tracker = Tracker(dt=86400, advection="EF")
    state = State()

    # Initialize
    X0, Y0 = initial_release(grid=g)
    state.append(X=X0, Y=Y0, Z=5)

    # Time loop
    for n in range(Nsteps):
        tracker.update(state, grid=g, force=f)

    # Plot results
    plot_particles(state, X0, Y0, forcing=f)


class Grid:
    def __init__(self):
        km = 1000.0  # Kilometer                             [m]
        D = 200.0  # Depth                                   [m]
        lambda_ = 10000 * km  # West-east extent of domain   [m]
        b = 6300 * km  # South-north extent of domain        [m]
        dt = 24 * 3600  # Day                                [s]

        # Selfify: v -> self.v
        for v in "D lambda_ b dt".split():
            setattr(self, v, locals()[v])

        self.xmin, self.xmax, self.ymin, self.ymax = 0, lambda_, 0, b

    @staticmethod
    def metric(X, Y):
        return np.ones_like(X), np.ones_like(X)

    def ingrid(self, X, Y):
        return (0 < X) & (X < self.lambda_) & (0 < Y) & (Y < self.b)

    def depth(self, X, Y):
        return self.D + np.zeros_like(X)

    @staticmethod
    def atsea(X, Y):
        return np.ones(X.shape, dtype="bool")


class Forcing:
    def __init__(self, grid):
        b = grid.b
        lambda_ = grid.lambda_
        D = grid.D

        r = 1.0e-6              # Bottom friction coefficient          [s-1]
        beta = 1.0e-11          # Coriolis derivative                  [m-1 s-1]
        alfa = beta / r                                              # [m-1]
        F = 0.1                 # Wind stress amplitude                [N m-2]
        rho = 1025.0            # Density                              [kg/m3]
        gamma = F * pi / (r * b)                                     # [kg m2 s-1]
        G = (1 / rho) * (1 / D) * gamma * (b / pi) ** 2              # [m2 s-1]

        A = -0.5 * alfa + np.sqrt(0.25 * alfa ** 2 + (pi / b) ** 2)  # [m-1]
        B = -0.5 * alfa - np.sqrt(0.25 * alfa ** 2 + (pi / b) ** 2)  # [m-1]
        p = (1.0 - exp(B * lambda_)) / (exp(A * lambda_) - exp(B * lambda_))
        q = 1 - p

        # Selfify: v -> self.v
        for v in "b A B G p q".split():
            setattr(self, v, locals()[v])

    def velocity(self, X, Y, Z):

        # Unselfify: self.v -> v
        b, A, B, G, p, q = [getattr(self, v) for v in "b A B G p q".split()]

        U = G * (pi / b) * cos(pi * Y / b) * (p * exp(A * X) + q * exp(B * X) - 1)
        V = -G * sin(pi * Y / b) * (p * A * exp(A * X) + q * B * exp(B * X))
        return U, V

    def psi(self, X, Y):
        """Stream function"""

        # Unselfify: self.v -> v
        b, A, B, G, p, q = [getattr(self, v) for v in "b A B G p q".split()]

        return G * sin(pi * Y / b) * (p * exp(A * X) + q * exp(B * X) - 1)


def initial_release(grid):
    """Initalize with particles in two concentric circles"""
    km = 1000
    x0 = grid.lambda_ / 3.0
    y0 = grid.b / 3.0
    r1 = 800 * km
    r2 = 1600 * km

    T = np.linspace(0, 2 * np.pi, 1000)
    X01 = x0 + r1 * cos(T)
    Y01 = y0 + r1 * sin(T)
    X02 = x0 + r2 * cos(T)
    Y02 = y0 + r2 * sin(T)

    X0 = np.concatenate((X01, X02))
    Y0 = np.concatenate((Y01, Y02))
    return X0, Y0


def plot_particles(state, X0, Y0, forcing):

    # Discetize and contour the streamfunction
    km = 1000
    imax, jmax = 101, 64
    dx = 100 * km

    # Plot stream function background
    I = np.arange(imax) * dx
    J = np.arange(jmax) * dx
    JJ, II = np.meshgrid(I, J)
    Psi = forcing.psi(JJ, II)
    plt.contour(I / km, J / km, Psi, colors="k", linestyles=":", linewidths=0.5)

    # Initial state
    plt.plot(X0 / km, Y0 / km, ".b")
    # Final state
    plt.plot(state.X / km, state.Y / km, ".r")

    plt.axis("image")
    plt.show()


# --------------------------------------
if __name__ == "__main__":
    main()
