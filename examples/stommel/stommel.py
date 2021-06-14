import numpy as np
from numpy import pi, exp, sin, cos
import matplotlib.pyplot as plt

from ladim2.state import State
from ladim2 import analytical

# Global variables (parameters)

km = 1000.0                                                  # [m]
D = 200.0             # Depth                                  [m]
r = 1.0e-6            # Bottom friction coefficient            [s-1]
beta = 1.0e-11        # Coriolis gradient                      [m-1 s-1]
alfa = beta / r                                              # [m-1]
lambda_ = 10000 * km  # West-east extent of domain             [m]
b = 6300 * km         # South-north extent of domain           [m]
F = 0.1               # Wind stress amplitude                  [N m-2]
rho = 1025.0          # Density                                [kg/m3]
gamma = F * pi / (r * b)                                     # [kg m2 s-1]
G = (1 / rho) * (1 / D) * gamma * (b / pi) ** 2              # [m2 s-1]

A = -0.5 * alfa + np.sqrt(0.25 * alfa ** 2 + (pi / b) ** 2)  # [m-1]
B = -0.5 * alfa - np.sqrt(0.25 * alfa ** 2 + (pi / b) ** 2)  # [m-1]
p = (1.0 - exp(B * lambda_)) / (exp(A * lambda_) - exp(B * lambda_))
q = 1 - p

# --- Simulation ---
day = 86400
simulation_time = 1736 * day
dt = day
Nsteps = simulation_time // dt


def main():

    state = State()

    # Initialize
    X0, Y0 = initial_release()
    state.append(X=X0, Y=Y0, Z=5)

    # Time loop
    for n in range(Nsteps):
        velocity = get_velocity(state, sample_velocity, dt)
        state["X"] += dt * velocity[0]
        state["Y"] += dt * velocity[1]

    # Plot results
    plot_particles(state, X0, Y0)


def sample_velocity(X, Y):
    U = G * (pi / b) * cos(pi * Y / b) * (p * exp(A * X) + q * exp(B * X) - 1)
    V = -G * sin(pi * Y / b) * (p * A * exp(A * X) + q * B * exp(B * X))
    return U, V


get_velocity = analytical.get_velocity2


def psi(X, Y):
    """Stream function"""
    return G * sin(pi * Y / b) * (p * exp(A * X) + q * exp(B * X) - 1)


def initial_release():
    x0 = lambda_ / 3.0
    y0 = b / 3.0
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


def plot_particles(state, X0, Y0):

    # Discetize and contour the streamfunction
    imax, jmax = 101, 64
    dx = 100 * km
    I = np.arange(imax) * dx
    J = np.arange(jmax) * dx
    JJ, II = np.meshgrid(I, J)
    Psi = psi(JJ, II)
    plt.contour(I / km, J / km, Psi, colors="k", linestyles=":", linewidths=0.5)

    plt.plot(X0 / km, Y0 / km, ".b")
    plt.plot(state.X / km, state.Y / km, ".r")

    plt.axis("image")
    plt.show()


# --------------------------------------
if __name__ == "__main__":
    main()
