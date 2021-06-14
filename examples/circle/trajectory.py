# Move particle in a circle (clockwise)

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from ladim2.state import State
from ladim2 import analytical

a: float = 0.01  # Velocity scale factor
# dt = 0.3   # seconds    <1% error EF
dt: int = 7  # seconds    <1% error RK2
# dt = 30    # seconds    <1% error RK4
nsteps: int = 2000


def main():

    state = State()

    # Initialize
    X0, Y0 = 0, 100
    state.append(X=X0, Y=Y0, Z=5)
    trajectory = [X0], [Y0]

    # Time loop
    for n in range(1, nsteps):
        advance(state, dt)
        append_trajectory(trajectory, state)

    plot_trajectory(trajectory)
    plot2(trajectory)


def sample_velocity(x, y):
    # Clockwise circular motion
    return a * y, -a * x


get_velocity = analytical.get_velocity2


def advance(state, dt) -> None:
    velocity = get_velocity(state, sample_velocity, dt)
    state["X"] += dt * velocity[0]
    state["Y"] += dt * velocity[1]


def append_trajectory(trajectory, state):
    trajectory[0].append(state.X[0])
    trajectory[1].append(state.Y[0])


def plot_trajectory(trajectory):
    plt.plot(trajectory[0], trajectory[1])
    plt.plot(trajectory[0][0], trajectory[1][0], "ro")
    plt.axis("image")
    plt.show()


def plot2(trajectory):
    x = np.array(trajectory[0])
    y = np.array(trajectory[1])
    r = np.sqrt(x * x + y * y)
    print("Relative error [%]: ", 100 * (r[-1] / r[0] - 1))


if __name__ == "__main__":
    main()
