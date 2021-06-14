"""Module containing the LADiM Model class definition"""

import numpy as np

import ladim2
import ladim2.state
import ladim2.timekeeper
import ladim2.ROMS
import ladim2.out_netcdf
import ladim2.tracker

# --------------
# Settings
# --------------

start_time = "1989-05-24 12"
stop_time = "1989-06-20"
reference_time = "1970-01-01"

data_file = "../data/ocean_avg_0014.nc"

num_particles = 1000

advection = "EF"
dt = 3600  # seconds

output_variables = dict(
    pid=dict(
        encoding=dict(datatype="i4", zlib=True),
        attributes=dict(long_name="particle identifier"),
    ),
    X=dict(
        encoding=dict(datatype="f4", zlib=True),
        attributes=dict(long_name="particle X-coordinate"),
    ),
    Y=dict(
        encoding=dict(datatype="f4", zlib=True),
        attributes=dict(long_name="particle Y-coordinate"),
    ),
    Z=dict(
        encoding=dict(datatype="f4", zlib=True),
        attributes=dict(
            long_name="particle depth",
            standard_name="depth below surface",
            units="m",
            positive="down",
        ),
    ),
)

# ------------
# Initiate
# ------------

state = ladim2.state.State()

grid = ladim2.ROMS.Grid(filename=data_file)

timer = ladim2.timekeeper.TimeKeeper(
    start=start_time, stop=stop_time, dt=dt, reference=reference_time
)

force = ladim2.ROMS.Forcing(
    filename=data_file, modules=dict(time=timer, grid=grid, state=state)
)

output = ladim2.out_netcdf.Output(
    filename="out.nc",
    output_period=10800,  # 3 hours
    instance_variables=output_variables,
    modules=dict(state=state, time=timer, grid=grid),
)

tracker = ladim2.tracker.Tracker(
    advection=advection, modules=dict(state=state, time=timer, grid=grid, forcing=force)
)

# --- Initiate particle distribution
x0, x1 = 63.55, 123.45
y0, y1 = 90.0, 90
X0 = np.linspace(x0, x1, num_particles)
Y0 = np.linspace(y0, y1, num_particles)
Z0 = 5  # Fixed particle depth
state.append(X=X0, Y=Y0, Z=Z0)

# ----------------
# Time stepping
# ----------------

for step in range(timer.Nsteps + 1):

    if step > 0:
        timer.update()

    # --- Update forcing ---
    force.update()

    # --- Output
    output.update()

    # --- Update state to next time step
    # Improve: no need to update after last write
    tracker.update()

# -----------
# Clean up
# -----------

force.close()
output.close()
