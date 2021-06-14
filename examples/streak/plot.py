"""Plot a snapshot of the particle distribution"""

# --------------------------------
# Bjørn Ådlandsvik <bjorn@ho.no>
# Institute of Marine Research
# November 2020
# --------------------------------

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from postladim import ParticleFile

# ---------------
# User settings
# ---------------

# Files
particle_file = "out.nc"
grid_file = "../data/ocean_avg_0014.nc"

# Subgrid definition
i0, i1 = 100, 136
j0, j1 = 90, 121

# timestamp
t = 176

# ----------------

# ROMS grid, plot domain

with Dataset(grid_file) as nc:
    H = nc.variables["h"][j0:j1, i0:i1]
    M = nc.variables["mask_rho"][j0:j1, i0:i1]
    lon = nc.variables["lon_rho"][j0:j1, i0:i1]
    lat = nc.variables["lat_rho"][j0:j1, i0:i1]
M[M > 0] = np.nan   # Mask out sea cells

# particle_file
pf = ParticleFile(particle_file)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1)

# Center and boundary grid points
Xc = np.arange(i0, i1)
Yc = np.arange(j0, j1)
Xb = np.arange(i0 - 0.5, i1)
Yb = np.arange(j0 - 0.5, j1)

# --- Background map ---
# Bottom topography
cmap = plt.get_cmap("Blues")
h = ax.contourf(Xc, Yc, H, cmap=cmap, alpha=0.3)
# Land mask
cmap = plt.matplotlib.colors.ListedColormap(["darkkhaki"])
ax.pcolormesh(Xb, Yb, M, cmap=cmap)
# Lon/lat lines
ax.contour(
    Xc, Yc, lon, levels=[2, 4, 6], colors="black", linestyles=":", linewidths=0.5
)
ax.contour(
    Xc, Yc, lat, levels=[59, 60, 61], colors="black", linestyles=":", linewidths=0.5
)

# --- Particle distribution ---
X, Y = pf.position(time=t)
timestring = pf.time(t)
ax.plot(X, Y, ".", color="red", markeredgewidth=0, lw=0.5)
ax.set_title(timestring)

# Show the results
plt.axis("image")
plt.axis((i0, i1 - 1, j0, j1 - 1))
plt.show()
