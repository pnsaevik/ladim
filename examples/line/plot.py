"""Simple plot of LADiM particle distribution"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import xarray as xr
from postladim import ParticleFile

#%%

# ---------------
# User settings
# ---------------

#%%
# Files
particle_file = "out.nc"
grid_file = "../data/ocean_avg_0014.nc"

# Time step
t = 48    # 6 days (3 hours between output)

#%%



#%%

# particle_file
with ParticleFile(particle_file) as pf:
    X, Y = pf.position(t)

#%%

plt.plot(X, Y, 'r.')

#%%

# Use the grid file to add a simple land mask

M = xr.load_dataset(grid_file)["mask_rho"]
#%%

M.plot()
plt.plot(X, Y, 'ro')
plt.axis("image")
#%%

plt.pcolormesh(M)

#%%


# Set up the plot area
fig = plt.figure(figsize=(9, 8))
ax = plt.axes(xlim=(i0 + 1, i1 - 1), ylim=(j0 + 1, j1 - 1), aspect="equal")

# Background bathymetry
cmap = plt.get_cmap("Blues")
ax.contourf(Xcell, Ycell, H, cmap=cmap, alpha=0.5)

# Lon/lat lines
ax.contour(
    Xcell, Ycell, lat, levels=range(55, 64), colors="black", linestyles=":", alpha=0.5
)
ax.contour(
    Xcell,
    Ycell,
    lon,
    levels=range(-4, 10, 2),
    colors="black",
    linestyles=":",
    alpha=0.5,
)

# Landmask
constmap = plt.matplotlib.colors.ListedColormap([0.2, 0.6, 0.4])
M = np.ma.masked_where(M > 0, M)
plt.pcolormesh(Xb, Yb, M, cmap=constmap)

# Plot initial particle distribution
X, Y = pf.position(0)
(particle_dist,) = ax.plot(X, Y, ".", color="red", markeredgewidth=0, lw=0.5)
timestamp = ax.text(
    0.02, 0.96, pf.time(0), fontsize=15, backgroundcolor="white", transform=ax.transAxes
)


# Update function
def animate(t):
    X, Y = pf.position(t)
    particle_dist.set_data(X, Y)
    timestamp.set_text(pf.time(t))
    return particle_dist, timestamp


# Make mouse click halt the animation
anim_running = True


def onClick(event):
    global anim_running
    if anim_running:
        anim.event_source.stop()
        anim_running = False
    else:
        anim.event_source.start()
        anim_running = True


fig.canvas.mpl_connect("button_press_event", onClick)

# Do the animation
anim = FuncAnimation(
    fig,
    animate,
    frames=num_times,
    interval=40,
    repeat=True,
    repeat_delay=500,
    blit=True,
)

# anim.save('line.gif',  writer='imagemagick')
plt.show()

# %%
