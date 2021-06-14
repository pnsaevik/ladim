"""Animate the time evolution of the particle distributions"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation

import shapely.wkb as wkb
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from postladim import ParticleFile

# ---------------
# User settings
# ---------------

# Files
particle_file = "latlon.nc"
grid_file = "../data/ocean_avg_0014.nc"
coast_file = "coast.wkb"  # Made by make_coast.py

# Geographical extent
lon0, lat0, lon1, lat1 = -6, 56, 12, 63

# ------------------------------

pf = ParticleFile(particle_file)

# Coast file
lonlat = ccrs.PlateCarree()
try:
    with open(coast_file, mode="rb") as fid:
        mpoly = wkb.load(fid)
except FileNotFoundError:
    print("No coast file, run python make_coast.py")
    raise SystemExit(3)
coast = cfeature.ShapelyFeature(mpoly, crs=lonlat, facecolor="Khaki")

# Bathymetry
with Dataset(grid_file) as nc:
    H = nc.variables["h"][:, :]
    lon_grid = nc.variables["lon_rho"][:, :]
    lat_grid = nc.variables["lat_rho"][:, :]


# Make background map

fig = plt.figure(figsize=(9, 7))

# Map setup
proj = ccrs.NorthPolarStereo(central_longitude=0.5 * (lon0 + lon1))
ax = plt.axes(projection=proj)
eps = 0.4  # Extend slightly southwards
ax.set_extent([lon0, lon1, lat0 - eps, lat1], lonlat)

south = proj.transform_points(
    lonlat, np.linspace(lon0, lon1, 100), np.array(100 * [lat0])
)
north = proj.transform_points(
    lonlat, np.linspace(lon1, lon0, 100), np.array(100 * [lat1])
)
boundary = np.vstack((north[:, :2], south[:, :2]))
ax.set_boundary(Path(boundary), transform=proj)

# Bathymetry
levels = [25, 50, 100, 250, 500, 1000, 2500]
plt.contourf(
    lon_grid,
    lat_grid,
    np.log(H),
    levels=np.log(levels),
    cmap="Blues",
    alpha=0.8,
    transform=lonlat,
)

# Coast
ax.add_feature(coast, facecolor="Khaki", edgecolor="Black")

# Graticule
ax.gridlines(xlocs=range(lon0, lon1 + 2, 2), ylocs=range(lat0, lat1 + 1))

# Plot initial particle distribution

lon = pf["lon"][0]
lat = pf["lat"][0]
(particle_dist,) = ax.plot(lon, lat, ".", color="red", markersize=3, transform=lonlat)
timestamp = ax.text(
    x=0.12,
    y=0.91,
    s=str(pf.time(0))[:-6],
    fontsize=13,
    backgroundcolor="white",
    transform=ax.transAxes,
)


# Update function
def animate(t):
    lon = pf["lon"][t]
    lat = pf["lat"][t]
    particle_dist.set_data(lon, lat)
    timestamp.set_text(str(pf.time(t))[:-6])
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
    frames=pf.num_times,
    interval=40,
    repeat=True,
    repeat_delay=500,
    blit=True,
)

# anim.save('line.gif',  writer='imagemagick')
plt.show()

pf.close()
