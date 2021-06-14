"""Plot the particle distribution at given time step

This version uses a coast file precomputed by make_coast.py

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

import shapely.wkb as wkb
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from postladim import ParticleFile

# ---------------
# User settings
# ---------------

# Files
particle_file = "latlon.nc"
coast_file = "coast.wkb"  # Made by make_coast.py

# time step to plot
tstep = 90

# Geographical extent
lon0, lat0, lon1, lat1 = -6, 54, 12, 62

# --------------------
# Read particle_file
# --------------------

with ParticleFile(particle_file) as pf:
    lon = pf["lon"][tstep]
    lat = pf["lat"][tstep]

# --------------------------------------------
# Read the coast file into a cartopy feature
# -------------------------------------------
lonlat = ccrs.PlateCarree()
try:
    with open(coast_file, mode="rb") as fid:
        mpoly = wkb.load(fid)
except FileNotFoundError:
    print("No coast file, run python make_coast.py")
    raise SystemExit(3)
coast = cfeature.ShapelyFeature(mpoly, crs=lonlat, facecolor="Khaki")

# ---------------------
# Make background map
# ---------------------

proj = ccrs.NorthPolarStereo(central_longitude=0.5 * (lon0 + lon1))
ax = plt.axes(projection=proj)
eps = 0.06 * (lat1 - lat0)  # Extend slightly southwards
ax.set_extent([lon0, lon1, lat0 - eps, lat1], lonlat)

# Set up the wedge-shaped boundary
south = proj.transform_points(
    lonlat, np.linspace(lon0, lon1, 100), np.array(100 * [lat0])
)
north = proj.transform_points(
    lonlat, np.linspace(lon1, lon0, 100), np.array(100 * [lat1])
)
boundary = np.vstack((north[:, :2], south[:, :2]))
ax.set_boundary(Path(boundary), transform=proj)

# Plot the coast
ax.add_feature(coast, facecolor="Khaki", edgecolor='Black')

# Add graticule
ax.gridlines(xlocs=range(lon0, lon1 + 2, 2), ylocs=range(lat0, lat1 + 1))

# ---------------------
# Plot the particles
# ---------------------
ax.plot(lon, lat, ".", color="red", markersize=3, transform=lonlat)

plt.show()
