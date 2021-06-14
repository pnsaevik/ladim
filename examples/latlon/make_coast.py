"""Save a regional part of GSHHS  to a wkb-file"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute if Marine Research
# 2021-01-15
# ----------------------------------

from itertools import chain
from shapely import geometry, wkb
import cartopy.io.shapereader as shapereader

# Choose between c, l, i, h, f resolutions
GSHHS_resolution = "i"

# Name of output coast file
fname = "coast.wkb"

# Define regional domain
lonmin, lonmax, latmin, latmax = -6, 12, 54, 63  # North Sea

# Global coastline from GSHHS as shapely collection generator
path = shapereader.gshhs(scale=GSHHS_resolution)
coast = shapereader.Reader(path).geometries()

# Restrict the coastline to the regional domain
bbox = geometry.box(lonmin, latmin, lonmax, latmax)
coast = (bbox.intersection(p) for p in coast if bbox.intersects(p))
# Filter out isolated points
coast = filter(
    lambda p: isinstance(p, geometry.MultiPolygon) or isinstance(p, geometry.Polygon),
    coast,
)
# The filtered intersection can consist of both polygons, multipolygons
# which may not be dumped correctly to file.
# First make a generator expression of multipolygons = lists of polygons
# and thereafter flatten it into one large MultiPolygon
coast = (p if isinstance(p, geometry.MultiPolygon) else [p] for p in coast)
coast = geometry.MultiPolygon(chain(*coast))

# Save to WKB file
with open(fname, mode="wb") as fp:
    wkb.dump(coast, fp, output_dimension=2)
