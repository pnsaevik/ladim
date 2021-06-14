"""Make an initial particle distribution by perturbing a record from a particle file"""

from numpy.random import default_rng
from postladim import ParticleFile

# --- Settings ---

pfile = "../line/out.nc"
record_nr = 200       # 25 days
sigma = 0.0          # standard deviation of perturbation (grid units)
rls_file = "backwards.rls"

# --- Read the particle file ---

try:
    with ParticleFile(pfile) as pf:
        release_time = pf.time(record_nr)
        X = pf.X[record_nr]
        Y = pf.Y[record_nr]
        Z = pf.Z[record_nr]
except FileNotFoundError:
    print(f"Particle file {pfile} not found.")
    print("The line example must be run before the present example.")
    raise SystemExit(1)

# --- Perturb the distribution ---

rng = default_rng()
npart = len(X)
X += sigma * rng.standard_normal(npart)
Y += sigma * rng.standard_normal(npart)

# --- Write the release file ---

with open(rls_file, mode="w") as fid:
    fid.write("release_time         X          Y         Z\n")
    for x, y, z in zip(X, Y, Z):
        fid.write(f"{release_time} {float(x):10.6f} {float(y):10.6f} {float(z)}\n")
