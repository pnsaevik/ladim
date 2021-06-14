"""Verify that output split and warm start works correctly"""

from postladim import ParticleFile


# Note: t = 9, pid = 3 produces non-indentical X-value for restart

# Choose time step number (0 <= t <= 12) in unsplit output
t = 9

# Choose particle number (0 <= pid <= t)
pid = 3

# Specify value used in split.yaml and restart.yaml
numrec = 4

# restart file number (0 <= n <= 3)
# i.e. warm_start_file = f"split_{n:03d}.nc" in restart.yaml
n = 1

# Corresponding file number and record in split output
filenr, tsplit = divmod(t, numrec)

print("\n --- unsplit ---")
with ParticleFile("unsplit.nc") as pf:
    print("time = ", pf.time(t))
    print(f"X[{pid}] = ", float(pf.X[t][pid]))
    print(f"Y[{pid}] = ", float(pf.Y[t][pid]))

print("\n --- split ---")
with ParticleFile(f"split_{filenr:03d}.nc") as pf:
    print("time = ", pf.time(tsplit))
    print(f"X[{pid}] = ", float(pf.X[tsplit][pid]))
    print(f"Y[{pid}] = ", float(pf.Y[tsplit][pid]))

print("\n --- restart ---")
if t < n * numrec:
    print(f"Restart has not started at t = {t} < {n*numrec}")
else:
    with ParticleFile(f"restart_{filenr:03d}.nc") as pf:
        print("time = ", pf.time(tsplit))
        print(f"X[{pid}] = ", float(pf.X[tsplit][pid]))
        print(f"Y[{pid}] = ", float(pf.Y[tsplit][pid]))
