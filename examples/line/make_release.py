"""Make a particle release file for the line example"""

import numpy as np

# End points of line in grid coordinates
x0, x1 = 63.55, 123.45
y0, y1 = 90.0, 90

# Number of particles along the line
Npart = 1000

# Fixed particle depth
Z = 5

X = np.linspace(x0, x1, Npart)
Y = np.linspace(y0, y1, Npart)

with open("line.rls", mode="w") as f:
    f.write("release_time   X       Y         Z\n")
    for x, y in zip(X, Y):
        f.write("1989-05-24T12 {:7.3f} {:7.3f} {:6.1f}\n".format(x, y, Z))
