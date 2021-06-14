from postladim import ParticleFile

pf = ParticleFile("out.nc")

n = 42
p = 1000

print(pf.X[n, p])
print(pf.age[n, p])


# Verdi uten numba =
#  391.55527
#  10.027895
