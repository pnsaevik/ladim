Grid plug-in
============

Similar to the forcing plug-in, the grid plug-in is taylored for the particular ocean
used for the forcing. It consist of a class named Grid. The class can live in the same
module as the Forcing.

To init the grid class, a file name is needed. Other arguments can be given by key word
arguments. For instance the ROMS Grid can take a subgrid argument to limit the I/O time.

It should have attributes xmin, xmax, ymin, ymax describing the limits of the coordinate
system.

metric
depth (eller sample_depth?)

ingrid, atsea