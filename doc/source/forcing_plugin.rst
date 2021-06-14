Forcing plug-in
===============

To take LADiM forcing from an ocean model require a plug-in directed towards the model
output. Two plug-ins are included, ``ROMS`` for working with the Regional Ocean Model
system and ``ROMS2`` for using adaptive subgrid. The plug-in system can also be used for
nesting model output with different resolution. Forcing plug-ins can also be analytical
for idealized cases, the examples ....

The forcing plug-in is defined by a class called Forcing. It should take arguments like a
TimeKeeper class, a Grid class and the file name (or file name pattern) for the forcing
files [IBM forcing eller extra forcing]. It can use more keywords arguments taken from
the configuration.

The Forcing class should have an ``update`` method.

The Forcing class should have a method, velocity, and an attribute variables [dårlig
navn] estimating the velocity and providing scalar fields at the particle positions.