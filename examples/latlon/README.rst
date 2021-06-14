=========================
Geographical coordinates
=========================

This is example demonstrates the use of geographical coordinates with
``LADiM``. A line of particles is released along the 59° parallel from Scotland
to Norway.

To accept geographical coordinates in the release file, the configuration file
should use the reserved terms "lon" and "lat" instead of "X" and "Y" in the
release variables list (head line in release file or names field in the yaml-file)

For output, the ``ladim.yaml`` file shows how to write longitude and latitude.
Basically, add "lon" and "lat" to the list of output instance variables and
make sure they have the necessary arguments.

The plot script, ``plot0.py``, demonstrates how to plot the particle distributions
with the ``cartopy`` library for ``matplotlib``.

The plotting above is slow. This is mainly due to the time used in handling the
intermediate resolution coast line. This may be speeded up several times by
using a pre-made coast line. The script ``make_coast.py`` extract the coast line
and saves it to ``coast.wkb``. The n``plot.py`` is then several times faster.