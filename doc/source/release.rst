Particle release
================

Particle release is controlled by a release file. This is a text file with one row per
release location per time. The name of the release file is defined in the configuration
file.

The format of the lines can be specified by the first line in the file. Alternatively,
it can be given by a ``names`` field in the configuration. The type of the non-reserved
columns are inferred by pandas or can be explicitly set by providing a  ``datatypes``
dictionary in the configuration.


There are seven reserved column names.

mult
  Integer, number of particles. Optional, default = 1.

release_time
  Time of particle release with format ``yyyy-mm-ddThh:mm:ss`` or ``"yyyy-mm-dd
  hh:mm:ss"``.

X
  Float, grid X-coordinate of the release position

Y
  Float, grid Y-coordinate of the release position

lon
  Float, longitude of release position

lat
  Float, latitute of release position

Z
  Float, release depth in meters (positive downwards).

.. note::
  It is essential that the release time is interpreted as a *single* string. This can be
  done by a "T" between date and clock parts or by enclosing it with double ticks. Time
  components can be dropped from the end with obvious defaults.

.. note::
  Either (X, Y) or (lon, lat) must be given, If both are present, the grid position (X,
  Y) is used.

.. note::
  Using (lon, lat) requires a ``ll2xy`` method in the :class:`Grid` class.

The release file can provide additional variables.

Example:

Using the configuration:

.. code-block:: yaml

  particle_release:
      release_file: ladim2.rls
      # release_type, discrete or continuous
      release_type: continuous
      release_frequency: [1, h]   # Hourly release

The header and a typical line in the release file may look like this:

.. code-block:: none

  mult   release_time      X      Y     Z   farmid   super
  ...
     5  2015-03-28T12  379.0  539.0   5.0    10333  1243.2
  ...

This means that 5 particles are released at time ``2015-03-28 12:00:00``, at a location
with ``farmid``-label 10333, with grid coordinates (379, 539) at 5 meters depth.
Each particle is a superindividual with weight 1243.2 so the particle release corresponds
to a total of 6216 individuals. The release is repeated every hour until a later particle
release anywhere (or the end of the simulation).

.. warning::
  Strange things may happen if particle release is not aligned with the model time
  stepping. The user is presently responsible for synchronizing model and release times.

.. note::
  Entries with release_time before the start or after the stop time of LADiM are ignored.
  In particular, constant continuous release will not work if the release_time is before
  the model's start time. (still true?)

.. note::
  Entries in a release row after the configured fields are silently ignored. (at least
  should be, check).

.. note::
  Particles released outside the model grid are ignored. A warning is logged.

.. note::
  Particles released on land are retained in the simulation, but do not move. In
  particular check that particles released by longitude and latitude near the coast is in
  a sea cell. TODO: Provide a warning for particles initially on land.

.. seealso::
  Module :mod:`release`
    Documentation of the :mod:`release` module
