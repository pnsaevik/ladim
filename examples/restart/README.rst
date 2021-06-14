================
Example: restart
================

The purpose of this example is to demonstrate the restart mechanism.
There are three yaml files:

``unsplit.yaml``
  Runs the streak example unchanged, result in ``unsplit.nc``

``split.yaml``
  Runs the streak example, splitting the output in half-day files
  each with 4 records, 3 hours apart. Files are named ``split_000.nc``, ...

``restart.yaml``
  Restarts the split example from ``split_001.nc``. File names become
  ``restart_002.nc``, ...

The script ``make_release.py`` should be run initially to make the release
file. The example ``ladim split.yaml`` should be run before ``ladim
restart.yaml``.

The script ``verify.py`` can be used to verify the results. There results are almost
identical. The differences is correspond to representation error for 32 bits floating
point.
