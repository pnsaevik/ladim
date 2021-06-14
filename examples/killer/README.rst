Killer example
==============

This example demonstrates two "killer" features

1. How to kill particles, by using the alive state variable.

2. How to have a local ibm-module in the working directory
   that can easily be modified without re-installing ladim.

Particles are released continuously from a fixed point and are
killed when they reach the age of two days.

Run the example
---------------

To make the release file, `killer.rls`::

  python make_release.py

To run the model::

  ladim2

To animate the results::

  python animate.py