Individual Based Model plug-in
==============================

The behaviour of the particles is described by the Individual Base Model (IBM). This
behaviour defines the particular use case of LADiM and is therefore a component that is
very likely to be replaced or modified. This is done by the plug in system.

The IBM plug-in shall consist of a function a class named IBM. This provides a standard
interface for LADiM to create an instance of this IBM class. When called from the
``main``-module, the ``__init__`` method should accept as general  arguments a
TimeKeeper, State, Forcing, and Grid instance in addition to any specific arguments for
the IBM. For an application the arguments are given in the ``IBM`` section in the yaml
configuration file. The class itself may optionally inherit from the Abstract Base Class
``BaseIBM``. The class should have one method called ``update`` with the obvious task of
updating the IBM to the next model step.

[Dette er ikke korrekt]
Variables in the IBM module can be forcing variables from the ocean model, such as
temperature, salinity and turbulence. Or they can be computed internally. Variables can
be kept between updates as attributes of the IBM class itself or as state variables.
Forcing variables should be kept as state variables as they will automatically be
updated. Also, variables that should be written as part of the standard output has to be
kept as state variables.

Example IBM
-----------

.. code::

    # A simple IBM that kills the particles at fixed "age" in degree-days.
    # age is configured as a state variable, inital value = 0
    # temp is congigured as a state variable, also as ibm_forcing

    import numpy as np
    from ladim2.ibm import BaseIBM
    from ladim2.timekeeper import TimeKeeper
    from ladim2.state import State

    [Sjekk opp, bruke forcing.temp vs. state.temp]
    class IBM(BaseIBM):
        def __init__(
            self,
            lifetime: float,   # Particle life time, units= degree-days
            timer: TimeKeeper,
            state: State,
            **args:   # This IBM is not using grid and forcing
        ) -> None:

            self.lifetime = lifetime
            self.state = state
            self.dt = timer.dt / np.timedelta64(1, 'd') # Time step [days]

        def update(self) -> None:

            state = self.state

            # Update the particle age
            state["age"] += state.temp * self.dt

            # Particles older than prescribed lifetime are added to dead
            state["alive"] &= state.age < self.lifetime