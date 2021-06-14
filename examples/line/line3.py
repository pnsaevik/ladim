"""Module containing the LADiM Model class definition"""

from ladim2.configure import configure
from ladim2.model import init_module


configuration_file = "ladim2.yaml"
config_version = 2
config = configure(configuration_file, config_version)

# The module names can be taken from the configuration file.
# But, the order matters
module_names = [
    "state",
    "time",
    "grid",
    "forcing",
    "release",
    "output",
    "tracker",
]

# Initiate
modules = dict()
for name in module_names:
    modules[name] = init_module(name, config[name], modules)

# Time stepping
Nsteps = modules["time"].Nsteps
for step in range(Nsteps + 1):

    if step > 0:
        modules["time"].update()
    modules["release"].update()
    modules["forcing"].update()
    modules["output"].update()
    if step < Nsteps:
        modules["tracker"].update()

# Clean up
for name in module_names:
    try:
        modules[name].close()
    except (AttributeError, KeyError):
        pass
