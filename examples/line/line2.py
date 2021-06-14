"""Main function for running LADiM as an application"""

from ladim2.configure import configure
from ladim2.model import Model

# ----------------
# Configuration
# ----------------

configuration_file = "ladim2.yaml"
config_version = 2
config = configure(configuration_file, config_version)

# -------------------
# Initialization
# -------------------

model = Model(config)

# --------------
# Time loop
# --------------

for step in range(model.timer.Nsteps + 1):
    model.update(step)

# --------------
# Finalisation
# --------------

model.finish()
