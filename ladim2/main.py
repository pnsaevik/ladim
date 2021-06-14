"""Main function for running LADiM as an application"""

import sys
import os
import platform
from pathlib import Path
import logging
import datetime
import argparse
from typing import Union


from ladim2 import __version__, __file__
from ladim2.configure import configure
from ladim2.model import Model

# from ladim2.warm_start import warm_start
from ladim2.timekeeper import duration2iso


def main(
    configuration_file: Union[Path, str],
    loglevel: int = logging.INFO,
    config_version: int = 2,
) -> None:
    """Main function for LADiM

    args:
        configuration_file:
            Path to yaml configuration

        loglevel:
            Log level

    """

    wall_clock_start = datetime.datetime.now()

    # ---------------------
    # Logging
    # ---------------------

    # set master log level
    logging.basicConfig(level=loglevel, format="%(levelname)s:%(module)s - %(message)s")
    logger = logging.getLogger("main")

    # ----------------
    # Start info
    # ----------------

    # Set log level at least to INFO for the main function
    logger.setLevel(min(logging.INFO, loglevel))

    logger.debug("Host machine: %s", platform.node())
    logger.debug("Platform: %s", platform.platform())
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", None)
    if conda_env:
        logger.info("conda environment: %s", conda_env)
    logger.info("python executable: %s", sys.executable)
    logger.info("python version: %s", sys.version.split()[0])
    logger.info("LADiM version: %s", __version__)
    logger.debug("LADiM path: %s\n", Path(__file__).parent)
    logger.info("Configuration file: %s", configuration_file)
    logger.info("Loglevel: %s", logging.getLevelName(loglevel))
    logger.info("Wall clock start time: %s\n", str(wall_clock_start)[:-7])

    # ----------------
    # Configuration
    # ----------------

    config = configure(configuration_file, config_version)

    # -------------------
    # Initialization
    # -------------------

    logger.info("Initiating")
    model = Model(config)

    # --------------------------
    # Initial particle release
    # --------------------------

    logger.debug("Initial particle release")

    # --------------
    # Time loop
    # --------------

    logger.info("Starting time loop")
    for step in range(model.timer.Nsteps + 1):
        model.update(step)

    # --------------
    # Finalisation
    # --------------

    logger.setLevel(logging.INFO)

    logger.info("Cleaning up")
    model.finish()

    wall_clock_stop = datetime.datetime.now()
    logger.info("Wall clock stop time: %s", str(wall_clock_stop)[:-7])
    delta = wall_clock_stop - wall_clock_start
    logger.info("Wall clock running time: %s", duration2iso(delta))


def script():
    """Function for running LADiM as a command line application"""

    parser = argparse.ArgumentParser(
        description="LADiM 2.0 — Lagrangian Advection and Diffusion Model"
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Show debug information",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        "-s",
        "--silent",
        help="Show less information",
        action="store_const",
        dest="loglevel",
        const=logging.WARNING,
        default=logging.INFO,
    )
    parser.add_argument(
        "-v", "--version", help="Configuration format version", type=int, default=2
    )
    parser.add_argument(
        "config_file", nargs="?", help="Configuration file", default="ladim2.yaml"
    )

    args = parser.parse_args()

    # Start up message
    print("")
    print(" ========================================================")
    print(" === LADiM – Lagrangian Advection and Diffusion Model ===")
    print(" ========================================================")
    print("")

    main(args.config_file, loglevel=args.loglevel, config_version=args.version)
