#! /usr/bin/env python

"""Main program for running LADiM

Lagrangian Advection and Diffusion Model

"""

# ---------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# ---------------------------------

import logging

import ladim
from .configuration import configure
from .gridforce import Grid, Forcing
from .release import ParticleReleaser
from .state import State
from .output import OutPut


def main(config_stream, loglevel=logging.INFO):
    """Main function for LADiM"""

    # ==================
    # Initiate the model
    # ==================

    # Logging
    logging.getLogger().setLevel(loglevel)

    # --- Configuration ---
    config = configure(config_stream)

    # --- Initiate modules ---
    modules = dict()
    modules['grid'] = Grid(config)
    modules['forcing'] = Forcing(config, modules['grid'])
    modules['release'] = ParticleReleaser(config, modules['grid'])
    modules['state'] = State(config, modules['grid'])
    modules['output'] = OutPut(config, modules['release'])

    # ==============
    # Main time loop
    # ==============

    logging.info("Starting time loop")
    for step in range(config["numsteps"] + 1):

        # --- Particle release ---
        if step in modules['release'].steps:
            V = next(modules['release'])
            modules['state'].append(V, modules['forcing'])

        # --- Update forcing ---
        modules['forcing'].update(step)

        # --- Save to file ---
        if step % config["output_period"] == 0:
            modules['output'].write(modules['state'], modules['grid'])

        # --- Update the model state ---
        modules['state'].update(modules['grid'], modules['forcing'])

    # ========
    # Clean up
    # ========

    # TODO: should also close the releaser
    modules['forcing'].close()
    # out.close()


def run():
    import sys
    import argparse
    import logging
    import datetime
    from pathlib import Path

    # ===========
    # Logging
    # ===========

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(module)s - %(message)s')

    # ====================
    # Parse command line
    # ====================

    parser = argparse.ArgumentParser(
        description='LADiM — Lagrangian Advection and Diffusion Model')
    parser.add_argument(
        '-d', '--debug',
        help="Show more information",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO)
    parser.add_argument(
        '-s', '--silent',
        help='Show less information',
        action="store_const", dest="loglevel", const=logging.WARNING)
    parser.add_argument('config_file', nargs='?', default='ladim.yaml')

    args = parser.parse_args()

    logging.info(" ================================================")
    logging.info(" === Lagrangian Advection and Diffusion Model ===")
    logging.info(" ================================================\n")

    logging.info(f"LADiM path: {ladim.__file__.strip('__init.py__')}")
    logging.info(f"python version:  {sys.version.split()[0]}\n")

    logging.info(f"  Configuration file: {args.config_file}")
    logging.info(f"  loglevel = {logging.getLevelName(args.loglevel)}")

    # =============
    # Sanity check
    # =============

    if not Path(args.config_file).exists():
        logging.critical(f'Configuration file {args.config_file} not found')
        raise SystemExit(1)

    # ===================
    # Run the simulation
    # ===================

    # Start message
    now = datetime.datetime.now().replace(microsecond=0)
    logging.info(f'LADiM simulation starting, wall time={now}')

    fp = open(args.config_file, encoding='utf8')
    ladim.main(config_stream=fp, loglevel=args.loglevel)

    # Reset logging and print final message
    logging.getLogger().setLevel(logging.INFO)
    now = datetime.datetime.now().replace(microsecond=0)
    logging.info(f'LADiM simulation finished, wall time={now}')
