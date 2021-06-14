""""

Configuration reader for LADiM version 2
with compability wrapper for LADiM version 1 configuration

"""

# -----------------------------------
# Bjørn Ådlandsvik <bjorn@hi.no>
# Institute of Marine Research
# December 2020
# -----------------------------------

import sys
from pathlib import Path
import logging
from typing import Union, Dict, Any

import numpy as np
from netCDF4 import Dataset, num2date  # type: ignore
import yaml


# from .timekeeper import normalize_period

DEBUG = False
logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)


def configure(config_file: Union[Path, str], version: int = 2) -> Dict[str, Any]:
    """Main configuration function of LADiM

    Args:
        config_file:
            Name of configuration file
        version:
            configuration format version,
                = 1 for LADiM version 1.x
                = 2 for LADiM version 2.x

    Returns:
        2-level configuration dictionary

    """

    logger.info("Configuration")
    logger.info("  Configuration file %s:", config_file)

    if not Path(config_file).exists():
        logger.critical("No configuration file %s:", config_file)
        raise SystemExit(3)

    try:
        with open(config_file) as fid:
            config: Dict[str, Any] = yaml.safe_load(fid)
    except yaml.parser.ParserError:
        logger.critical("Not a valid yaml file: %s", config_file)
        raise SystemExit(3)

    logger.info("  Configuration file version: %s", version)

    # Try old configure version
    if version == 1:
        if config.get("version", None) == 2:
            logger.warning("Configuration format mismatch")
            logger.warning("Trying version 1 as requested by command line")
            config = configure_v1(config)
    elif version != 2:
        logger.warning("Configuration version should be 1 or 2, trying version 2")

    # Some sections may be missing
    if "state" not in config:
        config["state"] = dict()
    if "grid" not in config:
        config["grid"] = dict()
    if "ibm" not in config:
        config["ibm"] = dict()
    if "warm_start" not in config:
        config["warm_start"] = dict()

    # Handle non-orthogonality

    # Use time step from time
    # config["tracker"]["dt"] = config["time"]["dt"]

    # If grid["filename"] is missing, use forcing["filename"]
    if "module" not in config["grid"]:
        config["grid"]["module"] = config["forcing"]["module"]
    if "filename" not in config["grid"]:
        filename = Path(config["forcing"]["filename"])
        # glob if necessary and use first file
        if ("*" in str(filename)) or ("?" in str(filename)):
            directory = filename.parent
            filename = sorted(directory.glob(filename.name))[0]
        config["grid"]["filename"] = filename

    # Warm start
    if "filename" in config["warm_start"]:
        warm_start_file = config["warm_start"]["filename"]
        # Warm start overrides start time
        try:
            nc = Dataset(warm_start_file)
        except (FileNotFoundError, OSError):
            logging.critical("Could not open warm start file:%s", warm_start_file)
            raise SystemExit(1)
        tvar = nc.variables["time"]
        # Use last record in restart file
        warm_start_time = np.datetime64(num2date(tvar[-1], tvar.units))
        warm_start_time = warm_start_time.astype("M8[s]")
        config["time"]["start"] = warm_start_time
        logging.info("    Warm start at %s", warm_start_time)

        if "variables" not in config["warm_start"]:
            config["warm_start"]["variables"] = []

        # warm start -> release
        config["release"]["warm_start_file"] = config["warm_start"]["filename"]

    # skip_initial is default with warm start
    if "filename" in config["warm_start"] and "skip_initial" not in config["output"]:
        config["output"]["skip_initial"] = True

    # Possible improvement: write a yaml-file
    if DEBUG:
        yaml.dump(config, stream=sys.stdout)

    return config


# --------------------------------------------------------------------


def configure_v1(config: Dict[str, Any]) -> Dict[str, Any]:
    """Tries to read version 1 configuration files

    This function may fail for valid configuration files for LADiM version 1.
    Ordinary use cases at IMR should work.

    """

    conf2: Dict[str, Any] = dict()  # output version 2 configuration

    # time
    conf2["time"] = dict(
        start=config["time_control"]["start_time"],
        stop=config["time_control"]["stop_time"],
        dt=config["numerics"]["dt"],
    )
    if "reference_time" in config["time_control"]:
        conf2["time"]["reference"] = config["time_control"]["reference_time"]

    # grid and forcing
    conf2["grid"] = dict()
    conf2["forcing"] = dict()
    if "ladim.gridforce.ROMS" in config["gridforce"]["module"]:
        conf2["grid"]["module"] = "ladim2.ROMS"
        if "gridfile" in config["gridforce"]:
            conf2["grid"]["filename"] = config["gridforce"]["gridfile"]
        elif "gridfile" in config["files"]:
            conf2["grid"]["filename"] = config["files"]["gridfile"]
        conf2["forcing"]["module"] = "ladim2.ROMS"
        if "input_file" in config["gridforce"]:
            conf2["forcing"]["filename"] = config["gridforce"]["input_file"]
        elif "input_file" in config["files"]:
            conf2["forcing"]["filename"] = config["files"]["input_file"]
    if "subgrid" in config["gridforce"]:
        conf2["grid"]["subgrid"] = config["gridforce"]["subgrid"]
    if "ibm_forcing" in config["gridforce"]:
        conf2["forcing"]["ibm_forcing"] = config["gridforce"]["ibm_forcing"]

    # state
    conf2["state"] = dict()
    instance_variables = dict()
    particle_variables = dict()
    if "ibm" in config and "variables" in config["ibm"]:
        for var in config["ibm"]["variables"]:
            instance_variables[var] = "float"
    for var in config["particle_release"]:
        if var in ["mult", "X", "Y", "Z"]:  # Ignore
            continue
        if (
            "particle_variables" in config["particle_release"]
            and var in config["particle_release"]["particle_variables"]
        ):
            particle_variables[var] = config["particle_release"].get(var, "float")
    conf2["state"]["instance_variables"] = instance_variables
    conf2["state"]["particle_variables"] = particle_variables
    conf2["state"]["default_values"] = dict()
    for var in conf2["state"]["instance_variables"]:
        conf2["state"]["default_values"][var] = 0

    # tracker
    conf2["tracker"] = dict(advection=config["numerics"]["advection"])
    if config["numerics"]["diffusion"]:
        conf2["tracker"]["diffusion"] = config["numerics"]["diffusion"]

    # release
    conf2["release"] = dict(
        release_file=config["files"]["particle_release_file"],
        names=config["particle_release"]["variables"],
    )
    if "release_type" in config["particle_release"]:
        if config["particle_release"]["release_type"] == "continuous":
            conf2["release"]["continuous"] = True
            conf2["release"]["release_frequency"] = config["particle_release"][
                "release_frequency"
            ]
    # ibm
    if "ibm" in config:
        conf2["ibm"] = dict()
        for var in config["ibm"]:
            if var == "ibm_module":
                conf2["ibm"]["module"] = config["ibm"][var]
                continue
            if var != "variables":
                conf2["ibm"][var] = config["ibm"][var]

    # output
    conf2["output"] = dict(
        filename=config["files"]["output_file"],
        output_period=config["output_variables"]["outper"],
        instance_variables=dict(),
        particle_variables=dict(),
        ncargs=dict(
            data_model=config["output_variables"].get("format", "NETCDF3_CLASSIC")
        ),
    )
    for var in config["output_variables"]["instance"]:
        conf2["output"]["instance_variables"][var] = dict()
        D = config["output_variables"][var].copy()
        conf2["output"]["instance_variables"][var]["encoding"] = dict(
            datatype=D.pop("ncformat")
        )
        conf2["output"]["instance_variables"][var]["attributes"] = D
    for var in config["output_variables"]["particle"]:
        conf2["output"]["particle_variables"][var] = dict()
        D = config["output_variables"][var].copy()
        conf2["output"]["particle_variables"][var]["encoding"] = dict(
            datatype=D.pop("ncformat")
        )
        conf2["output"]["particle_variables"][var]["attributes"] = D

    return conf2
