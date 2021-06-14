"""Output module for NetCDF contiguous ragged array"""


import re
from pathlib import Path
from datetime import date
import logging
from typing import Dict, Union, Optional, Any, Generator, Tuple

import numpy as np  # type: ignore
from netCDF4 import Dataset  # type: ignore

# from ladim2.timekeeper import TimeKeeper, normalize_period
from ladim2.timekeeper import TimeDelta, normalize_period
from ladim2.state import State  # For typing

# from ladim2.grid import BaseGrid
from ladim2.output import BaseOutput

Variable = Dict[str, Any]

DEBUG = False
logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)


class Output(BaseOutput):
    """LADiM output to NetCDF

    """

    def __init__(
        self,
        modules: Dict[str, Any],
        filename: Union[Path, str],
        output_period: TimeDelta,
        instance_variables: Dict[str, Variable],
        particle_variables: Optional[Dict[str, Variable]] = None,
        layout: str = "sparse",  # "sparse" or "dense"
        ncargs: Optional[Dict[str, Any]] = None,
        numrec: int = 0,  # Number of records per file, no multfile if zero
        skip_initial: Optional[bool] = False,
        global_attributes: Optional[Dict[str, Any]] = None,
    ) -> None:

        logger.info("Initializing output")
        timer = modules["time"]
        grid = modules["grid"]
        self.modules = modules
        self.filename = filename
        self.layout = layout
        self.timer = timer
        # self.num_particles = modules['release'].total_particle_count
        self.instance_variables = instance_variables
        if self.layout == "dense":  # No need to save pid in orthogonal layout
            self.instance_variables.pop("pid", None)
        self.particle_variables = particle_variables if particle_variables else dict()
        logger.info("  Filename: %s", filename)
        if self.layout == "dense":
            logger.info("  Layout: %s", "netcdf dense = orthogonal array")
        else:
            logger.info("  Layout: %s", "netcdf sparse = contiguous ragged array")
        logger.info("  Instance variables: %s", list(instance_variables))
        logger.info("  Particle variables: %s", list(self.particle_variables))

        self.skip_initial = skip_initial
        if skip_initial:
            logger.info("  Skipping initial output")
        # self.numrec = numrec if numrec else 0
        self.numrec = numrec
        self.ncargs = ncargs if ncargs else dict()
        self.ncargs["format"] = "NETCDF4"  # Only accepted format
        if "mode" not in self.ncargs:
            self.ncargs["mode"] = "w"  # Default = (ower)write

        if global_attributes:
            self.global_attributes = global_attributes
        else:
            self.global_attributes = dict()
        if self.layout == "dense":
            self.global_attributes[
                "type"
            ] = "LADiM output, dense = netcdf orthogonal array"
        else:
            self.global_attributes[
                "type"
            ] = "LADiM output, sparse = netcdf contiguous ragged array"
        self.global_attributes["history"] = f"Created by LADiM, {date.today()}"

        self.output_period = normalize_period(output_period)
        self.output_period_step = self.output_period // timer.dt
        if timer.time_reversal:
            self.output_period = -self.output_period
        logger.info("  Output period: %s", str(self.output_period))

        self.num_records = int(
            abs((timer.stop_time - timer.start_time) // self.output_period)
        )
        if not skip_initial:  # Add an initial record
            self.num_records += 1
        logger.info("  Number of records: %s", self.num_records)

        if self.numrec:
            self.multifile = True
            self.filenames = fname_gnrt(Path(filename))
            self.filename = next(self.filenames)
            logger.info("  Multifile output")
        else:
            self.multifile = False
            self.filename = Path(filename)
            self.numrec = 999999

        self.record_count = 0
        self.instance_count = 0

        self.nc = self.create_netcdf()
        self.local_instance_count = 0
        self.local_record_count = 0

        self.step2nctime = timer.step2nctime
        self.time_unit = "s"
        self.nctime = timer.step2nctime(0, "s")
        self.cf_units = timer.cf_units(self.time_unit)

        if "lon" in self.instance_variables or "lat" in self.instance_variables:
            self.lonlat = True
        else:
            self.lonlat = False
        if self.lonlat:
            try:
                self.xy2ll = grid.xy2ll
            except AttributeError:
                self.xy2ll = lambda x, y: (x, y)

    def update(self) -> None:
        step = self.modules["time"].step
        if step % self.output_period_step == 0:
            self.write(self.modules["state"])

    def create_netcdf(self) -> Dataset:
        """Create a LADiM output netCDF file, sparse (default) or dense layout

        Returns:
            An open NetCDF Dataset
        """

        logging.debug("Creating new output file: %s", self.filename)

        # Handle netcdf args
        ncargs = self.ncargs
        nc = Dataset(self.filename, **ncargs)

        # self.offset = self.record_count  # record_count at start of file

        # Number of records in the file (the final file may be smaller)
        self.local_num_records = min(self.numrec, self.num_records - self.record_count)

        # Dimensions
        # nc.createDimension("time", self.local_num_records)
        # nc.createDimension("particle", self.num_particles)ma
        nc.createDimension("time", None)
        nc.createDimension("particle", None)
        if self.layout == "dense":
            instance_dim: Tuple[str, ...] = ("time", "particle")
        else:
            nc.createDimension("particle_instance", None)  # Unlimited
            instance_dim = ("particle_instance",)

        # Variables
        v = nc.createVariable("time", "f8", ("time",))
        v.long_name = "time"
        v.standard_name = "time"
        v.units = f"seconds since {self.timer.reference_time}"
        if self.layout == "sparse":
            v = nc.createVariable("particle_count", "i", ("time",))
            v.long_name = "Number of particles"
            v.ragged_row_count = "particle count at nth timestep"

        if self.instance_variables is not None:
            for var, conf in self.instance_variables.items():
                v = nc.createVariable(var, conf["encoding"]["datatype"], instance_dim)
                for att, value in conf["attributes"].items():
                    setattr(v, att, value)

        if self.particle_variables is not None:
            for var, conf in self.particle_variables.items():
                # xarray requires nan as fillvalue to interpret time
                if conf["encoding"]["datatype"] in ["f4", "f8"]:
                    v = nc.createVariable(
                        var,
                        conf["encoding"]["datatype"],
                        ("particle",),
                        fill_value=np.nan,
                    )
                else:
                    v = nc.createVariable(
                        var, conf["encoding"]["datatype"], ("particle",),
                    )
                for att, value in conf["attributes"].items():
                    # Replace string "reference_time" with actual reference time
                    if "reference_time" in value:
                        value = value.replace(
                            "reference_time", str(self.timer.reference_time)
                        )
                    setattr(v, att, value)

        if self.global_attributes is not None:
            for att, value in self.global_attributes.items():
                setattr(nc, att, value)

        return nc

    def write(self, state: State) -> None:
        """Write output instance variables to a (multi-)file

        Arguments:
          state: Model state

        """

        logger.debug("Writing output")

        # May skip initial output
        if self.skip_initial:
            self.skip_initial = False
            return

        if self.layout == "sparse":
            state.compactify()

        self.nc.variables["time"][self.local_record_count] = self.timer.nctime()

        if self.layout == "dense":
            # Fill out state.alive, False for unborn particles
            # has_value = np.full(self.num_particles, False)
            has_value = np.full(len(state), False)
            has_value[: len(state)] = state.alive
            for var in self.instance_variables:
                # values = getattr(state, var)
                self.nc.variables[var][self.local_record_count, has_value] = getattr(
                    state, var
                )[state.alive]
        else:  # sparse
            count = len(state)  # Present number of particles
            start = self.local_instance_count
            end = start + count
            self.nc.variables["particle_count"][self.local_record_count] = count
            for var in self.instance_variables:
                self.nc.variables[var][start:end] = getattr(state, var)

        # Compute and save lon, lat if requested
        if self.lonlat:
            lon, lat = self.xy2ll(state.X, state.Y)
            if self.layout == "dense":
                self.nc.variables["lon"][self.local_record_count, :] = lon
                self.nc.variables["lat"][self.local_record_count, :] = lat
            else:
                self.nc.variables["lon"][start:end] = lon
                self.nc.variables["lat"][start:end] = lat

        # Flush to file
        self.nc.sync()

        # Prepare for next time
        if self.layout == "sparse":
            self.instance_count += count
            self.local_instance_count += count
        self.record_count += 1
        self.local_record_count += 1
        self.nctime += float(self.output_period / np.timedelta64(1, self.time_unit))

        # File finished?
        if self.local_record_count == self.local_num_records:
            self.write_particle_variables(state)
            self.nc.close()
            # New file?
            if self.record_count < self.num_records:
                self.filename = next(self.filenames)
                self.nc = self.create_netcdf()
                self.local_instance_count = 0
                self.local_record_count = 0

    def write_particle_variables(self, state: State) -> None:
        """Write all output particle variables

        Args:
            state: A Ladim State instance
        """
        npart = int(state.pid.max()) + 1  # Total number of particles so far
        for var in self.particle_variables:
            if state.dtypes[var] == np.dtype("datetime64[s]"):
                unit = self.time_unit
                delta = state[var].astype("M8[s]") - self.timer.reference_time
                self.nc.variables[var][:npart] = delta[:npart] / np.timedelta64(1, unit)
            else:
                self.nc.variables[var][:npart] = state[var][:npart]

    def close(self) -> None:
        if self.nc.isopen():
            self.nc.close()


def fname_gnrt(filename: Path) -> Generator[Path, None, None]:
    """Generate file names based on prototype

    Args:
        filename: File name root
    Yields:
        Sequence of numbered file names

    Examples:
    output/cake.nc -> output/cake_000.nc, output/cake_001.nc, ...
    cake_04.nc -> cake_04.nc, cake_05.nc, ....
    """

    stem = filename.stem  # filename without parent and extension
    pattern = r"_(\d+)$"  # _digits at end of string
    m = re.search(pattern, stem)

    if m:  # Start from a number (or trailing underscore)
        ddd = m.group(1)
        filenumber = int(ddd)
        number_width = len(ddd)
        xxxx = stem[: -number_width - 1]  # remove _ddd
    else:  # Start from zero
        filenumber = 0
        number_width = 3
        xxxx = stem
    filename_template = f"{xxxx}_{{:0{number_width}d}}{filename.suffix}"

    while True:
        yield filename.parent / filename_template.format(filenumber)
        filenumber += 1
