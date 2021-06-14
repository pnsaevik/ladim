import numpy as np
from netCDF4 import Dataset

# from ladim2.timer import Timer


class Output:
    def __init__(
        self,
        state,
        timer,
        # release,
        filename,
        frequency,
        total_num_particles=0,
        instance_variables=None,
        particle_variables=None,
        **args,
    ):


        if instance_variables is None:
            particle_variables = dict()
        if particle_variables is None:
            particle_variables = dict()

        self.instance_count = 0
        self.record_count = 0
        self.step2nctime = timer.step2nctime
        self.time_unit = "s"
        self.cf_units = timer.cf_units(self.time_unit)
        try:
            value, unit = frequency
            frequency = np.timedelta64(value, unit)
        except TypeError:  # use 's' as default unit
            frequency = np.timedelta64(frequency, "s")
        self.frequency = frequency // np.timedelta64(timer.dt, "s")
        self.num_records = timer.Nsteps // self.frequency
        self.instance_variables = instance_variables
        self.particle_variables = particle_variables
        self.state = state
        self.num_particles = total_num_particles
        self.ncid = self._create_netcdf(filename)

    def write(self, step):

        state = self.state
        count = len(state)  # Present number of particles
        start = self.instance_count
        end = start + count

        # time_ = timer.step2nctime(step)
        self.ncid.variables["time"][self.record_count] = self.step2nctime(
            step, self.time_unit
        )
        self.ncid.variables["particle_count"][self.record_count] = count

        for var in self.instance_variables:
            self.ncid.variables[var][start:end] = getattr(state, var)

        # Flush
        self.ncid.sync()

        # Update counters
        self.instance_count = end
        self.record_count += 1

    def save_particle_variables(self):
        for var in self.particle_variables:
            self.ncid.variables[var][:] = self.state[var]

    def close(self):
        self.ncid.close()

    def _create_netcdf(self, filename):

        ncid = Dataset(filename, mode="w", format="NETCDF3_64BIT")

        ncid.createDimension("time", self.num_records)
        ncid.createDimension("particle_instance", None)
        # Works only for initial particle release
        # Else releaser should give total number
        if self.particle_variables:
            ncid.createDimension("particle", self.num_particles)

        # Coordinate variables
        v = ncid.createVariable("time", "f8", ("time",))
        v.long_name = "time"
        v.standard_name = "time"
        # v.units = timer.cfunits()
        v.units = self.cf_units

        v = ncid.createVariable("particle_count", "i", ("time",))
        v.long_name = "number of particles in a given timestep"
        v.ragged_row_count = "particle count at nth timestep"


        # Define output instance variables
        for var, ncatts in self.instance_variables.items():
            v = ncid.createVariable(var, ncatts["ncformat"], ("particle_instance",))
            for att, value in ncatts.items():
                if att == "ncformat":
                    continue
                setattr(v, att, value)

        # TODO: Handle particle variables at end of simulation
        # Eller end of file.
        # Define particle variables
        for var, ncatts in self.particle_variables.items():
            v = ncid.createVariable(var, ncatts["ncformat"], ("particle",))
            for att, value in ncatts.items():
                if att == "ncformat":
                    continue
                setattr(v, att, value)

        # global attributes
        ncid.institution = "Institute of Marine Research"
        ncid.source = "Lagrangian Advection and Diffusion Model, python version"
        ncid.representation = "Ragged contiguous by time"

        return ncid
