# Chemicals

The module represents passive particles that have a vertical migration rate due
to upwelling/downwelling and vertical turbulence. Also, the module contains
improved land collision treatment to prevent artificial tracer concentration
buildup near shores.


## Usage

Modify `ladim.yaml` and `particles.rls` to suit your needs.

Common changes applied to `ladim.yaml`:
- Start date of simulation (`time_control.start_time`)
- Stop date of simulation (`time_control.stop_time`)
- Forcing input file (`gridforce.input_file`)
- Horizontal diffusion parameter (`numerics.diffusion`)
- Vertical diffusion parameter (`ibm.vertical_mixing`)
- Time step length (`numerics.dt`)
- Output frequency (`output_variables.outper`)
- Land collision treatment (`ibm.land_collision`)

The file `particles.rls` is a tab-delimited text file containing particle
release time and location, as well as particle attributes at the release time.
The order of the columns is given by the entry `particle_release.variables`
within `ladim.yaml`.

Finally, copy `ladim.yaml` and `particles.rls` to a separate directory and
run `ladim` here.


## Land collision treatment

The standard behaviour of Ladim is that particles crossing the land boundary
are rewinded to the position they had at the beginning of the time step. This
IBM checks if particles have not moved since the previous time step, in which
case they are assumed to have collided with land. Collided particles are
repositioned randomly within the cell where the particles originated.

To use the default Ladim land collision treatment instead, set the config entry
`ibm.land_collisions` to `"freeze"`.


## Output

The simulation result is stored in a file specified by the `files.output_file`
entry in `ladim.yaml`. The output variables are specified by the
`output_variables` entries. 


## History

May 2025: Bugfix in gridforce module
September 2024: Added lifespan option
December 2021: Added coastal hyperdiffusion as a land collision avoidance method
September 2021: Added heterogeneous diffusion
November 2020: Applied memory optimizations
October 2020: Applied corrections to vertical velocity computation

Created by Pål Næverlid Sævik (2020).
