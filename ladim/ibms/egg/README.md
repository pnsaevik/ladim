# Boyant eggs

The module represents fish eggs that have a vertical migration rate due to
buoyancy and vertical turbulence. Advection due to upwelling/downwelling is
not included.

The theoretical foundation for the module is documented by 
[Myksvoll et al. (2011)](http://dx.doi.org/10.1080/19425120.2011.595258).

## Usage

Modify `ladim.yaml` and `particles.rls` to suit your needs.

Common changes applied to `ladim.yaml`:
- Start date of simulation (`time_control.start_time`)
- Stop date of simulation (`time_control.stop_time`)
- Forcing input file (`gridforce.input_file`)
- Vertical diffusion parameter (`ibm.vertical_mixing`)
- Horizontal diffusion parameter (`numerics.diffusion`)
- Egg diameter (`ibm.egg_diam`)
- Time step length (`numerics.dt`)
- Output frequency (`output_variables.outper`)

The file `particles.rls` is a tab-delimited text file containing particle
release time and location, as well as particle attributes at the release time.
The order of the columns is given by the entry `particle_release.variables`
within `ladim.yaml`.

Finally, copy `ladim.yaml` and `particles.rls` to a separate directory and
run `ladim` here.


## Output

The simulation result is stored in a file specified by the `files.output_file`
entry in `ladim.yaml`. The output variables are specified by the
`output_variables` entries. 

## SVIM example

An example of configuration file and release file used in a real scenario is
provided by `svim_ladim.yaml` and `svim_particles.rls`. The code used to make
the release file is provided by `svim_make_release.m`.  

## History

Created by Mari Myksvoll (2011).
