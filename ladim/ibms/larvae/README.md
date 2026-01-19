# Larvae

The module represents fish eggs and larvae that have a vertical migration rate due to
buoyancy, active swimming and vertical turbulence. Advection due to upwelling/downwelling is
not included.

The theoretical foundation for the module is documented by the following publications: 
- Egg drift: [Myksvoll et al. (2011)](http://dx.doi.org/10.1080/19425120.2011.595258)
- Larvae drift: [Vikebø et al. (2007)](https://doi.org/10.3354/meps06979)


## Usage

Modify `ladim.yaml` and `particles.rls` to suit your needs.

Common changes applied to `ladim.yaml`:
- Start date of simulation (`time_control.start_time`)
- Stop date of simulation (`time_control.stop_time`)
- Forcing input file (`gridforce.input_file`)
- Vertical diffusion parameter (`ibm.vertical_mixing`)
- Horizontal diffusion parameter (`numerics.diffusion`)
- Time step length (`numerics.dt`)
- Output frequency (`output_variables.outper`)
- Light extinction coefficient (`ibm.extinction_coefficient`)
- Species (`ibm.species`)

When species name is given, the module assumes default values for a set of
egg- and larvae-related parameters. The default values are specified in
`ibm.py` and can be overridden in `ladim.yaml`. See comments in `ladim.yaml`
for details.

The file `particles.rls` is a tab-delimited text file containing particle
release time and location, as well as particle attributes at the release time.
The order of the columns is given by the entry `particle_release.variables`
within `ladim.yaml`. To produce a release file, one may also use the `makrel`
tool provided with `ladim_plugins`. See comments in `release.yaml` for details.

Finally, copy `ladim.yaml` and `particles.rls` to a separate directory and
run `ladim` here.


## Output

The simulation result is stored in a file specified by the `files.output_file`
entry in `ladim.yaml`. The output variables are specified by the
`output_variables` entries. 


## History

Created by Mari Myksvoll (2011) and Frode Vikebø (2007)
Adapted to Python Ladim by Pål Næverlid Sævik (2021)
