# Sediments

The module represents passive particles that have a vertical migration rate due
to sinking and vertical turbulence. Advection due to upwelling/downwelling is
not included. Constant vertical turbulence is assumed.


## Algorithm

The IBM processes applies four different processes to the particles, in
sequence. The processes are:

1.  *Resuspension*. Settled particles where the bottom shear stress is greater
    than `ibm.taucrit` are resuspended, meaning that they are free to move.

2.  *Vertical diffusion*. Active particles are diffused according to
    `ibm.vertical_mixing`. Reflective boundary conditions are employed at the
    bottom and top.
    
    At the moment, variable vertical diffusion is not implemented.
    *Therefore, the `ibm.vertical_mixing` parameter should reflect the
    typical vertical diffusion observed by a particle on the bottom under
    resuspension conditions.*
    
3.  *Sinking*. Active particles sink according to their sinking velocity, as
    specified in `particles.rls`. Particles that sink to the sea floor are
    marked as settled and inactive. They might be resuspended in the next
    timestep.
    
4.  *Ageing*. Particles whose age exceed `ibm.lifespan` are taken out of the
    simulation.


## Usage

Modify `ladim.yaml` and `particles.rls` to suit your needs.

Common changes applied to `ladim.yaml`:
- Start date of simulation (`time_control.start_time`)
- Stop date of simulation (`time_control.stop_time`)
- Forcing input file (`gridforce.input_file`)
- Horizontal diffusion parameter (`numerics.diffusion`)
- Time step length (`numerics.dt`)
- Output frequency (`output_variables.outper`)
- Vertical diffusion parameter (`ibm.vertical_mixing`)
- Particle life span (`ibm.lifespan`)
- Critical shear stress for resuspension (`ibm.taucrit`)

The file `particles.rls` is a tab-delimited text file containing particle
release time and location, as well as particle attributes at the release time.
The order of the columns is given by the entry `particle_release.variables`
within `ladim.yaml`.

Finally, copy `ladim.yaml` and `particles.rls` to a separate directory and
run `ladim` here.


## Creating particles

The module comes with a utility function for creating particle files.
The utility can create particles from an area, with sinking velocities taken
from Bannister et al. (2016,
[doi: 10.1093/icesjms/fsw027](http://dx.doi.org/10.1093/icesjms/fsw027)).

To create a release file, use either of the commands

`python -m ladim_plugins.release release.yaml out.rls`
`makrel release.yaml out.rls`

where `release.yaml` is the release config file, and `out.rls` is the output
particle release file. For details about the config file, see the attached
example `release.yaml` in the sedimentation module directory.


## Output

The simulation result is stored in a file specified by the `files.output_file`
entry in `ladim.yaml`. The output variables are specified by the
`output_variables` entries. 


## History

Update April 2024: Released particles gain sinking velocity automatically. Resuspension
and vertical diffusion can be skipped.

Update May 2021: Linear interpolation of bathymetry.

Created by Marcos Carvajalino Fernandez and Pål Næverlid Sævik (2020).
