# Effluents from mining activity

The module represents passive particles that have a vertical migration rate due
to sinking, vertical turbulence and advection due to ocean upwelling/downwelling.
Constant vertical diffusion/turbulence is assumed.


## Algorithm

The IBM applies four different processes to the particles, in
sequence. The processes are:

1.  *Resuspension*. Settled particles where the bottom shear stress is greater
    than `ibm.taucrit` are resuspended, meaning that they are free to move.

2.  *Vertical diffusion*. Active particles are diffused according to
    `ibm.vertical_mixing`. Reflective boundary conditions are employed at the top.
    
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
- Output file for settled particles (`ibm.output_file`)

The file `particles.rls` is a tab-delimited text file containing particle
release time and location, as well as particle attributes at the release time.
The order of the columns is given by the entry `particle_release.variables`
within `ladim.yaml`.

Finally, copy `ladim.yaml` and `particles.rls` to a separate directory and
run `ladim` here.


## Output

The simulation result is stored in a file specified by the `files.output_file`
entry in `ladim.yaml`. The output variables are specified by the
`output_variables` entries. Additionally, the exact position of settled particles
is stored in the file specified by `ibm.output_file`, if present. 


## History

**Version 1.6.3** (June 2022):
Fixed bug which caused `ladim` to crash when particles were close to the edge.

Created by Pål Næverlid Sævik (2021).
