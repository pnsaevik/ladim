# Eel larvae with lunar compass

The module represents eel larvae (glass eels) that drift randomly in the
horizontal direction except under favorable lunar conditions, where the eels
swim towards the south.

Favorable lunar conditions is defined as:

1.  Moon illumination should be less than 25 % (new moon) or larger than 
    75 % (full moon).

2.  The moon should be visible above the horizon.      

The experimental evidence for this behaviour is described by 
[Cresci et al. (2019)](https://doi.org/10.1098/rsos.190812).

The eels also wander randomly in the vertical direction, in a specified part
of the vertical column.

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
- Position for computing lunar phase and altitude (`ibm.lunar_latlon`)
- Speed of the glass eels when oriented (`ibm.speed`)
- Limits for vertical movement (`ibm.vertical_limits`)

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

## History

Created by Pål Næverlid Sævik (2019) from an early draft by Bjørn Ådlandsvik
and Anne Dagrun Sandvik.
