"""
Functions for parsing configuration parameters.

The module contains functions for parsing input configuration
parameters, appending default values and converting between
different versions of config file formats.
"""
import numpy as np


def configure(module_conf):
    import yaml

    # Handle variations of input config type
    if isinstance(module_conf, dict):
        config_dict = module_conf
    else:
        config_dict = yaml.safe_load(module_conf)

    if 'version' not in config_dict:
        if 'particle_release' in config_dict:
            config_dict['version'] = 1
        else:
            config_dict['version'] = 2

    return _versioned_configure(config_dict)


def _versioned_configure(config_dict):
    if config_dict['version'] == 1:
        config_dict = _convert_1_to_2(config_dict)

    return config_dict


def _convert_1_to_2(c):
    # Read timedelta
    dt_value, dt_unit = c['numerics']['dt']

    # Read output variables
    outvars = dict()
    outvar_names = c['output_variables'].get('particle', []) + c['output_variables'].get('instance', [])
    for v in outvar_names:
        outvars[v] = c['output_variables'][v].copy()
    for v in c['output_variables'].get('particle', []):
        outvars[v]['kind'] = 'initial'
    if 'release_time' in outvars and 'units' in outvars['release_time']:
        outvars['release_time']['units'] = 'seconds since 1970-01-01'

    # Read release variables
    relvars = dict(
        file=c['files']['particle_release_file'],
        frequency=c['particle_release'].get('release_frequency', [0, 's']),
        colnames=c['particle_release']['variables'],
        formats={
            c['particle_release'][v]: v
            for v in c['particle_release']['variables']
            if v in c['particle_release'].keys()
        },
    )
    if c['particle_release'].get('release_type', '') != 'continuous':
        del relvars['frequency']

    # Read ibm vars
    ibmvars = c.get('ibm', dict()).copy()
    if 'ibm_module' in ibmvars:
        ibmvars['module'] = ibmvars['ibm_module']
        del ibmvars['ibm_module']

    config_dict = dict(
        version=2,
        solver=dict(
            start=c['time_control']['start_time'],
            stop=c['time_control']['stop_time'],
            step=np.timedelta64(dt_value, dt_unit).astype('int64'),
            seed=c['numerics'].get('seed', None),
            order=['release', 'forcing', 'output', 'tracker', 'ibm', 'state'],
        ),
        grid=dict(
            file=c['gridforce']['input_file'],
            legacy_module=c['gridforce']['module'] + '.Grid',
            start_time=c['time_control']['start_time'],
        ),
        forcing=dict(
            file=c['gridforce']['input_file'],
            legacy_module=c['gridforce']['module'] + '.Forcing',
            start_time=c['time_control']['start_time'],
            stop_time=c['time_control']['stop_time'],
            dt=np.timedelta64(dt_value, dt_unit).astype('int64'),
        ),
        release=relvars,
        output=dict(
            file=c['files']['output_file'],
            frequency=c['output_variables']['outper'],
            variables=outvars,
        ),
        tracker=dict(
            method=c['numerics']['advection'],
            diffusion=c['numerics']['diffusion'],
        ),
        ibm=ibmvars,
    )
    return config_dict
