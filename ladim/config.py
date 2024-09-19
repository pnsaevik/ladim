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
        config_dict = convert_1_to_2(config_dict)

    return config_dict


def convert_1_to_2(c):
    # Read timedelta
    dt_value, dt_unit = c['numerics']['dt']
    dt_sec = np.timedelta64(dt_value, dt_unit).astype('timedelta64[s]').astype('int64')

    # Read output variables
    outvars = dict()
    outvar_names = c['output_variables'].get('particle', []) + c['output_variables'].get('instance', [])
    for v in outvar_names:
        outvars[v] = c['output_variables'][v].copy()
    for v in c['output_variables'].get('particle', []):
        outvars[v]['kind'] = 'initial'
    if 'release_time' in outvars and 'units' in outvars['release_time']:
        outvars['release_time']['units'] = 'seconds since 1970-01-01'

    # Read release config
    relconf = dict(
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
        del relconf['frequency']
    ibmvars = c.get('state', dict()).get('ibm_variables', [])
    ibmvars += c.get('ibm', dict()).get('variables', [])
    relconf['defaults'] = {
        k: np.float64(0)
        for k in ibmvars
        if k not in relconf['colnames']
    }

    # Read ibm config
    ibmconf_legacy = c.get('ibm', dict()).copy()
    if 'module' in ibmconf_legacy:
        ibmconf_legacy['ibm_module'] = ibmconf_legacy.pop('module')
    ibmconf = dict()
    if 'ibm_module' in ibmconf_legacy:
        ibmconf['module'] = 'ladim.ibms.LegacyIBM'
        ibmconf['legacy_module'] = ibmconf_legacy['ibm_module']
        ibmconf['conf'] = dict(
            dt=dt_sec,
            output_instance=c.get('output_variables', {}).get('instance', []),
            nc_attributes={k: v for k, v in outvars.items()}
        )
        ibmconf['conf']['ibm'] = {
            k: v
            for k, v in ibmconf_legacy.items()
            if k != 'ibm_module'
        }

    config_dict = dict(
        version=2,
        solver=dict(
            start=c['time_control']['start_time'],
            stop=c['time_control']['stop_time'],
            step=dt_sec,
            seed=c['numerics'].get('seed', None),
            order=['release', 'forcing', 'output', 'tracker', 'ibm', 'state'],
        ),
        grid=dict(
            file=c['gridforce']['input_file'],
            legacy_module=c['gridforce']['module'] + '.Grid',
            start_time=np.datetime64(c['time_control']['start_time'], 's'),
        ),
        forcing=dict(
            file=c['gridforce']['input_file'],
            legacy_module=c['gridforce']['module'] + '.Forcing',
            start_time=np.datetime64(c['time_control']['start_time'], 's'),
            stop_time=np.datetime64(c['time_control']['stop_time'], 's'),
            dt=dt_sec,
            ibm_forcing=c['gridforce'].get('ibm_forcing', []),
        ),
        release=relconf,
        output=dict(
            file=c['files']['output_file'],
            frequency=c['output_variables']['outper'],
            variables=outvars,
        ),
        tracker=dict(
            method=c['numerics']['advection'],
            diffusion=c['numerics']['diffusion'],
        ),
        ibm=ibmconf,
    )
    return config_dict
