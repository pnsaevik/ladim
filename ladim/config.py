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


def dict_get(d, items, default=None):
    if isinstance(items, str):
        items = [items]

    for item in items:
        try:
            return dict_get_single(d, item)
        except KeyError:
            pass

    return default


def dict_get_single(d, item):
    tokens = str(item).split(sep='.')
    sub_dict = d
    for t in tokens:
        if t in sub_dict:
            sub_dict = sub_dict[t]
        else:
            raise KeyError

    return sub_dict


def convert_1_to_2(c):
    out = {}

    # If any of the top-level attribute values in `c` are None, they should be
    # converted to empty dicts
    top_level_nones = [k for k in c if c[k] is None]
    c = c.copy()
    for k in top_level_nones:
        c[k] = dict()

    # Read timedelta
    dt_sec = None
    if 'numerics' in c:
        if 'dt' in c['numerics']:
            dt_value, dt_unit = c['numerics']['dt']
            dt_sec = int(np.timedelta64(dt_value, dt_unit).astype('timedelta64[s]').astype(int))

    out['version'] = 2

    out['solver'] = {}
    out['solver']['start'] = dict_get(c, 'time_control.start_time')
    out['solver']['stop'] = dict_get(c, 'time_control.stop_time')
    out['solver']['step'] = dt_sec
    out['solver']['seed'] = dict_get(c, 'numerics.seed')

    out['grid'] = {}
    out['grid']['file'] = dict_get(c, [
        'gridforce.first_file',
        'files.grid_file', 'gridforce.grid_file',
        'files.input_file', 'gridforce.input_file'])
    out['grid']['legacy_module'] = dict_get(c, 'gridforce.module', '') + '.Grid'
    out['grid']['start_time'] = np.datetime64(dict_get(c, 'time_control.start_time', '1970'), 's')
    out['grid']['subgrid'] = dict_get(c, 'gridforce.subgrid', None)

    out['forcing'] = {k: v for k, v in c.get('gridforce', {}).items() if k not in ('input_file', 'module')}
    out['forcing']['file'] = dict_get(c, ['gridforce.input_file', 'files.input_file'])
    out['forcing']['first_file'] = dict_get(c, 'gridforce.first_file', "")
    out['forcing']['last_file'] = dict_get(c, 'gridforce.last_file', "")
    out['forcing']['legacy_module'] = dict_get(c, 'gridforce.module', '') + '.Forcing'
    out['forcing']['start_time'] = np.datetime64(dict_get(c, 'time_control.start_time', '1970'), 's')
    out['forcing']['stop_time'] = np.datetime64(dict_get(c, 'time_control.stop_time', '1970'), 's')
    out['forcing']['subgrid'] = dict_get(c, 'gridforce.subgrid', None)
    out['forcing']['dt'] = dt_sec
    out['forcing']['ibm_forcing'] = dict_get(c, 'gridforce.ibm_forcing', [])

    out['output'] = {}
    out['output']['file'] = dict_get(c, 'files.output_file')
    out['output']['frequency'] = dict_get(c, 'output_variables.outper')
    out['output']['variables'] = {}

    # Convert output variable format spec
    outvar_names = dict_get(c, 'output_variables.particle', []).copy()
    outvar_names += dict_get(c, 'output_variables.instance', [])
    for v in outvar_names:
        out['output']['variables'][v] = c['output_variables'][v].copy()
        if v == 'release_time' and 'units' in c['output_variables'][v]:
            out['output']['variables'][v]['units'] = 'seconds since 1970-01-01'
    for v in dict_get(c, 'output_variables.particle', []):
        out['output']['variables'][v]['kind'] = 'initial'

    out['tracker'] = {}
    out['tracker']['method'] = dict_get(c, 'numerics.advection')
    out['tracker']['diffusion'] = dict_get(c, 'numerics.diffusion')

    # Read release config
    out['release'] = {}
    out['release']['file'] = dict_get(c, 'files.particle_release_file')
    out['release']['colnames'] = dict_get(c, 'particle_release.variables', [])
    if dict_get(c, 'particle_release.release_type', '') == 'continuous':
        out['release']['frequency'] = dict_get(c, 'particle_release.release_frequency', [0, 's'])
    out['release']['formats'] = {
        c.get('particle_release', {})[v]: v
        for v in dict_get(c, 'particle_release.variables', [])
        if v in c.get('particle_release', {}).keys()
    }
    out['release']['defaults'] = {
        k: np.float64(0)
        for k in dict_get(c, 'state.ibm_variables', []) + dict_get(c, 'ibm.variables', [])
        if k not in out['release']['colnames']
    }

    out['ibm'] = {}
    if 'ibm' in c:
        out['ibm']['legacy_module'] = dict_get(c, ['ibm.ibm_module', 'ibm.module'])
        if out['ibm']['legacy_module'] == 'ladim.ibms.ibm_salmon_lice':
            out['ibm']['legacy_module'] = 'ladim_plugins.salmon_lice'
        out['ibm']['conf'] = {}
        out['ibm']['conf']['dt'] = dt_sec
        out['ibm']['conf']['output_instance'] = dict_get(c, 'output_variables.instance', [])
        out['ibm']['conf']['nc_attributes'] = {
            k: v
            for k, v in out['output']['variables'].items()
        }
        out['ibm']['conf']['ibm'] = {k: v for k, v in c['ibm'].items() if k != 'ibm_module'}

    return out
