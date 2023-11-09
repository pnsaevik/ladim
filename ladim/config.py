"""
Functions for parsing configuration parameters.

The module contains functions for parsing input configuration
parameters, appending default values and converting between
different versions of config file formats.
"""


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


def _convert_1_to_2(config_dict):
    from .legacy.configuration import configure as legacy_configure
    config_dict = legacy_configure(config_dict)
    config_dict['version'] = 2
    return config_dict
