def configure(module_conf):
    import yaml

    # Handle variations of input config type
    if isinstance(module_conf, dict):
        config_dict = module_conf
    else:
        config_dict = yaml.safe_load(module_conf)

    if 'version' in config_dict:
        return _versioned_configure(config_dict)
    else:
        return _legacy_configure(config_dict)


def _versioned_configure(config_dict):
    return config_dict


def _legacy_configure(config_dict):
    from .legacy import configure as legacy_configure
    return legacy_configure(config_dict)
