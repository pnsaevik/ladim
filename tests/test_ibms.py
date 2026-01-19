import netCDF4 as _

import ladim
import ladim.ibms
import tempfile
import contextlib
import os
import pathlib
import yaml
import xarray as xr
import pytest
import numpy as np
import importlib.resources


module_names = [
    d.name for d in pathlib.Path(ladim.ibms.__file__).parent.glob('*/')
    if d.joinpath('ladim.yaml').is_file()
]


modules_with_makrel = [
    d.name for d in pathlib.Path(ladim.ibms.__file__).parent.glob('*/')
    if d.joinpath('release.yaml').is_file()
]


@pytest.mark.parametrize("module_name", module_names)
def test_output_matches_snapshot(module_name):
    out = run_ladim(module_name)

    # Create reference outfile if it does not exist
    outname = get_module_dir(module_name).joinpath('out.nc')
    if not outname.exists():
        out.to_netcdf(outname)
        with open(str(outname) + '.yaml', 'w', encoding='utf-8') as fp:
            yaml.safe_dump(out.to_dict(), fp)

    ref = xr.load_dataset(outname, decode_cf=False)
    check_equal(out, ref)


@pytest.mark.parametrize("module_name", modules_with_makrel)
def test_makrel_matches_snapshot(module_name):
    out = run_makrel(module_name)
    ref_fname = get_module_dir(module_name).joinpath('out.rls')
    # with open(ref_fname, 'w', encoding='utf-8') as f:
    #     f.writelines(out.replace('\r', ''))
    with open(ref_fname, encoding='utf-8') as f:
        ref = f.read()

    assert out.splitlines() == ref.splitlines()


def check_equal(new, ref):
    new_dict = {k: v.values.tolist() for k, v in new.variables.items()}
    ref_dict = {k: v.values.tolist() for k, v in ref.variables.items()}

    assert new.variables.keys() == ref.variables.keys()
    assert new.coords.keys() == ref.coords.keys()
    assert new.data_vars.keys() == ref.data_vars.keys()
    assert new.sizes.items() == ref.sizes.items()
    for k in new_dict.keys():
        assert np.all(np.isclose(new_dict[k], ref_dict[k]))


def run_makrel(module_name):
    package = 'ladim_plugins.' + module_name
    traversable = importlib.resources.files(package).joinpath('release.yaml')
    with traversable.open() as config_file:
        conf = yaml.safe_load(config_file)

    import io
    from ladim.makrel import make_release

    buf = io.StringIO()
    make_release(conf, buf)
    buf.seek(0)
    return buf.read()


def run_ladim(module_name):
    # Change into a temporary folder `test_dir`
    with chdir_temp() as test_dir:
        np.random.seed(0)  # To ensure consistent results
        ladim.main(get_config(module_name))

        # Read and return output data
        return xr.load_dataset(test_dir.joinpath('out.nc'), decode_cf=False)


def get_config(module_name):
    # Load yaml config string
    package = 'ladim_plugins.' + module_name
    traversable = importlib.resources.files(package).joinpath('ladim.yaml')
    with traversable.open() as config_file:
        config_string = config_file.read()

    # Append module dir to file names so that ladim can find the config files
    # from a different folder
    module_dir = get_module_dir(module_name)
    config_dict = yaml.safe_load(config_string)
    config_dict['files']['particle_release_file'] = os.path.join(
        module_dir, config_dict['files']['particle_release_file'])

    input_file = config_dict['gridforce']['input_file']
    if not '://' in input_file:  # URLs should be unchanged
        config_dict['gridforce']['input_file'] = os.path.join(module_dir, input_file)

    if 'ibm' in config_dict and 'fjord_index_file' in config_dict['ibm']:
        config_dict['ibm']['fjord_index_file'] = os.path.join(
            module_dir, config_dict['ibm']['fjord_index_file'])

    # Re-serialize as yaml
    import io
    buf = io.StringIO()
    yaml.safe_dump(config_dict, buf)
    buf.seek(0)
    return buf.read()


def get_module_dir(module_name):
    return pathlib.Path(ladim.ibms.__file__).parent / module_name


@contextlib.contextmanager
def chdir_temp():
    tempdir = None
    try:
        tempdir = pathlib.Path(tempfile.mkdtemp(prefix='ladim_plugins_test_dir_'))
        curdir = None
        try:
            curdir = os.getcwd()
        except FileNotFoundError:
            # If current working dir does not exist (may happen in some test environments)
            pass
        
        os.chdir(tempdir)
        yield tempdir
        
        if curdir and os.path.isdir(curdir):  # curdir may be None or non-existent
            os.chdir(curdir)
            
    finally:
        try:
            if tempdir is not None:
                for fname in tempdir.glob('*'):
                    fname.unlink()
                tempdir.rmdir()

        except PermissionError:
            pass
