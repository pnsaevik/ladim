# Pre-import netCDF4 to avoid stupid warning
# noinspection PyUnresolvedReferences
import netCDF4 as _

import subprocess
import ladim
from pathlib import Path
import os
import xarray as xr
import json
import io
import numpy as np
import pytest
import yaml


class Test_ladim_script:
    def test_can_show_help_message(self):
        cmd = ['ladim', '--help']
        output = subprocess.run(cmd, capture_output=True)
        assert output.stderr.decode('latin1') == ""
        assert output.stdout.decode('latin1').startswith("usage: ladim")

    @pytest.mark.parametrize("example_num", range(1, 5))
    def test_run_examples(self, example_num):
        curdir = Path.cwd()
        name = f"ex{example_num}"
        testpath = Path(__file__).parent / 'sample_data' / name
        outfiles = sorted(list(testpath.glob('output*.nc_txt')))
        infiles = sorted(list(testpath.glob('input*.nc_txt')))

        with open(testpath / 'ladim.yaml') as f:
            conf_str = f.read()

        expected = {}
        for output_fname in outfiles:
            postfix = output_fname.name[6:-7]
            ladim_outfile = f'out{postfix}.nc'
            with output_fname.open(mode='r', encoding='utf-8') as f:
                json_contents = json.load(f)
            expected[ladim_outfile] = json_contents

        result = {}
        try:
            for infile in infiles:
                unpack_nc_txt_file(infile)

            os.chdir(testpath)
            ladim.main(io.StringIO(conf_str))
            result = _load_ladim_outputs_as_json(expected.keys())

        finally:
            for ladim_outfile in expected.keys():
                Path(ladim_outfile).unlink(missing_ok=True)
            for infile in infiles:
                nc_file_name = str(infile)[:-4]
                Path(nc_file_name).unlink(missing_ok=True)

            os.chdir(curdir)

        if result != expected:
            _dump_ladim_outputs_as_json(result, testpath)

        assert result == expected


def _dump_ladim_outputs_as_json(result, root):
    for ladim_outfile, contents in result.items():
        dump_file = root / (Path(ladim_outfile).name + '_txt')
        with dump_file.open(mode='w', encoding='utf-8', newline='\n') as fp:
            json.dump(obj=contents, fp=fp, default=str, indent=4)


def _load_ladim_outputs_as_json(ladim_outfiles):
    out = {}
    for fname in ladim_outfiles:
        dset = xr.load_dataset(str(fname))
        dset_txt = json.dumps(obj=dset.to_dict(), default=str, indent=4)
        dset_dict = json.loads(dset_txt)
        for v in ['X', 'Y']:
            if v not in dset_dict['data_vars']:
                continue
            d = dset_dict['data_vars'][v]['data']
            dset_dict['data_vars'][v]['data'] = np.round(d, 3).tolist()

        del dset_dict['attrs']['date']
        del dset_dict['attrs']['history']

        out[Path(fname).name] = dset_dict

    return out


def unpack_nc_txt_file(file):
    if not str(file).endswith('.nc_txt'):
        raise ValueError('Expected file ending .nc_txt')

    with open(file, mode='r', encoding='utf-8') as fp:
        contents = yaml.safe_load(fp)
    
    dset = xr.Dataset.from_dict(contents)

    outfile = str(file)[:-4]
    dset.to_netcdf(outfile)