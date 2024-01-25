# Pre-import netCDF4 to avoid stupid warning
# noinspection PyUnresolvedReferences
import netCDF4

import subprocess
import ladim.main
from pathlib import Path
import os
import xarray as xr
import json
import io
import numpy as np
import pytest


class Test_ladim_script:
    def test_can_show_help_message(self):
        cmd = ['ladim', '--help']
        output = subprocess.run(cmd, capture_output=True)
        assert output.stderr.decode('latin1') == ""
        assert output.stdout.decode('latin1').startswith("usage: ladim")

    @pytest.mark.parametrize("example_num", range(1, 3))
    def test_run_examples(self, example_num):
        curdir = Path.cwd()
        outfile = Path('out.nc')
        name = f"ex{example_num}"
        testpath = Path(__file__).parent / 'sample_data' / name

        with open(testpath / 'ladim.yaml') as f:
            conf_str = f.read()

        with open(testpath / 'output.nc_txt', 'r', encoding='utf-8') as f:
            expected = json.load(f)

        dset_dict = None
        try:
            os.chdir(testpath)
            ladim.main(io.StringIO(conf_str))
            dset = xr.load_dataset(str(outfile))
            dset_txt = json.dumps(obj=dset.to_dict(), default=str, indent=4)
            dset_dict = json.loads(dset_txt)
            for v in ['X', 'Y']:
                d = dset_dict['data_vars'][v]['data']
                dset_dict['data_vars'][v]['data'] = np.round(d, 3).tolist()

        finally:
            try:
                outfile.unlink()
            except IOError:
                pass
            os.chdir(curdir)

        del dset_dict['attrs']['date']
        del dset_dict['attrs']['history']

        if dset_dict != expected:
            with open(testpath / 'out.nc_txt', 'w', encoding='utf-8', newline='\n') as fp:
                json.dump(obj=dset_dict, fp=fp, default=str, indent=4)

        assert dset_dict == expected
