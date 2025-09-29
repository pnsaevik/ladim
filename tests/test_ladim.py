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


class Test_ladim_script_multi_output:

    def _discover_file_identifiers(self, testpath, expected_file_prfix):
        expected_files = testpath.glob(f'{expected_file_prfix}_*.nc_txt')

        identifiers = []
        for x in expected_files:
            parts = x.stem.split("_")
            if len(parts) > 1:
                identifiers.append(parts[-1])

        return sorted(identifiers)

    @pytest.mark.parametrize("example_num", range(1, 2))
    def test_run_examples(self, example_num):
        curdir = Path.cwd()
        name = f'mf_ex{example_num}'
        testpath = Path(__file__).parent / 'sample_data' / name
        expected_file_prefix = 'output'
        actual_file_prefix = 'out'

        with open(testpath / 'ladim.yaml') as f:
            conf_str = f.read()

        # Get the list of file identifiers (e.g., '0000', '0001', or '0002')
        identifiers = self._discover_file_identifiers(testpath, expected_file_prefix)

        generated_outfiles = []
        try:
            os.chdir(testpath)
            ladim.main(io.StringIO(conf_str))

            all_comparisons_passed = True
            for x in identifiers:
                with open(testpath / f'{expected_file_prefix}_{x}.nc_txt', 'r', encoding='utf-8') as f:
                    expected = json.load(f)

                outfile = Path(f'{actual_file_prefix}_{x}.nc')
                generated_outfiles.append(outfile)

                dset = xr.load_dataset(str(outfile))
                dset_txt = json.dumps(obj=dset.to_dict(), default=str, indent=4)
                dset_dict = json.loads(dset_txt)
                for v in ['X', 'Y']:
                    d = dset_dict['data_vars'][v]['data']
                    dset_dict['data_vars'][v]['data'] = np.round(d, 3).tolist()

                if 'attrs' in dset_dict:
                    dset_dict['attrs'].pop('date', None)
                    dset_dict['attrs'].pop('history', None)

                if dset_dict != expected:
                    all_comparisons_passed = False
                    out_txt_name = f'mc_out_{x}.nc_txt'
                    with open(testpath / out_txt_name, 'w', encoding='utf-8', newline='\n') as fp:
                        json.dump(obj=dset_dict, fp=fp, default=str, indent=4)

        finally:
            os.chdir(testpath)
            for outfile in generated_outfiles:
                try:
                    if outfile.exists():
                        outfile.unlink()
                except IOError:
                    pass
            os.chdir(curdir)

        assert all_comparisons_passed
