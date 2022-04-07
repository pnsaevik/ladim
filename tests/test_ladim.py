import subprocess
import ladim.main
from pathlib import Path
import os
import xarray as xr
import json
import io


class Test_ladim_script:
    def test_can_show_help_message(self):
        cmd = ['ladim','--help']
        output = subprocess.run(cmd, capture_output=True)
        assert output.stderr.decode('utf-8') == ""
        assert output.stdout.decode('utf-8').startswith("usage: ladim")

    def test_can_advect_particles_when_old_version(self):
        curdir = Path.cwd()
        outfile = Path('out.nc')
        testpath = Path(__file__).parent / 'sample_data'

        with open(testpath / 'ladim.yaml') as f:
            conf_str = f.read()

        with open(testpath / 'output.nc_txt', 'r', encoding='utf-8') as f:
            expected = json.load(f)

        dset_dict = None
        try:
            os.chdir(testpath)
            #subprocess.run('ladim')
            ladim.main(io.StringIO(conf_str))
            dset = xr.load_dataset(str(outfile))
            dset_dict = json.loads(json.dumps(
                obj=dset.to_dict(),
                default=str,
            ))

        finally:
            try:
                outfile.unlink()
            except IOError:
                pass
            os.chdir(curdir)

        del expected['attrs']['date']
        del dset_dict['attrs']['date']
        assert dset_dict == expected
