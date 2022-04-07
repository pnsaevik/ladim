import subprocess


class Test_ladim_script:
    def test_can_show_help_message(self):
        cmd = 'ladim --help'
        output = subprocess.run(cmd, shell=True, capture_output=True)
        assert output.stderr.decode('utf-8') == ""
        assert output.stdout.decode('utf-8').startswith("usage: ladim")
