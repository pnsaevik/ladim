from ladim import config
from pathlib import Path
import yaml
import numpy as np
import datetime


class Test_dict_get:
    def test_returns_correct_when_existing(self):
        mydict = dict(a=1, b=dict(c=2, d=3))
        assert config.dict_get(mydict, 'a') == 1
        assert config.dict_get(mydict, 'b') == dict(c=2, d=3)
        assert config.dict_get(mydict, 'b.c') == 2
        assert config.dict_get(mydict, 'b.d') == 3

    def test_returns_default_when_nonexisting(self):
        mydict = dict(a=1, b=dict(c=2, d=3))
        assert config.dict_get(mydict, 'A', 23) == 23
        assert config.dict_get(mydict, 'b.C') is None

    def test_can_try_alternatives(self):
        mydict = dict(a=1, b=dict(c=2, d=3))
        assert config.dict_get(mydict, ['A', 'a'], 23) == 1
        assert config.dict_get(mydict, ['b.C', 'b.c']) == 2
        assert config.dict_get(mydict, ['b.C', 'b.D']) is None


class Test_convert_1_to_2:
    def test_matches_snapshot(self):
        fname_in = Path(__file__).parent / 'sample_data/ex1/ladim.yaml'
        with open(fname_in, encoding='utf-8') as fp:
            dict_in = yaml.safe_load(fp)

        dict_out = config.convert_1_to_2(dict_in)

        assert dict_out == {
            'forcing': {
                'dt': 60,
                'file': '../forcing*.nc',
                'first_file': '',
                'last_file': '',
                'ibm_forcing': [],
                'legacy_module': 'ladim.gridforce.ROMS.Forcing',
                'start_time': np.datetime64('2015-09-07T01:00:00'),
                'stop_time': np.datetime64('2015-09-07T01:05:00'),
                'subgrid': None,
            },
            'grid': {
                'file': '../forcing*.nc',
                'legacy_module': 'ladim.gridforce.ROMS.Grid',
                'start_time': np.datetime64('2015-09-07T01:00:00'),
                'subgrid': None,
            },
            'ibm': {},
            'output': {
                'file': 'out.nc',
                'frequency': [60, 's'],
                'variables': {
                    'X': {
                        'long_name': 'particle X-coordinate', 'ncformat': 'f4'},
                    'Y': {
                        'long_name': 'particle Y-coordinate', 'ncformat': 'f4'},
                    'Z': {
                        'long_name': 'particle depth',
                        'ncformat': 'f4',
                        'positive': 'down',
                        'standard_name': 'depth_below_surface',
                        'units': 'm'},
                    'pid': {
                        'long_name': 'particle identifier', 'ncformat': 'i4'},
                    'release_time': {
                        'kind': 'initial',
                        'long_name': 'particle release time',
                        'ncformat': 'i4',
                        'units': 'seconds since '
                        '1970-01-01'}}},
            'release': {
                'colnames': ['release_time', 'X', 'Y', 'Z', 'group_id'],
                'defaults': {},
                'file': 'particles.rls',
                'formats': {'time': 'release_time'},
                'frequency': [1, 'm']},
            'solver': {
                'seed': 0,
                'start': datetime.datetime(2015, 9, 7, 1, 0),
                'step': 60,
                'stop': datetime.datetime(2015, 9, 7, 1, 5)},
            'tracker': {
                'diffusion': 0.1,
                'method': 'RK4',
            },
            'version': 2
        }

    def test_accepts_empty_dict(self):
        config.convert_1_to_2(dict())