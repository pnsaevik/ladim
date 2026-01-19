import pytest
import numpy as np
import typing
from ladim import output
import netCDF4 as nc
import uuid
import xarray as xr


class Test_append_netcdf:
    def test_can_create_new_netcdf_file(self):
        with nc.Dataset(filename=uuid.uuid4().hex, mode='w', diskless=True) as fp:
            data = xr.Dataset(
                data_vars=dict(
                    x=xr.DataArray(
                        data=[1, 2, 3],
                        dims='particle_instance',
                        attrs=dict(long_name='x coordinate')
                        )
                ),
                attrs=dict(root_attr='myattr'),
            )
            output.append_netcdf(data, fp)
            assert fp.dimensions['particle_instance'].size == 3
            assert fp.variables['x'][:].tolist() == [1, 2, 3]
            assert fp.variables['x'].long_name == 'x coordinate'
            assert fp.root_attr == 'myattr'

    def test_can_append_data(self):
        with nc.Dataset(filename=uuid.uuid4().hex, mode='w', diskless=True) as fp:
            fp.createDimension('mydim', size=None)
            fp.createVariable('x', int, 'mydim')
            fp['x'][:] = [1, 2, 3]

            data = xr.Dataset(
                data_vars=dict(
                    x=xr.DataArray(data=[4, 5, 6], dims='mydim')
                )
            )

            output.append_netcdf(data, fp)
            assert fp.variables['x'][:].tolist() == [1, 2, 3, 4, 5, 6]
