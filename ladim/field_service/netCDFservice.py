import os.path

import netCDF4
from netCDF4 import Dataset
from . import FieldService


class netCDFService(FieldService):
    def __init__(self, uri, depth_dimension_name="depth", time_dimension_name="time"):
        super().__init__(uri, depth_dimension_name, time_dimension_name)
        if not os.path.exists(uri):
            raise FileNotFoundError(f"{uri}")
        self.dataset = netCDF4.Dataset(uri)

    def close(self):
        self.dataset.close()

    def update_uri(self, new_uri):
        self.close()
        self.uri = new_uri
        self.dataset = Dataset(new_uri)

    def read_values(self, variable_name, Y, X, time=None, depth=None):
        if variable_name not in self.dataset.variables:
            raise NameError(f"Variable \'{variable_name}\' not found in dataset \'{self.uri}\'.")

        variable = self.dataset.variables[variable_name]
        if time is None and depth is None:
            return variable[Y, X]

        if depth is None:
            # Time,Y,X. Assume dimensions are (time,Y,X)
            return variable[time, Y, X]

        if time is not None:
            # Depth,Y,X. Assume dimensions are (Y,X.Depth)
            return variable[Y, X, depth]

        # Both time and depth have been set.
        # Assume dimensions are (time,Y,X,Depth)
        return [time, Y, X, depth]
