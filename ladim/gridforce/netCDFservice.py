import os.path

import netCDF4
from netCDF4 import Dataset


class netCDFService():
    def __init__(self, config):
        self.fields = {}

        extra_files = config["gridforce"].get("extra_files",[])
        for extra_file in extra_files:
            path = extra_file["path"]
            dataset = Dataset(path)
            for field in extra_file["fields"]:
                self.fields[field] = dataset

    def close(self):
        for dataset in self.fields.values():
            dataset.close()

    def read_values(self, variable_name, Y, X, time=None, depth=None):
        if variable_name not in self.fields:
            raise NameError(f"Variable \'{variable_name}\' not defined in config.")

        variable = self.fields[variable_name].variables[variable_name]
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

