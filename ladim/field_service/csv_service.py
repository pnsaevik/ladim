import os.path

from . import FieldService
import pandas as pd


class csvService(FieldService):
    def __init__(self, uri, depth_dimension_name="depth", time_dimension_name="time"):
        super().__init__(uri, depth_dimension_name, time_dimension_name)
        if not os.path.exists(uri):
            raise FileNotFoundError(f"{uri}")
        self.dataframe = pd.DataFrame(uri)

    def update_uri(self, new_uri):
        self.dataframe = pd.DataFrame(new_uri)
        # TODO: remove all views and copies of the previous df from the service object

    def read_values(self, variable_name, Y, X, time=None, depth=None):
        # Assume that the values in Y,X,time and depth are single values
        # TODO: Weaken the assumptions on the input values to handle _ranges_ in some or all input parameters.

        df = self.dataframe
        if variable_name not in df.columns:
            raise NameError(f"Variable name \'{variable_name}\' not found in data source \'{self.uri}\'.")

        if time is None and depth is None:
            view = df.loc[(df["Y"] == Y) & (df["X"] == X)]

            values = view[variable_name].tolist()
            return values

        if depth is None:
            view = df.loc[(df["Y"] == Y) & (df["X"] == X) & (df[self.time_dimension_name] == time)]

            values = view[variable_name].tolist()
            return values

        if time is None:
            view = df.loc[(df["Y"] == Y) & (df["X"] == X) & (df[self.depth_dimension_name] == depth)]

            values = view[variable_name].tolist()
            return values

        # Both time and depth are set
        view = df.loc[(df["Y"] == Y) &
                      (df["X"] == X) &
                      (df[self.time_dimension_name] == time) &
                      (df[self.depth_dimension_name] == depth)]

        values = view[variable_name].tolist()
        return values
