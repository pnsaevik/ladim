class FieldService:
    def __init__(self, uri, depth_dimension_name="depth", time_dimension_name="time"):
        self.uri = uri
        self.depth_dimension_name = depth_dimension_name
        self.time_dimension_name = time_dimension_name

    def update_uri(self, new_uri):
        # Close any open resource pointing to the old uri.
        # Open appropriate resources to the new uri
        raise NotImplementedError

    def read_values(self, variable_name, Y, X, time=None, depth=None):
        raise NotImplementedError
