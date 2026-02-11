import ladim.schema


class Test_create_schemadoc:
    def test_returns_string(self):
        result = ladim.schema.rest_doc()
        assert isinstance(result, str)
