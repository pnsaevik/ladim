from ladim import model


class Test_load_class:
    def test_can_load_fully_qualified_class(self):
        numpy_dtype_class = model.load_class('numpy.dtype')
        numpy_dtype_i4 = numpy_dtype_class('i4')
        assert numpy_dtype_i4.itemsize == 4
