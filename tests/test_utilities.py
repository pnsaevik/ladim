from ladim import utilities
import pytest
import os


@pytest.fixture()
def tmp_path_with_chdir(tmp_path):
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_dir)


class Test_load_class:
    def test_can_load_fully_qualified_class(self):
        numpy_dtype_class = utilities.load_class('numpy.dtype')
        numpy_dtype_i4 = numpy_dtype_class('i4')
        assert numpy_dtype_i4.itemsize == 4

    def test_can_load_class_in_current_directory(self, tmp_path_with_chdir):
        # Set up a python file in the temp dir which contains a class
        module_path = tmp_path_with_chdir / 'my_module.py'
        module_path.write_text(
            'class MyClass:\n'
            '  def timestwo(self, a):\n'
            '    return a * 2\n'
        )

        # Run and test code
        MyClass = utilities.load_class('my_module.MyClass')
        obj = MyClass()
        assert obj.timestwo(4) == 8
