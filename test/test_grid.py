from ladim2.model import init_module

def test_local():
    """Test the use of a local plug-in grid module"""

    config = dict(module="grid0")
    g = init_module('grid', config, dict())
    assert g.metric(10, 20) == 1
    assert g.ingrid(10, 20) is True
