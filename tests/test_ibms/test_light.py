import numpy as np
from ladim.ibms import light


class Test_light:
    def test_changes_with_latitude(self):
        Eb = light.light(time='2000-01-01T12', lon=5, lat=[0, 60])
        assert Eb.round(1).tolist() == [1500.1, 1484.1]

    def test_changes_with_time(self):
        Eb_1 = light.light(time='2000-01-01T12', lon=5, lat=60).round(1)
        Eb_2 = light.light(time='2000-01-01T00', lon=5, lat=60)
        assert [Eb_1, Eb_2] == [1484.1, 1.15e-05]

    def test_changes_with_date(self):
        Eb_1 = light.light(time='2000-01-01T12', lon=5, lat=60).round(1)
        Eb_2 = light.light(time='2000-06-01T12', lon=5, lat=60).round(1)
        assert [Eb_1, Eb_2] == [1484.1, 1502.4]

    def test_changes_with_depth(self):
        Eb = light.light(time='2000-01-01T12', lon=5, lat=60, depth=np.array([0, 5, 10]))
        assert Eb.round(1).tolist() == [1484.1, 546.0, 200.8]
