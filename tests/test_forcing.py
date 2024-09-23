from ladim.gridforce import ROMS
import numpy as np


class Test_z2s:
    def test_returns_interpolated_s_level(self):
        zrho = np.array([-5, -4, -3, -2]).reshape((4, 1, 1))
        k, a = ROMS.z2s(zrho, np.zeros(5), np.zeros(5), np.array([6, 5, 3.5, 2, 0]))
        assert k.tolist() == [1, 1, 2, 3, 3]
        assert a.tolist() == [1.0, 1.0, 0.5, 0.0, 0.0]