from ladim.gridforce import ROMS
import numpy as np
from ladim import forcing


class Test_z2s:
    def test_returns_interpolated_s_level(self):
        zrho = np.array([-5, -4, -3, -2]).reshape((4, 1, 1))
        k, a = ROMS.z2s(zrho, np.zeros(5), np.zeros(5), np.array([6, 5, 3.5, 2, 0]))
        assert k.tolist() == [1, 1, 2, 3, 3]
        assert a.tolist() == [1.0, 1.0, 0.5, 0.0, 0.0]


class Test_timestring_formatter:
    def test_can_format_simple_date(self):
        result = forcing.timestring_formatter(
            pattern="My time: {time:%Y-%m-%d %H:%M:%S}",
            time="2012-12-31T23:58:59",
        )
        assert result == "My time: 2012-12-31 23:58:59"

    def test_can_format_shifted_dates(self):
        result = forcing.timestring_formatter(
            pattern="My time: {time - 3600:%Y-%m-%d %H:%M:%S}",
            time="2012-12-31T23:58:59",
        )
        assert result == "My time: 2012-12-31 22:58:59"
