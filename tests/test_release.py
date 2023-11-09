from ladim import release
import numpy as np
import pytest


class Test_sorted_interval:
    @pytest.mark.parametrize("ab", [
        (1.5, 4.5), (1, 4.5), (1.5, 5), (-1, 4.5), (1.5, 9), (-2, -1), (8, 9),
    ])
    def test_correct_interval_when_sorted_array(self, ab):
        v = [0, 1, 2, 3, 4, 5, 6]
        a, b = ab
        start, stop = release.sorted_interval(v, a, b)

        w = np.array(v)
        expected = w[(w >= a) & (w < b)].tolist()
        assert v[start:stop] == expected
