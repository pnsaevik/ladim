from ladim.ibms.larvae import ibm
import numpy as np


class Test_sinkvel_egg:
    def test_changes_with_egg_diam(self):
        s = ibm.sinkvel_egg(
            mu_w=0.001, dens_w=1000, dens_egg=1001,
            diam_egg=np.array([0.0005, 0.001, 0.002]),
        )

        assert s.tolist() == [0.00013625, 0.0005423076178904741, 0.001424807617890474]

    def test_changes_with_egg_dens_when_large_egg(self):
        s = ibm.sinkvel_egg(
            mu_w=0.001, dens_w=1000, dens_egg=np.array([999, 1000, 1001]),
            diam_egg=0.002,
        )

        assert s.tolist() == [-0.001424807617890474, 0, 0.001424807617890474]

    def test_changes_with_egg_dens_when_small_egg(self):
        s = ibm.sinkvel_egg(
            mu_w=0.001, dens_w=1000, dens_egg=np.array([999, 1000, 1001]),
            diam_egg=0.0005,
        )

        assert s.tolist() == [-0.00013625, 0, 0.00013625]


class Test_growth_cod_larvae:
    def test_changes_with_temp(self):
        g = ibm.growth_cod_larvae(temp=np.array([2, 4, 6]), weight=1, dt=86400)
        g_rounded = np.round(g, 4)
        assert g_rounded.tolist() == [0.0466, 0.0824, 0.1182]

    def test_changes_with_weight(self):
        g = ibm.growth_cod_larvae(temp=16, weight=[.1, 1, 10], dt=86400)
        g_rounded = np.round(g, 3)
        assert g_rounded.tolist() == [0.022, 0.297, 2.100]


class Test_weight_to_length:
    def test_changes_with_weight(self):
        L = ibm.weight_to_length(weight=np.array([0.1, 1, 10]))
        L_rounded = np.round(L, 3)
        assert L_rounded.tolist() == [5.109, 9.934, 18.295]


# def test_snapshot():
#     import ladim_plugins.tests.test_examples
#     import os
#     os.chdir(os.path.dirname(ladim_plugins.tests.test_examples.__file__))
#     ladim_plugins.tests.test_examples.test_output_matches_snapshot('larvae')
