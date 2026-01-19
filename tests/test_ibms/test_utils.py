import numpy as np
from ladim.ibms import eos


class Test_density:
    def test_changes_with_temperature(self):
        rho = eos.density(temp=np.array([0, 50]), salt=30)
        assert rho.tolist() == [1024.0715523751858, 1009.9641764883575]

    def test_changes_with_salinity(self):
        rho = eos.density(temp=4, salt=np.array([10, 40]))
        assert rho.tolist() == [1007.9473603468945, 1031.7686242667996]


class Test_viscosity:
    def test_changes_with_temperature(self):
        mu = eos.viscosity(temp=np.array([0, 50]), salt=30)
        assert mu.tolist() == [0.0018605000000000002, 0.0009204999999999999]

    def test_changes_with_salinity(self):
        mu = eos.viscosity(temp=10, salt=np.array([0, 40]))
        assert mu.tolist() == [0.0013235000000000002, 0.0014155000000000003]
