

def density(temp, salt):
    """
    Compute density of sea water according to Fofonoff, N.P. & Millard, R.C. (1983).
    "Algorithms for Computation of Fundamental Properties of Seawater".
    Unesco Techical Papers in Marine Science.

    :param temp: Temperature in degrees Celcius
    :param salt: Salinity in PSU
    :returns: Sea water density in kg/m^3
    """
    T68 = temp * 1.00024

    a0 = 999.842594
    a1 = 6.793952e-2
    a2 = -9.095290e-3
    a3 = 1.001685e-4
    a4 = -1.120083e-6
    a5 = 6.536332e-9

    b0 = 8.24493e-1
    b1 = -4.0899e-3
    b2 = 7.6438e-5
    b3 = -8.2467e-7
    b4 = 5.3875e-9

    c0 = -5.72466e-3
    c1 = 1.0227e-4
    c2 = -1.6546e-6

    d0 = 4.8314e-4

    dens_T = a0 + (a1 + (a2 + (a3 + (a4 + a5 * T68) * T68) * T68) * T68) * T68

    dens_ST = dens_T + (
                b0 + (b1 + (b2 + (b3 + b4 * T68) * T68) * T68) * T68) * salt + (
                          c0 + (c1 + c2 * T68) * T68) * salt * (
                          salt ** (1 / 2)) + d0 * salt ** 2

    return dens_ST


def viscosity(temp, salt):
    """Calculate dynamic viscosity of sea water

    :param temp: Temperature in degrees Celcius
    :param salt: Salinity in PSU
    :returns: Viscosity of seawater in kg m^-1 s^-1
    """
    return 0.001 * (1.7915 + temp * (-0.0538 + temp * 0.0007) + 0.0023 * salt)
