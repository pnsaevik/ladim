"""
Grid and Forcing for LADiM for the Regional Ocean Model System (ROMS)

"""

# -----------------------------------
# Bjørn Ådlandsvik, <bjorn@imr.no>
# Institute of Marine Research
# Bergen, Norway
# 2017-03-01
# -----------------------------------

# import sys
import glob
import logging
import numpy as np
from netCDF4 import Dataset, num2date
import ladim.gridforce.ROMS

from ladim.sample import sample2D, bilin_inv
from ladim.gridforce.ROMS import sample3D, z2s

class Grid(ladim.gridforce.ROMS.Grid):
    def __init__(self, config):
        super().__init__(config)

    def nearest_sea(self, X, Y):
        I = X - self.i0
        J = Y - self.j0
        i_new, j_new = nearest_unmasked(np.logical_not(self.M), I, J)
        return i_new, j_new

    def is_close_to_land(self, X, Y):
        I = X - self.i0
        J = Y - self.j0
        return is_close_to_land(self.M, I, J)


# -----------------------------------------------
# The Forcing class from the old forcing module
# -----------------------------------------------


class Forcing(ladim.gridforce.ROMS.Forcing):
    """
    Class for ROMS forcing

    """

    def __init__(self, config, grid):
        self.W = None
        super().__init__(config, grid)

    def _remaining_initialization(self):
        super()._remaining_initialization()
        steps = self.steps
        V = [step for step in steps if step < 0]
        if V:  # Forcing available before start time
            prestep = max(V)
            stepdiff = self.stepdiff[steps.index(prestep)]
            self.W = self.compute_w(self.U, self.V)
            self.Wnew = self.compute_w(self.Unew, self.Vnew)
            self.dW = (self.Wnew - self.W) / stepdiff
            # Interpolate to time step = -1
            self.W = self.W - (prestep + 1)*self.dW

        elif steps[0] == 0:
            self.W = self.compute_w(self.U, self.V)
            self.Wnew = self.compute_w(self.Unew, self.Vnew)
            self.dW = (self.Wnew - self.W) / steps[1]
            # Synchronize with start time
            self.Wnew = self.W
            # Extrapolate to time step = -1
            self.W = self.W - self.dW

        else:
            # No forcing at start, should already be excluded
            raise SystemExit(3)

        self.initialization_finished = True

    def update(self, t):
        super().update(t)

        # Read from config?
        interpolate_velocity_in_time = True

        logging.debug("Updating forcing, time step = {}".format(t))
        if t in self.steps:  # No time interpolation
            self.W = self.Wnew
        else:
            if t - 1 in self.steps:  # Need new fields
                stepdiff = self.stepdiff[self.steps.index(t - 1)]
                nextstep = t - 1 + stepdiff
                self.Wnew = self.compute_w(self.Unew, self.Vnew)
                if interpolate_velocity_in_time:
                    self.dW = (self.Wnew - self.W) / stepdiff

            # "Ordinary" time step (including self.steps+1)
            if interpolate_velocity_in_time:
                self.W += self.dW


    def vertdiff(self, X, Y, Z, name):
        MAXIMUM_K = len(self._grid.Cs_w) - 2
        MINIMUM_K = 1
        MINIMUM_D = 0

        I = np.int32(np.round(X)) - self._grid.i0
        J = np.int32(np.round(Y)) - self._grid.j0
        K, A = z2s(self._grid.z_w, I, J, Z)
        K_nearest = np.round(K - A).astype(np.int32)
        K_nearest = np.minimum(MAXIMUM_K, K_nearest)
        K_nearest = np.maximum(MINIMUM_K, K_nearest)
        F = self[name]
        return np.maximum(MINIMUM_D, F[K_nearest, J, I])

    def horzdiff(self, X, Y, Z):
        # Compute horizontal diffusivity using Smagorinsky (1963)

        jmax, imax = self._grid.H.shape

        I = np.int32(np.round(X)) - self._grid.i0
        J = np.int32(np.round(Y)) - self._grid.j0
        I = np.maximum(0, np.minimum(imax - 2, I))
        J = np.maximum(0, np.minimum(jmax - 2, J))
        K, A = z2s(self._grid.z_r, I, J, Z)

        u1 = (1 - A) * self['U'][K, J, I] + A * self['U'][K - 1, J, I]
        u2 = (1 - A) * self['U'][K, J + 1, I] + A * self['U'][K - 1, J + 1, I]
        v1 = (1 - A) * self['V'][K, J, I] + A * self['V'][K - 1, J, I]
        v2 = (1 - A) * self['V'][K, J, I + 1] + A * self['V'][K - 1, J, I + 1]

        dudy = u2 - u1
        dvdx = v2 - v1

        AHs = 0.04 * self._grid.dx[J, I] * np.abs(dudy + dvdx)
        AHs[~self._grid.atsea(I + self._grid.i0, J + self._grid.j0)] = 0

        return AHs

    def compute_w(self, u_in, v_in):
        z_r = self._grid.z_r[np.newaxis, :, :, :]
        z_w = self._grid.z_w[np.newaxis, :, :, :]
        u = u_in[np.newaxis, :, :, 1:-1]
        v = v_in[np.newaxis, :, 1:-1, :]
        pm = 1 / self._grid.dx
        pn = 1 / self._grid.dy

        w = compute_w(pn, pm, u, v, z_w, z_r)
        return w[0]

    def wvel(self, X, Y, Z, tstep=0.0, method='bilinear'):
        i0 = self._grid.i0
        j0 = self._grid.j0
        K, A = z2s(self._grid.z_w, X-i0, Y-j0, Z)
        F = self['W']
        if tstep >= 0.001:
            F += tstep*self['dW']
        return sample3D(F, np.round(X-i0), np.round(Y-j0), K, A, method=method)


def compute_w(pn, pm, u, v, z_w, z_r):
    # horizontal flux
    Hz_r = z_w[:, 1:, :, :] - z_w[:, :-1, :, :]
    Hz_u = 0.5 * (Hz_r[:, :, :, :-1] + Hz_r[:, :, :, 1:])
    Hz_v = 0.5 * (Hz_r[:, :, :-1, :] + Hz_r[:, :, 1:, :])
    on_u = 2 / (pn[:, :-1] + pn[:, 1:])
    om_v = 2 / (pm[:-1, :] + pm[1:, :])
    Huon = Hz_u * u * on_u
    Hvom = Hz_v * v * om_v
    del Hz_r, Hz_u, Hz_v, on_u, om_v

    # vertical flux
    dW = (Huon[:, :, 1:-1, :-1] - Huon[:, :, 1:-1, 1:]
          + Hvom[:, :, :-1, 1:-1] - Hvom[:, :, 1:, 1:-1])
    del Huon, Hvom
    W_0 = 0 * dW[:, 0:1, :, :]
    W = np.concatenate((W_0, dW.cumsum(axis=1)), axis=1)
    del dW, W_0

    # remove contribution from moving ocean surface
    wrk = W[:, -1:, :, :] / (z_w[:, -1:, 1:-1, 1:-1] - z_w[:, 0:1, 1:-1, 1:-1])
    W -= wrk * (z_w[:, :, 1:-1, 1:-1] - z_w[:, 0:1, 1:-1, 1:-1])
    del wrk

    # scale the flux
    Wscl = W * (pm[1:-1, 1:-1] * pn[1:-1, 1:-1])
    del W

    # find contribution of horizontal movement to vertical flux
    wrk_u = u * (z_r[:, :, :, 1:] - z_r[:, :, :, :-1]) * (pm[:, :-1] + pm[:, 1:])
    vert_u = 0.25 * (wrk_u[:, :, :, :-1] + wrk_u[:, :, :, 1:])
    del wrk_u
    wrk_v = v * (z_r[:, :, 1:, :] - z_r[:, :, :-1, :]) * (pn[:-1, :] + pn[1:, :])
    vert_v = 0.25 * (wrk_v[:, :, :-1, :] + wrk_v[:, :, 1:, :])
    del wrk_v
    vert = vert_u[:, :, 1:-1, :] + vert_v[:, :, :, 1:-1]
    del vert_u, vert_v

    # --- Cubic interpolation to move vert from rho-points to w-points ---

    cff1 = 3 / 8
    cff2 = 3 / 4
    cff3 = 1 / 8
    cff4 = 9 / 16
    cff5 = 1 / 16

    # Bottom layers
    slope_bot = ((z_r[:, 0, 1:-1, 1:-1] - z_w[:, 0, 1:-1, 1:-1]) /
                 (z_r[:, 1, 1:-1, 1:-1] - z_r[:, 0, 1:-1, 1:-1]))
    vert_b0 = (cff1 * (vert[:, 0, :, :]
                       - slope_bot * (vert[:, 1, :, :] - vert[:, 0, :, :]))
               + cff2 * vert[:, 0, :, :] - cff3 * vert[:, 1, :, :])
    vert_b1 = (cff1 * vert[:, 0, :, :]
               + cff2 * vert[:, 1, :, :] - cff3 * vert[:, 2, :, :])
    del slope_bot

    # Middle layers

    vert_m = (cff4 * (vert[:, 1:-2, :, :] + vert[:, 2:-1, :, :])
              - cff5 * (vert[:, 0:-3, :, :] + vert[:, 3:, :, :]))

    # Top layers
    slope_top = ((z_w[:, -1, 1:-1, 1:-1] - z_r[:, -1, 1:-1, 1:-1]) /
                 (z_r[:, -1, 1:-1, 1:-1] - z_r[:, -2, 1:-1, 1:-1]))
    vert_t0 = ((cff1 * (vert[:, -1, :, :]
                        + slope_top * (
                                    vert[:, -1, :, :] - vert[:, -2, :, :]))
                + cff2 * vert[:, -1, :, :] - cff3 * vert[:, -2, :, :]))
    vert_t1 = (cff1 * vert[:, -1, :, :]
               + cff2 * vert[:, -2, :, :] - cff3 * vert[:, -3, :, :])
    del slope_top

    # Bundle together

    vert_w = np.concatenate((vert_b0[:, np.newaxis, :, :],
                             vert_b1[:, np.newaxis, :, :],
                             vert_m,
                             vert_t1[:, np.newaxis, :, :],
                             vert_t0[:, np.newaxis, :, :]), axis=1)
    del vert_b0, vert_b1, vert_m, vert_t1, vert_t0

    vert = Wscl + vert_w
    del Wscl, vert_w

    # --- End cubic interpolation ---

    # Add zeros as boundary values
    wvel_pad = np.pad(vert, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant')
    del vert

    return -wvel_pad[:]


def nearest_unmasked(mask, i, j):
    # All neighbours
    i_center = np.int32(np.round(i))
    j_center = np.int32(np.round(j))
    i_neigh_raw = i_center + np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])[:, np.newaxis]
    j_neigh_raw = j_center + np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])[:, np.newaxis]

    # Handle neighbours outside the domain
    i_neigh = np.clip(i_neigh_raw, 0, mask.shape[1] - 1)
    j_neigh = np.clip(j_neigh_raw, 0, mask.shape[0] - 1)

    # Compute distance to origin
    dist2 = (i_neigh - i)**2 + (j_neigh - j)**2
    dist2_mask = np.ma.masked_array(dist2, mask[j_neigh, i_neigh])

    # Find coordinates of closest unmasked cell
    idx = dist2_mask.argmin(axis=0)
    i_close = i_neigh[idx, np.arange(len(idx))]
    j_close = j_neigh[idx, np.arange(len(idx))]
    
    return i_close, j_close


def is_close_to_land(mask, i, j):
    i_center = np.int32(np.round(i))
    j_center = np.int32(np.round(j))
    is_land = ~np.array(mask, dtype=bool)

    i_stencil = np.array([-1, 0, 1, 1, 1, 0, -1, -1])[:, np.newaxis]
    j_stencil = np.array([-1, -1, -1, 0, 1, 1, 1, 0])[:, np.newaxis]
    i_around = np.minimum(mask.shape[1] - 1, np.maximum(0, i_center + i_stencil))
    j_around = np.minimum(mask.shape[0] - 1, np.maximum(0, j_center + j_stencil))
    is_land_around = is_land[j_around, i_around]
    return np.any(is_land_around, axis=0)
