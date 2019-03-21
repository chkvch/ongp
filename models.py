import numpy as np
import const
import scipy.optimize
import scipy.integrate
import time
import sys

import ongp
from ongp import EOSError, AtmError, HydroError, UnphysicalParameterError

class FitYmeanError(Exception):
    pass

class model_container:

    def get_rho_z(self, logp, logt):

        # if using reos water, extend down to T < 1000 K using aneos water.
        # for this, mask to find the low-T part.

        logp_high_t = logp[logt >= 3.]
        logt_high_t = logt[logt >= 3.]

        logp_low_t = logp[logt < 3.]
        logt_low_t = logt[logt < 3.]

        try:
            rho_z_low_t = 10 ** self.z_eos_low_t.get_logrho(logp_low_t, logt_low_t)
        except:
            print ('off low-t eos tables?')
            raise

        try:
            rho_z_high_t = 10 ** self.z_eos.get_logrho(logp_high_t, logt_high_t)
        except:
            print ('off high-t eos tables?')
            raise

        rho_z = np.concatenate((rho_z_high_t, rho_z_low_t))
        return rho_z

    def get_rho_xyz(self, logp, logt, y, z):
        # only meant to be called when Z is non-zero and Y is not 0 or 1
        if np.any(np.isnan(logp)):
            raise EOSError('have %i nans in logp' % len(logp[np.isnan(logp)]))
        elif np.any(np.isnan(logt)):
            raise EOSError('have %i nans in logt' % len(logt[np.isnan(logt)]))
        elif np.any(y <= 0.):
            raise EOSError('one or more bad y')
        elif np.any(y >= 1.):
            raise EOSError('one or more bad y')
        elif np.any(z < 0.):
            raise EOSError('one or more bad z')
        elif np.any(z > 1.):
            raise EOSError('one or more bad z')
        rho_hhe = 10 ** self.hhe_eos.get_logrho(logp, logt, y)
        rho_z = self.get_rho_z(logp, logt)
        rhoinv = (1. - z) / rho_hhe + z / rho_z
        return rhoinv ** -1

    def save_gyre_output(self, gyre_outfile, stride=1):

        #needs:
        # self.nz
        # self.mtot
        # self.rtot
        # self.lint
        # self.m
        # self.r
        # self.p
        # self.t
        # self.rho
        # self.gradt
        # self.brunt_n2
        # self.gamma1
        # self.grada
        # self.dlogrho_dlogt_const_p

        with open(gyre_outfile, 'w') as f:
            header_format = '%6i ' + '%19.12e '*3 + '%5i\n'
            line_format = '%6i ' + '%26.16e '*18 + '\n'
            ncols = 19

            # gyre doesn't like the first zone to have zero enclosed mass, so we omit the center point when writing the gyre model.

            if stride > 1:
                nz = self.nz / stride
            else:
                nz = self.nz - 1

            f.write(header_format % (nz, self.mtot, self.rtot, 1., ncols))
            w = np.zeros_like(self.m)
            w[:-1] = self.m[:-1] / (self.mtot - self.m[:-1])
            w[-1] = 1e10
            # things we're not bothering to calculate in the interior (fill with zeros or ones):
                # luminosity (column 4)
                # opacity and its derivatives (columns 13, 14, 15)
                # energy generation rate and its derivatives (columns 16, 17, 18)
                # rotation (column 19)

            dummy_luminosity = 0.
            dummy_kappa = 1.
            dummy_kappa_t = 0.
            dummy_kappa_rho = 0.
            dummy_epsilon = 0.
            dummy_epsilon_t = 0.
            dummy_epsilon_rho = 0.
            omega = 0.

            nwrote = 0
            for k in np.arange(self.nz):
                if k == 0: continue
                if k % stride != 0: continue
                f.write(line_format % (k / stride, self.r[k], w[k], dummy_luminosity, self.p[k], self.t[k], self.rho[k], \
                                       self.gradt[k], self.brunt_n2[k], self.gamma1[k], self.grada[k], -1. * self.dlogrho_dlogt_const_p[k], \
                                       dummy_kappa, dummy_kappa_t, dummy_kappa_rho,
                                       dummy_epsilon, dummy_epsilon_t, dummy_epsilon_rho,
                                       omega))
                nwrote += 1

            print ('wrote %i zones to %s' % (k, gyre_outfile))


    def set_derivatives_etc(self):
        return
        from scipy.integrate import trapz
        self.gradt = np.zeros_like(self.p)
        self.brunt_b = np.zeros_like(self.p)
        self.chirho = np.zeros_like(self.p)
        self.chit = np.zeros_like(self.p)
        self.chiy = np.zeros_like(self.p)
        self.dlogy_dlogp = np.zeros_like(self.p)

        # self.r[0] = 1. # 1 cm central radius to keep these things at least calculable at center zone
        self.g = const.cgrav * self.m / self.r ** 2
        # self.g[0] = self.g[1] # hack so that we don't get infs in, e.g., pressure scale height. won't effect anything

        # this is a structure derivative, not a thermodynamic one. wherever the profile is a perfect adiabat, this is also gamma1.
        self.dlogp_dlogrho = np.diff(np.log(self.p)) / np.diff(np.log(self.rho))

        self.gamma1 = np.zeros_like(self.p)
        self.csound = np.zeros_like(self.p)
        self.gradt_direct = np.zeros_like(self.p)

        self.gamma1[:self.kcore] = self.z_eos.get_gamma1(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
        self.gamma1[self.kcore:] = self.hhe_eos.get_gamma1(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
        self.csound = np.sqrt(self.p / self.rho * self.gamma1)
        self.lamb_s12 = 2. * self.csound ** 2 / self.r ** 2 # lamb freq. squared for l=1

        self.delta_nu = (2. * trapz(self.csound ** -1, x=self.r)) ** -1 * 1e6 # the large frequency separation in uHz
        self.delta_nu_env = (2. * trapz(self.csound[self.kcore:] ** -1, x=self.r[self.kcore:])) ** -1 * 1e6

        dlnp_dlnr = np.diff(np.log(self.p)) / np.diff(np.log(self.r))
        dlnrho_dlnr = np.diff(np.log(self.rho)) / np.diff(np.log(self.r))

        self.chirho[:self.kcore] = self.z_eos.get_chirho(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
        self.chit[:self.kcore] = self.z_eos.get_chit(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
        self.grada[:self.kcore] = (1. - self.chirho[:self.kcore] / self.gamma1[:self.kcore]) / self.chit[:self.kcore] # e.g., Unno's equations 13.85, 13.86

        # ignoring the envelope z component when it comes to calculating chirho and chit
        self.chirho[self.kcore:] = self.hhe_eos.get_chirho(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
        self.chit[self.kcore:] = self.hhe_eos.get_chit(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
        self.gradt_direct[:self.kcore] = 0. # was previously self.gradt
        self.gradt_direct[self.kcore+1:] = np.diff(np.log(self.t[self.kcore:])) / np.diff(np.log(self.p[self.kcore:])) # was previously self.gradt

        self.brunt_n2_direct = np.zeros_like(self.p)
        self.brunt_n2_direct[1:] = self.g[1:] / self.r[1:] * (dlnp_dlnr / self.gamma1[1:] - dlnrho_dlnr)

        # other forms of the BV frequency
        self.homology_v = const.cgrav * self.m * self.rho / self.r / self.p
        self.brunt_n2_unno_direct = np.zeros_like(self.p)
        self.brunt_n2_unno_direct[self.kcore+1:] = self.g[self.kcore+1:] * self.homology_v[self.kcore+1:] / self.r[self.kcore+1:] * \
            (self.dlogp_dlogrho[self.kcore:] ** -1. - self.gamma1[self.kcore+1:] ** -1.) # Unno 13.102

        # terms needed for calculation of the composition term brunt_B.
        if self.kcore > 0:
            self.dlogy_dlogp[self.kcore+1:] = np.diff(np.log(self.y[self.kcore:])) / np.diff(np.log(self.p[self.kcore:])) # structure derivative # throws divide by zero encountered in log; fix me!
        else:
            self.dlogy_dlogp[1:] = np.diff(np.log(self.y)) / np.diff(np.log(self.p))
        # this is the thermodynamic derivative (const P and T), for H-He.
        self.dlogrho_dlogy = np.zeros_like(self.p)
        self.dlogrho_dlogy[self.kcore:] = self.hhe_eos.get_dlogrho_dlogy(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])

        rho_z = np.zeros_like(self.p)
        rho_hhe = np.zeros_like(self.p)
        rho_z[self.z > 0.] = self.get_rho_z(np.log10(self.p[self.z > 0.]), np.log10(self.t[self.z > 0.]))
        rho_hhe[self.y > 0.] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[self.y > 0.]), np.log10(self.t[self.y > 0.]), self.y[self.y > 0.])
        rho_hhe[self.y == 0.] = 10 ** self.hhe_eos.get_h_logrho((np.log10(self.p[self.y == 0]), np.log10(self.t[self.y == 0.])))
        self.dlogrho_dlogz = np.zeros_like(self.p)
        # dlogrho_dlogz is only calculable where all of X, Y, and Z are non-zero.
        self.dlogrho_dlogz[self.z * self.y > 0.] = -1. * self.rho[self.z * self.y > 0.] \
                                                    * self.z[self.z * self.y > 0.] \
                                                    * (rho_z[self.z * self.y > 0.] ** -1 - rho_hhe[self.z * self.y > 0.] ** -1)
        self.dlogz_dlogp = np.zeros_like(self.p)
        self.dlogz_dlogp[1:] = np.diff(np.log(self.z + 1e-20)) / np.diff(np.log(self.p)) # fudge doesn't change answer, just avoids inf if z==0


        # this is the form of brunt_n2 that makes use of grad, grada, and the composition term brunt B.
        self.brunt_n2_unno = np.zeros_like(self.p)
        # in core, explicitly ignore the Y gradient.
        self.brunt_n2_unno[:self.kcore+1] = self.g[:self.kcore+1] * self.homology_v[:self.kcore+1] / self.r[:self.kcore+1] * \
            (self.chit[:self.kcore+1] / self.chirho[:self.kcore+1] * (self.grada[:self.kcore+1] - self.gradt[:self.kcore+1]) + \
            self.dlogrho_dlogz[:self.kcore+1] * self.dlogz_dlogp[:self.kcore+1])
        # in envelope, Y gradient is crucial.
        self.brunt_n2_unno[self.kcore:] = self.g[self.kcore:] * self.homology_v[self.kcore:] / self.r[self.kcore:] * \
            (self.chit[self.kcore:] / self.chirho[self.kcore:] * (self.grada[self.kcore:] - self.gradt[self.kcore:]) + \
            self.dlogrho_dlogy[self.kcore:] * self.dlogy_dlogp[self.kcore:])

        # self.brunt_b = self.brunt_b_mhm
        # self.brunt_n2 = self.brunt_n2_mhm
        self.brunt_n2_thermal = self.g ** 2 * self.rho / self.p * self.chit / self.chirho * (self.grada - self.gradt)

        self.brunt_n2 = self.brunt_n2_unno
        self.brunt_n2[self.kcore:] = 0.

        # this is the thermo derivative rho_t in scvh parlance. necessary for gyre, which calls this minus delta.
        # dlogrho_dlogt_const_p = chit / chirho = -delta = -rho_t
        self.dlogrho_dlogt_const_p = np.zeros_like(self.p)
        self.dlogrho_dlogt_const_p[:self.kcore] = self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
        t_boundary = 1200 # previously 1000; got some nans from reos for slightly higher temperatures though
        if self.z_eos_option == 'reos water' and self.t[-1] < t_boundary: # must be calculated separately for low T and high T part of the envelope
            k_t_boundary = np.where(self.t > t_boundary)[0][-1]
            try:
                self.dlogrho_dlogt_const_p[self.kcore:k_t_boundary+1] = \
                        self.rho[self.kcore:k_t_boundary+1] \
                        * (self.z[self.kcore:k_t_boundary+1] / rho_z[self.kcore:k_t_boundary+1] \
                            * self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[self.kcore:k_t_boundary+1]), \
                                                                    np.log10(self.t[self.kcore:k_t_boundary+1])) \
                        + (1. - self.z[self.kcore:k_t_boundary+1]) / rho_hhe[self.kcore:k_t_boundary+1] \
                            * self.hhe_eos.get_rhot(np.log10(self.p[self.kcore:k_t_boundary+1]),
                                                    np.log10(self.t[self.kcore:k_t_boundary+1]),
                                                    self.y[self.kcore:k_t_boundary+1])) # eq. (9) in equations.pdf
            except:
                print ('failed in dlogrho_dlogt_const_p for hi-T part of envelope')
                raise
            try: # fix for z==0
                self.dlogrho_dlogt_const_p[k_t_boundary+1:] = self.rho[k_t_boundary+1:] \
                                                * (self.z[k_t_boundary+1:] / rho_z[k_t_boundary+1:] \
                                                * self.z_eos_low_t.get_dlogrho_dlogt_const_p(np.log10(self.p[k_t_boundary+1:]), \
                                                                                            np.log10(self.t[k_t_boundary+1:])) \
                                                + (1. - self.z[k_t_boundary+1:]) / rho_hhe[k_t_boundary+1:] \
                                                    * self.hhe_eos.get_rhot(np.log10(self.p[k_t_boundary+1:]), \
                                                                            np.log10(self.t[k_t_boundary+1:]), \
                                                                            self.y[k_t_boundary+1:]))
            except:
                print ('failed in dlogrho_dlogt_const_p for lo-T part of envelope')
                raise

        else: # no need to sweat low vs. high t (only an REOS-H2O limitation)
            if self.z_eos_option:
                self.dlogrho_dlogt_const_p[self.kcore:] = self.rho[self.kcore:] * (self.z[self.kcore:] / rho_z[self.kcore:] * self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:])) + (1. - self.z[self.kcore:]) / rho_hhe[self.kcore:] * self.hhe_eos.get_rhot(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:]))
            else:
                assert np.all(self.z[self.kcore:] == 0.), 'consistency check failed: z_eos_option is None, but have non-zero z in envelope'
                self.dlogrho_dlogt_const_p[self.kcore:] = self.hhe_eos.get_rhot(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])

        if np.any(np.isnan(self.dlogrho_dlogt_const_p)):
            print (self.t[np.isnan(self.dlogrho_dlogt_const_p)])
            raise ValueError('nans in dlogrho_dlogt_const_p')

        self.pressure_scale_height = self.p / self.rho / self.g
        self.mf = self.m / self.mtot
        self.rf = self.r / self.rtot


class model:

    def set_mtot_rtot_rhobar(self, method='naor'):
        if method == 'naor':
            dm = 4. / 3 * np.pi * self.rho[1:] * (self.r[1:] ** 3 - self.r[:-1] ** 3)
            dm = np.insert(dm, 0, 4. / 3 * np.pi * self.rho[0] * self.r[0] ** 3)
            self.dm = dm
            self.m = np.cumsum(dm)
            self.mtot = self.m[-1]

            self.rtot = self.r[-1]

            self.rhobar = self.mtot * 3. / 4 / np.pi / self.rtot ** 3
        else:
            raise NotImplementedError


class poly1_k(model):

    def __init__(self, k, rhoi=None, rtot=1., nz=None, p_surface=1e6):

        if rhoi is None:
            assert nz, 'must specify nz directly if not passing densities to poly1_k.'
            self.nz = nz
        else:
            self.nz = len(rhoi)

        self.k = k
        rho_surface = np.sqrt(p_surface / self.k)
        self.rho = np.ones(self.nz) * rho_surface
        self.p = self.k * self.rho ** 2

        self.r = np.linspace(1. / self.nz, 1., self.nz) * rtot


class arbitrary_density(model):

    def __init__(self, si, rhoi):
        assert len(si) == len(rhoi)
        self.rho = rhoi
        self.r = si
        self.nz = len(rhoi)

        # self.rtot = self.r[-1]

        # self.r_mid = np.mean([self.r[1:], self.r[:-1]], axis=0)
        # self.r_mid = np.insert(self.r_mid, 0, self.r[0] / 2.)

        # dr = np.diff(self.r)
        # self.dr = np.insert(dr, 0, self.r[0])
        # self.dm = self.dr * 4 * np.pi * self.r_mid ** 2 * self.rho
        # self.m = np.cumsum(self.dm)
        # self.mtot = self.m[-1]

        self.p = -1 * np.ones_like(self.rho)

        self.set_mtot_rtot_rhobar()

        # self.rhobar = 3. * self.mtot / 4. / np.pi / self.rtot ** 3


class poly1_mr:
    def __init__(self, nz=512, mtot=const.mjup, rtot=const.rjup):

        self.nz = nz

        self.rtot = rtot
        self.mtot = mtot

        r_n = rtot / np.pi # scale radius
        pc = np.pi / 4 * const.cgrav * mtot ** 2 / rtot ** 4
        rhoc = 1. / r_n * np.sqrt(pc / 4 / np.pi / const.cgrav)

        def rho_poly1(r):
            return rhoc / np.pi * np.sin(np.pi * r / self.rtot) / (r / self.rtot)
        def p_poly1(r):
            return pc / np.pi ** 2 * np.sin(np.pi * r / self.rtot) ** 2 / (r / self.rtot) ** 2

        # # sample low pressures by putting mesh points at r/rtot = 0.999 , 0.9999, 0.99999, ...
        # n_atm = int(np.floor(1. * self.nz / 5))
        n_atm = int(np.floor(1. * self.nz / 10))
        self.r = np.linspace(1e7, self.rtot * (1. - 1e-2), self.nz - n_atm)
        for n in np.arange(n_atm):
            self.r = np.append(self.r, self.rtot * (1. - 10 ** - (3. + n)))

        self.rho = rho_poly1(self.r)
        self.p = p_poly1(self.r)

        self.m = np.zeros_like(self.r)
        self.m[:-1] = np.cumsum(4. * np.pi * self.r[:-1] ** 2 * self.rho[:-1] * np.diff(self.r))
        self.m[-1] = self.mtot

        # calculate polytrope K
        self.k = pc / rhoc ** 2.


class linear_density:
    def __init__(self):
        self.nz = 1000
        rtot = 7e9
        self.r = np.linspace(1e7, rtot, self.nz)
        self.p = np.logspace(13, 6, self.nz)
        rhoc = 4.
        self.rho = rhoc * (1. - self.r / rtot)
        self.rho[-1] = self.rho[-2] / 100. # finite outer density

        self.mtot = np.pi * rhoc * rtot ** 3 / 3

class podolak_jupiter:
    # set up morris podolak's starting model. see MOMENTS.FOR for format
    def __init__(self, take_out_density_discontinuities=False):
        with open('../podolak/mom.dat', 'r') as f:
            line = f.readline()
        omega = float(line[:10])
        self.mtot = float(line[10:20])
        self.rtot = float(line[20:30])
        self.rhobar = float(line[30:40])
        self.nz = int(line[-5:])

        self.r, self.rho = np.genfromtxt('../podolak/mom.dat', unpack=True, skip_header=1)
        self.r[0] = 2.2204e-16

        self.rho *= self.rhobar

        if take_out_density_discontinuities:
            delete_count = 0
            while np.any(np.diff(self.r) == 0):
                for k in np.arange(self.nz):
                    if self.r[k+1] == self.r[k]:
                        k_delete = k + 1
                        delete_count += 1
                        break

                self.r = np.delete(self.r, k_delete)
                self.rho = np.delete(self.rho, k_delete)
            print ('took out %i density discontinuities' % delete_count)

        self.r *= self.rtot

        self.small = omega ** 2 * 3.578e6 / self.rhobar # note hardcoded thing here
        print (self.small)
        self.p = np.ones_like(self.rho) # morris' code is purely the shape code; it doesn't converge to a barotrope.

        dm = 4. * np.pi * self.r[:-1] ** 2 * self.rho[:-1] * np.diff(self.r)
        self.m = np.cumsum(dm)
        self.m = np.append(self.m, self.mtot)


    def details(self):
        print ('mtot', self.mtot)
        print ('rtot', self.rtot)
        print ('rhobar', self.rhobar)
        print ('nz', self.nz)
        print ('smallness m', self.small)
        print ('rhoc/rhobar', self.rho[0] / self.rhobar)

        import matplotlib.pyplot as plt
        plt.plot(self.r, self.rho, '.')

class evol(ongp.evol):
    """
    make a static model with ongp.evol.static, and provide some extra methods that let us
        (a) re-evaluate the density from the assumed equations of state provided with updated pressure and a new core mass
        (b) optimize some input parameters (z and y) to match some output parameters (rtot and mean y)
    """

    def adjust_mcore(self, mcore, add_zone=False):

        """
        recompute full density profile to satisfy a new value for the core mass. does what the above function does and more.
        the resulting model will *not* satisfy hydrostatic balance or the original total mass. this function is designed
        just to enforce the assumed eos; theory of figures code worries about hydrostatic balance and continuity.
        """

        if mcore == self.mcore:
            print ('same mcore, do nothing')
            return

        self.mcore_old = self.mcore
        self.kcore_old = self.kcore
        self.rho_old = np.copy(self.rho)

        # create a slightly different mass grid so that core boundary is at right place to satisfy desired core mass.
        # remember, self.mass has been set to tof's m_calc before tof calls this routine.

        # banana
        self.mcore = mcore

        if mcore > 0.:

            self.kcore = np.where(self.m < self.mcore * const.mearth)[0][-1] + 1 # kcore - 1 is last zone with m < mcore

            if (not add_zone) or self.kcore == self.kcore_old:
                # just adjust P at core boundary to more accurately reflect the core-top pressure for desired core mass
                alpha = (self.mcore * const.mearth - self.m[self.kcore]) / (self.m[self.kcore + 1] - self.m[self.kcore])
                self.p[self.kcore] = alpha * self.p[self.kcore + 1] + (1. - alpha) * self.p[self.kcore]
                self.t[self.kcore] = alpha * self.t[self.kcore + 1] + (1. - alpha) * self.t[self.kcore]
            else:
                # actually add a zone. must interpolate to get values for r, rho, t
                alpha = (self.mcore * const.mearth - self.m[self.kcore]) / (self.m[self.kcore + 1] - self.m[self.kcore])
                self.m = np.insert(self.m, self.kcore, mcore * const.mearth)
                self.dm = np.diff(self.m)

                self.p = np.insert(self.p, self.kcore, alpha * self.p[self.kcore + 1] + (1. - alpha) * self.p[self.kcore])
                self.t = np.insert(self.t, self.kcore, alpha * self.t[self.kcore + 1] + (1. - alpha) * self.t[self.kcore])
                self.rho = np.insert(self.rho, self.kcore, -1) # will be set below
                self.y = np.insert(self.y, self.kcore, -1)
                self.z = np.insert(self.z, self.kcore, -1)

                # derivatives - only calculated once tof converged, if save final model, just track
                # the changes of zone number
                self.gradt = np.insert(self.gradt, self.kcore, -1)
                self.brunt_b = np.insert(self.brunt_b, self.kcore, -1)
                self.chirho = np.insert(self.chirho, self.kcore, -1)
                self.chit = np.insert(self.chit, self.kcore, -1)
                self.chiy = np.insert(self.chiy, self.kcore, -1)
                self.dlogy_dlogp = np.insert(self.dlogy_dlogp, self.kcore, -1)

                # this too
                self.entropy = np.insert(self.entropy, self.kcore, -1)

                self.grada = np.insert(self.grada, self.kcore, \
                    alpha * self.grada[self.kcore + 1] + (1. - alpha) * self.grada[self.kcore])

                self.nz += 1
                self.ktrans += 1

        else: # mcore == 0. no need for extra precautions
            assert False
            self.kcore = 0 # lets a slice like [kcore:ktrans] still make sense

        mcore_grows = self.mcore > self.mcore_old

        # set new temperature in core
        if mcore_grows:
            # core temperature trivial, just a cooler point on the adiabat
            self.t[:self.kcore] = self.t[self.kcore]
        else:
            # core is hotter because the envelope adiabat goes deeper before we reach core material.
            # must integrate grada to find deeper temperature.
            for k in np.arange(self.kcore_old + 1)[::-1]:
                self.y[k] = self.y[k+1]
                if k == self.kcore - 1:
                    break
                self.grada[k] = self.hhe_eos.get_grada(np.log10(self.p[k]), np.log10(self.t[k]), self.y[k])
                dlnp = np.log(self.p[k]) - np.log(self.p[k+1])
                dlnt = self.grada[k] * dlnp
                self.t[k] = self.t[k+1] * (1. + dlnt)

            self.t[:self.kcore] = self.t[self.kcore]

        # set density within the core
        self.set_core_density()

        # identify molecular-metallic transition and set Z, Y
        self.locate_transition_pressure()

        # change yenv_inner slightly to relax toward correct mean envelope y
        self.m1 = np.sum(self.dm[self.ktrans:])
        self.m2 = np.sum(self.dm[self.kcore:self.ktrans])

        if self.yenv_inner < 0.:
            raise FitYmeanError('adjust_mcore got bad yenv_inner %f' % self.yenv_inner)
        elif self.yenv_inner > 1.:
            raise FitYmeanError('adjust_mcore got bad yenv_inner %f' % self.yenv_inner)

        self.set_yz()

        self.y[:self.kcore] = 0.
        self.z[:self.kcore] = 1.

        # set density in XYZ envelope
        self.set_envelope_density()





    def initialize_ballpark_ymean(self):
        def zero_me(y2):
            if not y2 > self.y1 and y2 < 1.:
                raise FitYmeanError('initialize_ballpark_ymean got bad y2 %f' % y2)
            elif np.isinf(y2):
                raise FitYmeanError('initialize_ballpark_ymean got bad y2 %f' % y2)
            self.static(mtot=self.mtot,
                        mcore=self.mcore,
                        zenv=self.z1,
                        zenv_inner=self.z2,
                        yenv=self.y1,
                        t1=self.t1,
                        transition_pressure=self.ptrans,
                        yenv_inner=y2)

        if self.ptrans > 3.:
            y2_guess = 0.8
        elif self.ptrans > 2.:
            y2_guess = 0.6
        else:
            y2_guess = 0.3

        zero_me(y2_guess)

        m1 = np.sum(self.dm[self.ktrans:])
        m2 = np.sum(self.dm[self.kcore:self.ktrans])

        y2_guess = (self.ym * (m1 + m2) - self.y1 * m1) / m2
        if y2_guess < self.y1: raise ValueError('tried y2 > y1')
        zero_me(y2_guess)

        # ym = (y1 * m1 + y2 * m2) / (m1 + m2)
        # ym * (m1 + m2) = y1 * m1 + y2 * m2
        # ym * (m1 + m2) - y1 * m1 = y2 * m2




    def fit_three_layer_to_ymean(self, verbose=1, xtol=1e2):
        if verbose >= 1:
            sys.stdout.write('search y2')
        try:
            def zero_me(x, verbose):
                y2, = x
                if not 0. < y2 < 1.:
                    print ('bad y2 value %f' % y2)
                    return np.array([np.inf])
                if verbose > 1: sys.stdout.write(' %f ' % y2)
                if verbose > 0: sys.stdout.write('.')
                try:
                    self.static(mtot=self.mtot,
                                mcore=self.mcore,
                                zenv=self.z1,
                                zenv_inner=self.z2,
                                yenv=self.y1,
                                t1=self.t1,
                                transition_pressure=self.ptrans,
                                yenv_inner=y2)
                except AtmError: # y2 too high -> gravity too high
                    raise FitYmeanError('got AtmError trying trying root find for y2. y2 = %f, g = %f' % (y2, self.surface_g * 1e-2))

                if not self.ktrans > self.kcore:
                    raise FitYmeanError('hydrogen transition pressure not reached in envelope. ktrans=%i, kcore=%i' % (self.ktrans, self.kcore))

                m1 = np.sum(self.dm[self.ktrans:])
                m2 = np.sum(self.dm[self.kcore:self.ktrans])
                ym_now = (self.y1 * m1 + y2 * m2) / (m1 + m2) # current mean Y of envelope material

                if verbose > 1: print ('\t%16.14f %5s %16.14f [kcore=%i ktrans=%8i]' % (y2, '-->', ym_now, self.kcore, self.ktrans))

                return np.array([(ym_now - self.ym) / self.ym])

            if self.ptrans > 3.:
                y2_guess = 0.8
            elif self.ptrans > 2.:
                y2_guess = 0.6
            else:
                y2_guess = 0.3

            guess = np.array([y2_guess])
            res = scipy.optimize.root(zero_me, guess, args=(verbose), tol=1e-8, options={'xtol':xtol, 'eps':1e-2})
            if not res['success']:
                raise FitYmeanError('failed in root find; res.x = %s' % res.x)
            self.y2 = res.x[0]

            m1 = np.sum(self.dm[self.ktrans:])
            m2 = np.sum(self.dm[self.kcore:self.ktrans])
            ym_actual = (self.y1 * m1 + self.y2 * m2) / (m1 + m2) # current mean Y of envelope material

            if np.abs(ym_actual - self.ym) > xtol:
                # print res
                raise FitYmeanError('mysterious failure in root find. root says success, but ym not correct.')



            sys.stdout.write('\n')
        except EOSError:
            print ('got an EOS error trying to build static model. params not physically plausible? mcore, z1, z2, y1, t1, ptrans:', self.mcore, self.z1, self.z2, self.y1, self.t1, self.ptrans)
            raise
        except AtmError:
            print ('got an Atm error trying to build static model. wrong t1 for this mass?')
            raise
