# a fourth-order theory of figures based on
# the summmary in appendix B of Nettelmann (2017)
# https://arxiv.org/abs/1708.06177v1
# 2017arXiv170806177N

import numpy as np
import const
from scipy.optimize import root
from scipy.integrate import cumtrapz
from scipy.interpolate import splrep, splev
from scipy.special import legendre
import time
import os
from importlib import reload

import models
import ongp; reload(ongp)

class ConvergenceError(Exception):
    pass

class tof4:
    def __init__(self, params, mesh_params):

        # parse contents of the params dictionary
        self.max_iters_outer = params['max_iters_outer']
        self.max_iters_inner = params['max_iters_inner']
        self.verbosity = params['verbosity']

        self.j2n_rtol = params['j2n_rtol'] # satisfaction criterion for inner iterations
        self.mtot_rtol = params['mtot_rtol'] # satisfaction criterion for outer iterations

        self.nz = params['nz']

        if not 'save_vector_output' in params.keys():
            params['save_vector_output'] = True

        self.params = params
        self.mesh_params = mesh_params

        if 'output_path' in params.keys():
            output_path = params['output_path']
        else:
            output_path = 'output'
        if 'uid' in params.keys():
            self.uid = params['uid']
        else:
            self.uid = str(time.time())
        self.output_path = '%s/%s' % (output_path, self.uid) # one run goes into a unique subfolder within params['output_path']

        self.small = params['small'] # the smallness parameter, m
        assert self.small >= 0., 'small must be non-negative.'
        assert self.small < 1., 'no supercritical rotation fam.'
        if self.small == 0. and self.verbosity > 0: print( 'warning: rotation is off.')

        if 'method_for_aa2n_solve' in params.keys():
            self.method_for_aa2n_solve = params['method_for_aa2n_solve']
        else:
            self.method_for_aa2n_solve = 'full'

    def initialize_model(self):
        self.mtot = self.params['mtot']
        self.rtot = self.params['req'] * (1. - self.small ** 2)

        # mesh_params = {}
        # for key, value in self.params.items():
        #     if 'mesh' in key:
        #         mesh_params[key] = value
        mesh_params = self.mesh_params

        evol_params = {
        'nz':self.nz,
        'z_eos_option':self.params['z_eos_option'],
        'atm_option':self.params['atm_option'],
        }

        model = ongp.evol(evol_params, mesh_params)

        if 'evol_verbosity' in self.params.keys():
            self.evol_verbosity = self.params['evol_verbosity']
        else:
            self.evol_verbosity = 0

        static_args = {
            'mtot':self.params['mtot'],
            't1':self.params['t1'],
            'yenv':self.params['ym'], # flat helium abundance for first pass
            'mcore':self.params['mcore'],
            'zenv':self.params['z1'],
            'zenv_inner':self.params['z2'],
            'transition_pressure':self.params['ptrans']
        }
        if 'teq' in list(self.params):
            static_args['teq'] = self.params['teq']
        model.static(static_args)

        # could do one more pass to try and get y1 and y2 close to desired

        # check total mass right off the bat
        dm = 4. / 3 * np.pi * model.rho[1:] * (model.r[1:] ** 3 - model.r[:-1] ** 3)
        dm = np.insert(dm, 0, 4. / 3 * np.pi * model.rho[0] * model.r[0] ** 3)
        m_calc = np.cumsum(dm)
        mtot_calc = m_calc[-1]
        print( 'initial model: mtot_calc, reldiff', mtot_calc, abs(self.mtot - mtot_calc) / self.mtot)

        self.rho = model.rho
        self.p = model.p
        self.l = model.r

        self.model = models.model_container()
        self.model.z1 = self.params['z1']
        self.model.z2 = self.params['z2']
        self.model.ptrans = self.params['ptrans']
        self.model.y1 = self.params['y1']
        self.model.y2 = 0
        self.model.ym = self.params['ym']
        self.model.t1 = self.params['t1']
        self.model.mcore = model.mcore
        self.model.kcore = model.kcore
        self.model.ktrans = model.ktrans

        self.model.grada = model.grada

        self.model.hhe_eos = model.hhe_eos
        self.model.z_eos = model.z_eos
        self.model.z_eos_low_t = model.z_eos_low_t
        self.model.z_eos_option = self.params['z_eos_option']

    def initialize_tof_vectors(self):
        if self.l[0] == 0.:
            self.l[0] = self.l[1] / 2
        self.rm = self.l[-1]

        self.nz = len(self.rho)
        # r_pol :: the polar radii
        self.r_pol = np.zeros(self.nz)
        # r_eq :: equatorial radii
        self.r_eq = np.zeros(self.nz)
        # s_2n :: the figure functions
        self.s0 = np.zeros(self.nz)
        self.s2 = np.zeros(self.nz)
        self.s4 = np.zeros(self.nz)
        self.s6 = np.zeros(self.nz)
        self.s8 = np.zeros(self.nz)
        # A_2n
        self.aa0 = np.zeros(self.nz)
        self.aa2 = np.zeros(self.nz)
        self.aa4 = np.zeros(self.nz)
        self.aa6 = np.zeros(self.nz)
        self.aa8 = np.zeros(self.nz)
        # S_2n
        self.ss0 = np.zeros(self.nz)
        self.ss2 = np.zeros(self.nz)
        self.ss4 = np.zeros(self.nz)
        self.ss6 = np.zeros(self.nz)
        self.ss8 = np.zeros(self.nz)
        # S_2n^'
        self.ss0p = np.zeros(self.nz)
        self.ss2p = np.zeros(self.nz)
        self.ss4p = np.zeros(self.nz)
        self.ss6p = np.zeros(self.nz)
        self.ss8p = np.zeros(self.nz)
        # set f0 (only needs to be done once)
        self.f0 = np.ones(self.nz)
        # J_2n
        self.j2 = 0.
        self.j4 = 0.
        self.j6 = 0.
        self.j8 = 0.
        self.j2n = np.array([self.j2, self.j4, self.j6, self.j8])

        # legendre polynomials for calculating radii from shape.
        # these provide scalar functions of mu := cos(theta).
        self.pp0 = np.poly1d(legendre(0))
        self.pp2 = np.poly1d(legendre(2))
        self.pp4 = np.poly1d(legendre(4))
        self.pp6 = np.poly1d(legendre(6))
        self.pp8 = np.poly1d(legendre(8))


    def relax(self):

        color_by_outer_iteration = {0:'dodgerblue', 1:'gold', 2:'firebrick', 3:'forestgreen', 4:'purple', 5:'coral', 6:'teal'}

        import time
        time_start_outer = time.time()

        import matplotlib.pyplot as plt

        try:
            _ = self.mtot
        except AttributeError as e:
            print('%s; did you run self.initialize_model?' % e.args[0])
            raise

        try:
            _ = self.rm
        except AttributeError as e:
            print('%s; did you run self.initialize_tof_vectors?' % e.args[0])
            raise

        self.rhobar = 3. * self.mtot / 4 / np.pi / self.rm ** 3 # this will be updated with mtot_calc

        self.outer_done = False
        self.j2n_last_outer = np.zeros(4)

        for self.outer_iteration in np.arange(self.max_iters_outer):

            if self.outer_iteration == self.max_iters_outer - 1:
                raise ConvergenceError('tof outer iteration failed to converge after %i iterations.' % self.max_iters_outer)

            self.inner_done = False
            time_start_inner = time.time()
            for self.inner_iteration in np.arange(self.max_iters_inner):

                # relax the shape for this rho(l)

                self.set_f2n_f2np() # polynomials of shape functions
                self.set_ss2n_ss2np() # integrals over mass distribution weighted by f_2n, f_2n'
                self.set_s2n() # numerical solve to get shape functions from A_2n = 0
                self.set_req_rpol() # get new vectors of equatorial and polar radii from shape functions
                self.set_j2n() # functions only of S_2n at surface and R_eq

                # if j2n are converged for this rho(l), we're done with inner iterations
                if np.all(self.j2n != 0) and self.inner_iteration > 0:
                    if np.all(abs(self.dj2n) / self.j2n + 1e-14 < self.j2n_rtol):
                        if self.verbosity > 1:
                            data = self.inner_iteration, \
                                    self.j0, self.j2, self.j4, self.j6, self.j8, \
                                    self.r_eq[-1], self.rm, self.r_pol[-1], \
                                    self.q
                            print( ('%15i ' + ('%15.10f ' * 5) + ('%15g ' * 4)) % data)
                        if self.verbosity > 2:
                            print ('terminate inner loop; all dj2n/j2n < %g.' % self.j2n_rtol)
                        self.inner_done = True # but don't break yet, in case there's more output of interest

                if self.verbosity > 1:
                    # print various scalars
                    if self.inner_iteration == 0:
                        names = 'i_inner', "J0", "J2", "J4", "J6", "J8", 'R_eq', 'R_mean', 'R_pol', 'q'
                        print( '%15s ' * len(names) % names)
                    if not self.inner_done:
                        data = self.inner_iteration, \
                                self.j0, self.j2, self.j4, self.j6, self.j8, \
                                self.r_eq[-1], self.rm, self.r_pol[-1], \
                                self.q
                        print( ('%15i ' + ('%15.10f ' * 5) + ('%15g ' * 4)) % data)

                if self.inner_done or self.small == 0.:
                    break

            # we hit max inner iters although dj2n have not converged.
            # this is fine as long as it doesn't happen for late-stage outer iterations.
            if not self.inner_done and self.verbosity > 1:
                data = self.inner_iteration, \
                        self.j0, self.j2, self.j4, self.j6, self.j8, \
                        self.r_eq[-1], self.rm, self.r_pol[-1], \
                        self.small * (self.r_eq[-1] / self.rm) ** 3
                print (('%15i ' + ('%15.10f ' * 5) + ('%15g ' * 4)) % data)
                if self.small > 0. and self.verbosity > 2:
                    print ('warning: shape might not be fully converged.')

            self.et_inner_loop = time.time() - time_start_inner

            #
            # get the total potential from shape
            #
            self.set_aa0() # from the s_2n, S_2n, S_2n^'
            # NM recommends that for the sake of calculating U, for the sake of ease of late-stage
            # convergence, use target total mass rather than m_calc.
            self.rhobar_target = 3. * self.mtot / 4. / np.pi / self.rm ** 3
            self.u = -4. / 3 * np.pi * const.cgrav * self.rhobar_target * self.l ** 2 * self.aa0

            #
            # integrate hydrostatic balance and continuity
            #
            self.dl = np.diff(self.l)
            self.dl = np.insert(self.dl, 0, self.l[0])
            self.du = np.diff(self.u)

            if self.verbosity > 2: print( 'integrating hydro...')
            self.p[-1] = 1e6
            self.rho_mid = 0.5 * (self.rho[1:] + self.rho[:-1])
            for k in np.arange(self.nz)[::-1]:
                if k == self.nz - 1: continue
                assert self.rho[k+1] > 0., 'hydro integration found a bad density; cannot continue.'
                assert self.du[k] != 0., 'hydro integration found no change in potential; cannot continue.'
                self.p[k] = self.p[k+1] + self.rho_mid[k] * self.du[k]

            # MONGOOSE

            # integrate continuity for current mass distribution
            dm = 4. / 3 * np.pi * self.rho[1:] * (self.l[1:] ** 3 - self.l[:-1] ** 3)
            dm = np.insert(dm, 0, 4. / 3 * np.pi * self.rho[0] * self.l[0] ** 3)
            self.m_calc = np.cumsum(dm)

            self.mtot_calc = self.m_calc[-1]
            if self.verbosity > 2:
                print( 'check continuity before touch density: mtot_calc, reldiff', self.mtot_calc, abs(self.mtot_calc - self.mtot) / self.mtot)

            # adjust mcore in underlying model, without reintegrating hydrostatic balance or continuity. only enforces the physical eos.
            if abs(self.mtot_calc - self.mtot) / self.mtot > self.mtot_rtol or abs(self.model.ym - self.params['ym']) > 1e-6:
                if self.verbosity > 2:
                    print( 'tof will adjust core mass')
                self.new_mcore = self.model.mcore + (self.mtot - self.mtot_calc) / const.mearth * 2. / 3 # put the 2/3 back in 02202018

                if self.verbosity > 1:
                    print ('current mcore (me) %f, total mass (g) = %g, missing mass (me) = %f, new core mass (me) = %f' % \
                        (self.model.mcore, self.mtot_calc, (self.mtot - self.mtot_calc) / const.mearth, self.new_mcore))

                if self.new_mcore <= 0.:
                    self.new_mcore = 2. / 3 * self.model.mcore
                elif self.new_mcore > 40.:
                    self.new_mcore = 0.5 * (self.mcore + 30.)

                if self.new_mcore <= 1e-2:
                    raise models.UnphysicalParameterError('model wants negative core mass')

                self.adjust() # adjust mcore and y2

                # reintegrate continuity for updated mass distribution
                dm = 4. / 3 * np.pi * self.rho[1:] * (self.l[1:] ** 3 - self.l[:-1] ** 3)
                dm = np.insert(dm, 0, 4. / 3 * np.pi * self.rho[0] * self.l[0] ** 3)
                self.m_calc = np.cumsum(dm)
                self.mtot_calc = self.m_calc[-1]

                # this is rhobar for all purposes except for the rhobar appearing in U(A0).
                self.rhobar = self.mtot_calc * 3. / 4 / np.pi / self.l[-1] ** 3

                if self.verbosity > 2:
                    print( 'after adjust mcore: mtot_calc, reldiff', self.mtot_calc, abs(self.mtot_calc - self.mtot) / self.mtot)
                    print( 'rhobar', self.rhobar)

            self.et_total = time.time() - time_start_outer

            if self.verbosity > 1 or (self.outer_iteration == 0 and self.verbosity > 0):
                names = 'i_outer', 'rhobar', 'mtot_calc', 'mcore', 'kcore', 'nz', 'y2', 'ym', 'et_total', 'inr_itrs', 'j2', 'j4', 'j6', 'req', 'rm'
                print (('%8s ' + '%11s ' * (len(names) - 1)) % names)
            if self.verbosity > 0:
                data = self.outer_iteration, self.rhobar, self.mtot_calc, self.model.mcore, self.model.kcore, self.nz, \
                        self.model.y2, self.model.ym, self.et_total, self.inner_iteration, \
                        self.j2, self.j4, self.j6, \
                        self.r_eq[-1], self.rm
                print (('%8i ' + '%11.5g ' * (len(data) - 1)) % data)


            # scale mean radii to match surface equatorial radius
            self.l *= self.params['req'] / self.r_eq[-1]
            self.rm_old = self.rm
            self.rm = self.l[-1]

            # if change in all j2n since last outer iteration is within tolerance,
            # there's nothing to be gained from adjusting the mass distribution--we're done.
            if np.all(abs((self.j2n_last_outer - self.j2n) / self.j2n) < self.j2n_rtol) \
                and self.outer_iteration > 10 \
                and self.inner_done \
                and abs(self.mtot_calc - self.mtot) / self.mtot < self.mtot_rtol:

                if self.verbosity >= 3.:
                    print ('terminate outer loop; all dj2n/j2n < %g.' % self.j2n_rtol)

                self.outer_done = True

                # BANANA
                self.set_s2n(force_full=True)

                self.model_summary_path = str(time.time())
                self.save_model_summary()

                break

            self.j2n_last_outer = self.j2n

        return


    def adjust(self):
                        # self.model = model_container()
                        # self.model.z1 = self.params['z1']
                        # self.model.z2 = self.params['z2']
                        # self.model.ptrans = self.params['ptrans']
                        # self.model.y1 = self.params['y1']
                        # self.model.y2 = 0
                        # self.model.ym = self.params['ym']
                        # self.model.t1 = self.params['t1']
                        # self.model.mcore = model.mcore
                        # self.model.kcore = model.kcore
                        # self.model.ktrans = model.ktrans
                        #
                        # self.model.grada = model.grada
                        #
                        # self.model.hhe_eos = model.hhe_eos
                        # self.model.z_eos = model.z_eos
                        # self.model.z_eos_low_t = model.z_eos_low_t

        old_mcore = self.model.old_mcore = self.model.mcore
        self.model.old_kcore = self.model.kcore
        mcore = self.model.mcore = self.new_mcore
        kcore = self.model.kcore = np.where(self.m_calc < mcore * const.mearth)[0][-1]

        if self.outer_iteration > 30:
            # add a zone
            alpha_m = (mcore * const.mearth - self.m_calc[kcore]) / (self.m_calc[kcore+1] - self.m_calc[kcore])
            assert alpha_m > 0.
            self.m_calc = np.insert(self.m_calc, kcore, mcore * const.mearth)
            self.p = np.insert(self.p, kcore+1, alpha_m * self.p[kcore + 1] + (1. - alpha_m) * self.p[kcore])
            self.l = np.insert(self.l, kcore+1, alpha_m * self.l[kcore + 1] + (1. - alpha_m) * self.l[kcore])
            # print 'alpha_m', alpha_m
            # print 'm', self.m_calc[kcore-3:kcore+4]
            # print 'l', self.l[kcore-3:kcore+4]
            # print 'p', self.p[kcore-3:kcore+4]
            self.rho = np.insert(self.rho, kcore, 0.) # calculated below

            self.model.grada = np.insert(self.model.grada, kcore, self.model.grada[kcore])

            self.add_point_to_figure_functions()
            self.nz += 1
            kcore += 1
            self.model.kcore = kcore

        if np.any(np.diff(self.l) <= 0.):
            raise ValueError('radius not monotone increasing')

        ktrans = self.model.ktrans = np.where(self.p > self.model.ptrans * 1e12)[0][-1] # updated p

        t = np.zeros_like(self.p)
        y = np.zeros_like(self.p)
        z = np.zeros_like(self.p)

        z[:kcore] = 1.
        z[kcore:ktrans] = self.model.z2
        z[ktrans:] = self.model.z1

        y[:kcore] = 0.
        if self.model.y2:
            y[kcore:ktrans] = self.model.y2
            y[ktrans:] = self.model.y1
        else:
            y[kcore:] = self.model.ym

        t[-1] = self.model.t1

        try:
            assert not np.any(np.isnan(self.p)), 'nans in p during adjust mcore, before t integration'
            assert not np.any(np.isnan(t)), 'nans in t during adjust mcore, before t integration'
            assert not np.any(np.isnan(self.model.grada)), 'nans in grada during adjust mcore, before t integration'
        except AssertionError as e:
            raise models.EOSError(e.args[0])

        for k in np.arange(self.nz)[::-1]:
            if k == self.nz - 1: continue
            dlnp = np.log(self.p[k]) - np.log(self.p[k+1]) # positive
            # grada = self.model.hhe_eos.get_grada(np.log10(self.p[k+1]), np.log10(t[k+1]), y[k+1])
            # self.model.grada[k+1] = grada
            dlnt = self.model.grada[k+1] * dlnp
            t[k] = t[k+1] * (1. + dlnt)
            if k == kcore:
                t[:k] = t[k+1]
                break
        self.model.grada[:kcore] = 0.

        assert not np.any(np.isnan(t)), 'nans in t during adjust mcore, after t integration'
        self.model.grada[kcore:] = self.model.hhe_eos.get_grada(np.log10(self.p[kcore:]), np.log10(t[kcore:]), y[kcore:])

        self.rho[:kcore] = self.model.get_rho_z(np.log10(self.p[:kcore]), np.log10(t[:kcore]))
        self.rho[kcore:] = self.model.get_rho_xyz(np.log10(self.p[kcore:]), np.log10(t[kcore:]), y[kcore:], z[kcore:])

        self.model.t = t
        self.model.y = y
        self.model.z = z

        # figure out y2 for next pass
        m1 = self.mtot_calc - self.m_calc[ktrans]
        m2 = self.m_calc[self.model.ktrans] - self.m_calc[self.model.kcore]
        self.model.ym = (self.model.y1 * m1 + self.model.y2 * m2) / (m1 + m2) # current actual Y_mean
        if self.outer_iteration % 1 == 0:
            self.model.y2 = (self.params['ym'] * (m1 + m2) - self.model.y1 * m1) / m2 # new y2 to match desired Y_mean
            if self.model.y2 >= 1.:
                raise models.FitYmeanError('got y2 > 1 in adjust')

        return

    def add_point_to_figure_functions(self):
        self.s0 = np.insert(self.s0, self.model.kcore, self.s0[self.model.kcore])
        self.s2 = np.insert(self.s2, self.model.kcore, self.s2[self.model.kcore])
        self.s4 = np.insert(self.s4, self.model.kcore, self.s4[self.model.kcore])
        self.s6 = np.insert(self.s6, self.model.kcore, self.s6[self.model.kcore])
        self.s8 = np.insert(self.s8, self.model.kcore, self.s8[self.model.kcore])

        self.f0 = np.insert(self.f0, self.model.kcore, self.f0[self.model.kcore])
        self.f2 = np.insert(self.f2, self.model.kcore, self.f2[self.model.kcore])
        self.f4 = np.insert(self.f4, self.model.kcore, self.f4[self.model.kcore])
        self.f6 = np.insert(self.f6, self.model.kcore, self.f6[self.model.kcore])
        self.f8 = np.insert(self.f8, self.model.kcore, self.f8[self.model.kcore])

        self.f0p = np.insert(self.f0p, self.model.kcore, self.f0p[self.model.kcore])
        self.f2p = np.insert(self.f2p, self.model.kcore, self.f2p[self.model.kcore])
        self.f4p = np.insert(self.f4p, self.model.kcore, self.f4p[self.model.kcore])
        self.f6p = np.insert(self.f6p, self.model.kcore, self.f6p[self.model.kcore])
        self.f8p = np.insert(self.f8p, self.model.kcore, self.f8p[self.model.kcore])

        self.ss0 = np.insert(self.ss0, self.model.kcore, self.ss0[self.model.kcore])
        self.ss2 = np.insert(self.ss2, self.model.kcore, self.ss2[self.model.kcore])
        self.ss4 = np.insert(self.ss4, self.model.kcore, self.ss4[self.model.kcore])
        self.ss6 = np.insert(self.ss6, self.model.kcore, self.ss6[self.model.kcore])
        self.ss8 = np.insert(self.ss8, self.model.kcore, self.ss8[self.model.kcore])

        self.ss0p = np.insert(self.ss0p, self.model.kcore, self.ss0p[self.model.kcore])
        self.ss2p = np.insert(self.ss2p, self.model.kcore, self.ss2p[self.model.kcore])
        self.ss4p = np.insert(self.ss4p, self.model.kcore, self.ss4p[self.model.kcore])
        self.ss6p = np.insert(self.ss6p, self.model.kcore, self.ss6p[self.model.kcore])
        self.ss8p = np.insert(self.ss8p, self.model.kcore, self.ss8p[self.model.kcore])


    def set_scaled_barotrope(self):

        # what is current mtot_calc
        dm = 4. / 3 * np.pi * self.rho[1:] * (self.l[1:] ** 3 - self.l[:-1] ** 3)
        dm = np.insert(dm, 0, 4. / 3 * np.pi * self.rho[0] * self.l[0] ** 3)
        self.m_calc = np.cumsum(dm)
        self.mtot_calc = self.m_calc[-1]

        if self.verbosity > 4:
            print( 'in set_scaled_barotrope')
            print( 'mtot_calc', self.mtot_calc)
            print( 'rescale rho by', self.mtot / self.mtot_calc)

        # scale density so that we match total mass
        self.rho *= self.mtot / self.mtot_calc

        # reintegrate continuity to make sure it's what we expect
        dm = 4. / 3 * np.pi * self.rho[1:] * (self.l[1:] ** 3 - self.l[:-1] ** 3)
        dm = np.insert(dm, 0, 4. / 3 * np.pi * self.rho[0] * self.l[0] ** 3)
        self.m_calc = np.cumsum(dm)
        self.mtot_calc = self.m_calc[-1]

        relative_mass_error = abs((self.mtot_calc - self.mtot)/self.mtot)
        assert relative_mass_error < self.mtot_rtol, 'still have wrong mass after set_scaled_barotrope: calc, target, rdiff, rtol: %g %g %g %g' % (self.mtot_calc, self.mtot, relative_mass_error, self.mtot_rtol)

        if self.verbosity > 4:
            print ('new mtot_calc', self.mtot_calc)

        # get the total potential from shape
        self.set_aa0() # from the s_2n, S_2n, S_2n^'
        self.rhobar = 3. * self.mtot_calc / 4. / np.pi / self.rm ** 3
        self.u = -4. / 3 * np.pi * const.cgrav * self.rhobar * self.l ** 2 * self.aa0

        # integrate hydrostatic balance
        self.dl = np.diff(self.l)
        self.dl = np.insert(self.dl, 0, self.l[0])
        self.du = np.diff(self.u)

        if self.verbosity > 4:
            print ('reintegrate hydro: old p_c', self.p[0])

        self.p[-1] = 1e6
        self.rho_mid = 0.5 * (self.rho[1:] + self.rho[:-1])
        for k in np.arange(self.nz)[::-1]:
            if k == self.nz - 1: continue
            assert self.rho[k+1] > 0., 'hydro integration found a bad density; cannot continue.'
            assert self.du[k] != 0., 'hydro integration found no change in potential; cannot continue.'
            self.p[k] = self.p[k+1] + self.rho_mid[k] * self.du[k]

        if self.verbosity > 4:
            print ('reintegrate hydro: new p_c', self.p[0])

    def save_model_summary(self, do_gyre=False):
        """assumes model is converged. saves tof parameter/scalar/vector data, and usually also tells ongp to save a profile and gyre model."""
        import pickle

        # write a record of warnings if this model is not up to snuff for one reason or another
        relative_mass_error = abs((self.mtot_calc - self.mtot)/self.mtot)
        if relative_mass_error > self.mtot_rtol or not self.inner_done or not self.outer_done:
            if relative_mass_error > self.mtot_rtol:
                raise ConvergenceError('attempted to save model summary for model with mass error exceeding specified tolerance. relerr, rtol: %g %g' % (relative_mass_error, self.mtot_rtol))
            if not self.inner_done:
                raise ConvergenceError('attempted to save model summary with inner_done == False')
            if not self.outer_done:
                raise ConvergenceError('attempted to save model summary with outer_done == False')

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        scalars = {}
        scalars['uid'] = self.uid # unique identifier
        scalars['nz'] = self.nz
        scalars['j2'] = self.j2
        scalars['j4'] = self.j4
        scalars['j6'] = self.j6
        scalars['j8'] = self.j8
        scalars['req'] = self.r_eq[-1]
        scalars['rpol'] = self.r_pol[-1]
        scalars['rm'] = self.rm
        scalars['rhobar'] = self.rhobar
        scalars['mtot_calc'] = self.mtot_calc
        scalars['small'] = self.small
        scalars['q'] = self.q
        scalars['et_total'] = self.et_total
        try:
            scalars['mcore'] = self.model.mcore
        except:
            pass
        try:
            scalars['y2'] = self.model.y2
        except:
            pass

        if not self.params['save_vector_output']:
            with open('%s/tof4_params_scalars.pkl' % self.output_path, 'w') as f:
                pickle.dump((self.params, scalars), f)
            return

        vectors = {}
        vectors['l'] = self.l
        vectors['req'] = self.r_eq
        vectors['rpol'] = self.r_pol
        vectors['rho'] = self.rho
        vectors['p'] = self.p
        vectors['u'] = self.u
        vectors['m_calc'] = self.m_calc
        # was skipping these for a while
        vectors['s0'] = self.s0
        vectors['s2'] = self.s2
        vectors['s4'] = self.s4
        vectors['s6'] = self.s6
        vectors['s8'] = self.s8

        vectors['ss0'] = self.ss0
        vectors['ss2'] = self.ss2
        vectors['ss4'] = self.ss4
        vectors['ss6'] = self.ss6
        vectors['ss8'] = self.ss8

        with open('%s/tof4_data.pkl' % self.output_path, 'wb') as f:
            pickle.dump((self.params, scalars, vectors), f)

        if do_gyre:
            self.model.r = self.l
            self.model.rtot = self.model.r[-1]

            self.model.p = self.p
            self.model.m = self.m_calc
            self.model.mtot = self.mtot_calc
            self.model.rho = self.rho
            self.model.nz = self.nz
            self.model.lint = 0.

            self.model.set_derivatives_etc()
            self.model.save_gyre_output('%s/model.gyre' % self.output_path)




    def set_req_rpol(self):
        '''
        calculate equatorial and polar radius vectors from the figure functions s_2n and legendre polynomials P_2n.
        see N17 eq. (B.1) or ZT78 eq. (27.1).
        also calculates q from m and new r_eq[-1].
        '''

        # equator: mu = cos(pi/2) = 0
        self.r_eq = self.l * (1. + self.s0 * self.pp0(0.) \
                                 + self.s2 * self.pp2(0.) \
                                 + self.s4 * self.pp4(0.) \
                                 + self.s6 * self.pp6(0.) \
                                 + self.s8 * self.pp8(0.))

        # pole: mu = cos(0) = 1
        self.r_pol = self.l * (1. + self.s0 * self.pp0(1.) \
                                  + self.s2 * self.pp2(1.) \
                                  + self.s4 * self.pp4(1.) \
                                  + self.s6 * self.pp6(1.) \
                                  + self.s8 * self.pp8(1.))

        self.q = self.small * (self.r_eq[-1] / self.rm) ** 3


    def set_ss2n_ss2np(self):
        '''
        N17 eq. (B.9).
        '''

        self.z = self.l / self.rm

        ss2_integral = cumtrapz(self.z ** (2. + 3) * self.f2 / self.rhobar, x=self.rho, initial=0.)
        ss4_integral = cumtrapz(self.z ** (4. + 3) * self.f4 / self.rhobar, x=self.rho, initial=0.)
        ss6_integral = cumtrapz(self.z ** (6. + 3) * self.f6 / self.rhobar, x=self.rho, initial=0.)
        ss8_integral = cumtrapz(self.z ** (8. + 3) * self.f8 / self.rhobar, x=self.rho, initial=0.)

        # integrals from 0 to z
        ss0p_integral = cumtrapz(self.z ** (2. - 0) * self.f0p / self.rhobar, x=self.rho, initial=0.)
        ss2p_integral = cumtrapz(self.z ** (2. - 2) * self.f2p / self.rhobar, x=self.rho, initial=0.)
        ss4p_integral = cumtrapz(self.z ** (2. - 4) * self.f4p / self.rhobar, x=self.rho, initial=0.)
        ss6p_integral = cumtrapz(self.z ** (2. - 6) * self.f6p / self.rhobar, x=self.rho, initial=0.)
        ss8p_integral = cumtrapz(self.z ** (2. - 8) * self.f8p / self.rhobar, x=self.rho, initial=0.)

        # int_z^1 = int_0^1 - int_0^z
        ss0p_integral = ss0p_integral[-1] - ss0p_integral
        ss2p_integral = ss2p_integral[-1] - ss2p_integral
        ss4p_integral = ss4p_integral[-1] - ss4p_integral
        ss6p_integral = ss6p_integral[-1] - ss6p_integral
        ss8p_integral = ss8p_integral[-1] - ss8p_integral


        if False:
            self.ss0 = self.m / self.mtot / self.z ** 3 # (B.8)
        else:
            # this form doesn't require explicit knowledge of m or mtot.
            self.ss0 = self.rho / self.rhobar * self.f0 \
                            - 1. / self.z ** 3. * cumtrapz(self.z ** 3. / self.rhobar * self.f0, x=self.rho, initial=0.) # (B.9)

        self.ss2 = self.rho / self.rhobar * self.f2 - 1. / self.z ** (2. + 3) * ss2_integral
        self.ss4 = self.rho / self.rhobar * self.f4 - 1. / self.z ** (4. + 3) * ss4_integral
        self.ss6 = self.rho / self.rhobar * self.f6 - 1. / self.z ** (6. + 3) * ss6_integral
        self.ss8 = self.rho / self.rhobar * self.f8 - 1. / self.z ** (8. + 3) * ss8_integral

        self.ss0p = -1. * self.rho / self.rhobar * self.f0p + 1. / self.z ** (2. - 0) \
                    * (self.rho[-1] / self.rhobar * self.f0p[-1] - ss0p_integral)

        self.ss2p = -1. * self.rho / self.rhobar * self.f2p + 1. / self.z ** (2. - 2) \
                    * (self.rho[-1] / self.rhobar * self.f2p[-1] - ss2p_integral)

        self.ss4p = -1. * self.rho / self.rhobar * self.f4p + 1. / self.z ** (2. - 4) \
                    * (self.rho[-1] / self.rhobar * self.f4p[-1] - ss4p_integral)

        self.ss6p = -1. * self.rho / self.rhobar * self.f6p + 1. / self.z ** (2. - 6) \
                    * (self.rho[-1] / self.rhobar * self.f6p[-1] - ss6p_integral)

        self.ss8p = -1. * self.rho / self.rhobar * self.f8p + 1. / self.z ** (2. - 8) \
                    * (self.rho[-1] / self.rhobar * self.f8p[-1] - ss8p_integral)

    def set_f2n_f2np(self):
        '''
        N17 eqs. (B.16) and (B.17).
        '''

        self.f2 = 3. / 5 * self.s2 + 12. / 35 * self.s2 ** 2 + 6. / 175 * self.s2 ** 3 \
                    + 24. / 35 * self.s2 * self.s4 + 40. / 231 * self.s4 ** 2 \
                    + 216. / 385 * self.s2 ** 2 * self.s4 - 184. / 1925 * self.s2 ** 4

        self.f4 = 1. / 3 * self.s4 + 18. / 35 * self.s2 ** 2 + 40. / 77 * self.s2 * self.s4 \
                    + 36. / 77 * self.s2 ** 3 + 90. / 143 * self.s2 * self.s6 \
                    + 162. / 1001 * self.s4 ** 2 + 6943. / 5005 * self.s2 ** 2 * self.s4 \
                    + 486. / 5005 * self.s2 ** 4

        self.f6 = 3. / 13 * self.s6 + 120. / 143 * self.s2 * self.s4 + 72. / 143 * self.s2 ** 3 \
                    + 336. / 715 * self.s2 * self.s6 + 80. / 429 * self.s4 ** 2 \
                    + 216. / 143 * self.s2 ** 2 * self.s4 + 432. / 715 * self.s2 ** 4

        self.f8 = 3. / 17 * self.s8 + 168. / 221 * self.s2 * self.s6 + 2450. / 7293 * self.s4 ** 2 \
                    + 3780. / 2431 * self.s2 ** 2 * self.s4 + 1296. / 2431 * self.s2 ** 4

        self.f0p = 3. / 2 - 3. / 10 * self.s2 ** 2 - 2. / 35 * self.s2 ** 3 - 1. / 6 * self.s4 ** 2 \
                    - 6. / 35 * self.s2 ** 2 * self.s4 + 3. / 50 * self.s2 ** 4

        self.f2p = 3. / 5 * self.s2 - 3. / 35 * self.s2 ** 2 - 6. / 35 * self.s2 * self.s4 \
                    + 36. / 175 * self.s2 ** 3 - 10. / 231 * self.s4 ** 2 - 17. / 275 * self.s2 ** 4 \
                    + 36. / 385 * self.s2 ** 2 * self.s4

        self.f4p = 1. / 3 * self.s4 - 9. / 35 * self.s2 ** 2 - 20. / 77 * self.s2 * self.s4 \
                    - 45. / 143 * self.s2 * self.s6 - 81. / 1001 * self.s4 ** 2 + 1. / 5 * self.s2 ** 2 * self.s4
                    # f4p has an s_2**3 in Z+T. NN says it shouldn't be there (Oct 4 2017).

        self.f6p = 3. / 13 * self.s6 - 75. / 143 * self.s2 * self.s4 + 270. / 1001 * self.s2 ** 3 \
                    - 50. / 429 * self.s4 ** 2 + 810. / 1001 * self.s2 ** 2 * self.s4 - 54. / 143 * self.s2 ** 4 \
                    - 42. / 143 * self.s2 * self.s6

        self.f8p = 3. / 17 * self.s8 - 588. / 1105 * self.s2 * self.s6 - 1715. / 7293 * self.s4 ** 2 \
                    + 2352. / 2431 * self.s2 ** 2 * self.s4 - 4536. / 12155 * self.s2 ** 4

    def set_aa0(self):
        self.aa0 = (1. + 2. / 5 * self.s2 ** 2 - 4. / 105 * self.s2 ** 3 + 2. / 9 * self.s4 ** 2 \
                    + 43. / 175 * self.s2 ** 4 - 4. / 35 * self.s2 ** 2 * self.s4) * self.ss0 \
                    + (-3. / 5 * self.s2 + 12. / 35 * self.s2 ** 2 - 234. / 175 * self.s2 ** 3 \
                    + 24. / 35 * self.s2 * self.s4) * self.ss2 \
                    + (-5. / 9 * self.s4 + 6. / 7 * self.s2 ** 2) * self.ss4 \
                    + self.ss0p \
                    + (2. / 5 * self.s2 + 2. / 35 * self.s2 ** 2 + 4. / 35 * self.s2 * self.s4 \
                    - 2. / 25 * self.s2 ** 3) * self.ss2p \
                    + (4. / 9 * self.s4 + 12. / 35 * self.s2 ** 2) * self.ss4p \
                    + self.small / 3 * (1. - 2. / 5 * self.s2 - 9. / 35 * self.s2 ** 2 \
                    - 4. / 35 * self.s2 * self.s4 + 22. / 525 * self.s2 ** 3)

    def set_s2n(self, force_full=False):
        '''
        performs a root find to find the figure functions s_2n from the current S_2n, S_2n^prime, and m.
        '''

        def aa2n(s2n, k):
            """B.12-15"""
            s2, s4, s6, s8 = s2n

            aa2 = (-1. * s2 + 2. / 7 * s2 ** 2 + 4. / 7 * s2 * s4 - 29. / 35 * s2 ** 3 + 100. / 693 * s4 ** 2 \
                    + 454. / 1155 * s2 ** 4 - 36. / 77 * s2 ** 2 * s4) * self.ss0[k] \
                    + (1. - 6. / 7 * s2 - 6. / 7 * s4 + 111. / 35 * s2 ** 2 - 1242. / 385 * s2 ** 3 + 144. / 77 * s2 * s4) * self.ss2[k] \
                    + (-10. / 7 * s2 - 500. / 693 * s4 + 180. / 77 * s2 ** 2) * self.ss4[k] \
                    + (1. + 4. / 7 * s2 + 1. / 35 * s2 ** 2 + 4. / 7 * s4 - 16. / 105 * s2 ** 3 + 24. / 77 * s2 * s4) * self.ss2p[k] \
                    + (8. / 7 * s2 + 72. / 77 * s2 ** 2 + 400. / 693 * s4) * self.ss4p[k] \
                    + self.small / 3 * (-1. + 10. / 7 * s2 + 9. / 35 * s2 ** 2 - 4. / 7 * s4 + 20./ 77 * s2 * s4 - 26. / 105 * s2 ** 3)

            aa4 = (-1. * s4 + 18. / 35 * s2 ** 2 - 108. / 385 * s2 ** 3 + 40. / 77 * s2 * s4 + 90. / 143 * s2 * s6 + 162. / 1001 * s4 ** 2 \
                    + 16902. / 25025 * s2 ** 4 - 7369. / 5005 * s2 ** 2 * s4) * self.ss0[k] \
                    + (-54. / 35 * s2 - 60. / 77 * s4 + 648. / 385 * s2 ** 2 \
                    - 135. / 143 * s6 + 21468. / 5005 * s2 * s4 - 122688. / 25025 * s2 ** 3) * self.ss2[k] \
                    + (1. - 100. / 77 * s2 - 810. / 1001 * s4 + 6368. / 1001 * s2 ** 2) * self.ss4[k] \
                    - 315. / 143 * s2 * self.ss6[k] \
                    + (36. / 35 * s2 + 108. / 385 * s2 ** 2 + 40. / 77 * s4 + 3578. / 5005 * s2 * s4 \
                    - 36. / 175 * s2 ** 3 + 90. / 143 * s6) * self.ss2p[k] \
                    + (1. + 80. / 77 * s2 + 1346. / 1001 * s2 ** 2 + 648. / 1001 * s4) * self.ss4p[k] \
                    + 270. / 143 * s2 * self.ss6p[k] \
                    + self.small / 3 * (-36. / 35 * s2 + 114. / 77 * s4 + 18. / 77 * s2 ** 2 \
                    - 978. / 5005 * s2 * s4 + 36. / 175 * s2 ** 3 - 90. / 143 * s6)

            aa6 = (-s6 + 10. / 11 * s2 * s4 - 18. / 77 * s2 ** 3 + 28. / 55 * s2 * s6 + 72. / 385 * s2 ** 4 + 20. / 99 * s4 ** 2 \
                    - 54. / 77 * s2 ** 2 * s4) * self.ss0[k] \
                    + (-15. / 11 * s4 + 108. / 77 * s2 ** 2 - 42. / 55 * s6 - 144. / 77 * s2 ** 3 + 216. / 77 * s2 * s4) * self.ss2[k] \
                    + (-25. / 11 * s2 - 100. / 99 * s4 + 270. / 77 * s2 ** 2) * self.ss4[k] \
                    + (1. - 98. / 55 * s2) * self.ss6[k] \
                    + (10. / 11 * s4 + 18. / 77 * s2 ** 2 + 36. / 77 * s2 * s4 + 28. / 55 * s6) * self.ss2p[k] \
                    + (20. / 11 * s2 + 108. / 77 * s2 ** 2 + 80. / 99 * s4) * self.ss4p[k] \
                    + (1. + 84. / 55 * s2) * self.ss6p[k] \
                    + self.small / 3 * (-10. / 11 * s4 - 18. / 77 * s2 ** 2 + 34. / 77 * s2 * s4 + 82. / 55 * s6)

            aa8 = (-s8 + 56. / 65 * s2 * s6 + 72. / 715 * s2 ** 4 + 490. / 1287 * s4 ** 2 - 84. / 143 * s2 ** 2 * s4) * self.ss0[k] \
                    + (-84. / 65 * s6 - 144. / 143 * s2 ** 3 + 336. / 143 * s2 * s4) * self.ss2[k] \
                    + (-2450. / 1287 * s4 + 420. / 143 * s2 ** 2) * self.ss4[k] \
                    - 196. / 65 * s2 * self.ss6[k] \
                    + self.ss8[k] \
                    + (56. / 65 * s6 + 56. / 143 * s2 * s4) * self.ss2p[k] \
                    + (1960. / 1287 * s4 + 168. / 143 * s2 ** 2) * self.ss4p[k] \
                    + 168. / 65 * s2 * self.ss6p[k] \
                    + self.ss8p[k] \
                    + self.small / 3 * (-56. / 65 * s6 - 56. / 143 * s2 * s4)


            return np.array([aa2, aa4, aa6, aa8])

        if self.method_for_aa2n_solve == 'full' or force_full: # default

            for k in np.arange(self.nz):
                s2n = np.array([self.s2[k], self.s4[k], self.s6[k], self.s8[k]])
                sol = root(aa2n, s2n, args=(k)) # increasing xtol has essentially no effect
                if not sol['success']:
                    print (sol)
                    raise RuntimeError('failed in solve for s_2n.')

                # store solution
                self.s2[k], self.s4[k], self.s6[k], self.s8[k] = sol.x

                # store residuals for diagnostics
                # self.aa2[k], self.aa4[k], self.aa6[k], self.aa8[k] = aa2n(sol.x, k)

        elif 'cheb' in self.method_for_aa2n_solve or 'cubic' in self.method_for_aa2n_solve:
            try:
                fskip = int(self.method_for_aa2n_solve.split()[1])
            except IndexError:
                fskip = 10
            nskip = int(self.nz / fskip)
            # self.s2[:] = 0.
            # self.s4[:] = 0.
            # self.s6[:] = 0.
            # self.s8[:] = 0.

            for k in np.arange(self.nz)[::nskip]:
                s2n = np.array([self.s2[k], self.s4[k], self.s6[k], self.s8[k]])
                # sol = root(aa2n, s2n, args=(k)
                sol = root(aa2n, s2n, args=(k), options={'xtol':1e-4})
                if not sol['success']:
                    print (sol)
                    raise RuntimeError('failed in solve for s_2n.')

                # store solution
                self.s2[k], self.s4[k], self.s6[k], self.s8[k] = sol.x

            if 'cheb' in self.method_for_aa2n_solve:
                from numpy.polynomial.chebyshev import chebfit, chebval
                coef2 = chebfit(self.l[::nskip], self.s2[::nskip], 4)
                coef4 = chebfit(self.l[::nskip], self.s4[::nskip], 4)
                coef6 = chebfit(self.l[::nskip], self.s6[::nskip], 4)
                coef8 = chebfit(self.l[::nskip], self.s8[::nskip], 4)

                self.s2 = chebval(self.l, coef2)
                self.s4 = chebval(self.l, coef4)
                self.s6 = chebval(self.l, coef6)
                self.s8 = chebval(self.l, coef8)
            elif 'cubic' in self.method_for_aa2n_solve:
                tck2 = splrep(self.l[::nskip], self.s2[::nskip], k=3)
                tck4 = splrep(self.l[::nskip], self.s4[::nskip], k=3)
                tck6 = splrep(self.l[::nskip], self.s6[::nskip], k=3)
                tck8 = splrep(self.l[::nskip], self.s8[::nskip], k=3)

                self.s2 = splev(self.l, tck2)
                self.s4 = splev(self.l, tck4)
                self.s6 = splev(self.l, tck6)
                self.s8 = splev(self.l, tck8)

        else:
            raise ValueError('unable to parse aa2n solve method %s' % self.method_for_aa2n_solve)


        # note misprinted power on first term in Z+T (28.12)
        self.s0 = - 1. / 5 * self.s2 ** 2 \
                    - 2. / 105 * self.s2 ** 3 \
                    - 1. / 9 * self.s4 ** 2 \
                    - 2. / 35 * self.s2 ** 2 * self.s4

        return

    def set_j2n(self):
        '''
        J_2n :: the harmonic coefficients. eq. (B.11)
        '''

        self.j2n_old = self.j2n

        self.j2 = - 1. * (self.rm / self.r_eq[-1]) ** 2. * self.ss2[-1]
        self.j4 = - 1. * (self.rm / self.r_eq[-1]) ** 4. * self.ss4[-1]
        self.j6 = - 1. * (self.rm / self.r_eq[-1]) ** 6. * self.ss6[-1]
        self.j8 = - 1. * (self.rm / self.r_eq[-1]) ** 8. * self.ss8[-1]

        self.j0 = - self.ss0[-1]

        self.j2n = np.array([self.j2, self.j4, self.j6, self.j8])

        self.dj2n = self.j2n - self.j2n_old

        return

def run_one_tof4(params, mesh_params):
    '''
    basically the same as t=tof.tof4(params); t.relax(); but with a bunch of exception handling.
    will try and make things work and record them when they fail, but will keep going.'''
    import os
    import pickle
    # from models import EOSError, AtmError, HydroError, UnphysicalParameterError, FitYmeanError
    print ('trying', params['z1'], params['z2'], params['ptrans'], params['y1'])

    i = 0
    while True: # iterate until initial tof model creation works
        t = tof4(params, mesh_params)
        try:
            t.initialize_model()
            t.initialize_tof_vectors()
            break
        except models.FitYmeanError as e:
            # failed to relax y2 for the initial model build. 3 possibilities:
            # (1) y2 relax found ktrans < kcore. go again with a smaller core mass
            # (2) y2 relax got AtmError because gravity too high (too large a y2 guess)
            # (3) numerical root find itself fails to find a solution
            # can try tro work around issue (1). if (2) or (3), then give up on this model.
            if 'hydrogen transition pressure' in e.args[0]:
                params['mcore'] *= 0.5
                if params['mcore'] < 1e-4:
                    print ('failed to work around ktrans<kcore in initial model by guessing smaller cores')
                    i = 11
            elif 'mysterious' in e.args[0]:
                print ('fit_ymean thinks it converged but got bad solution. give up.')
                params['mcore'] += 3.5
                if params['mcore'] > 20:
                    print ('failed to work around mysterious y2 root find error by guessing larger cores')
                    i = 11
            elif 'bad y2' in e.args[0]:
                print ('tried bad y2 %s, give up' % e.args[0].split()[-1])
                i = 11
            else:
                raise
        except models.HydroError as e:
            # probably ongp failed to find mol-met transition for initial model.
            # try a smaller core mass.
            if 'no molecular-metallic transition' in e.args[0]:
                params['mcore'] *= 0.5
                if params['mcore'] < 1e-4:
                    print ('failed to work around ktrans<kcore in initial model by guessing smaller cores')
                    i = 11
            else:
                raise

        i += 1
        if i > 10:
            print ('***'*20)
            print ('failure', params['z1'], params['z2'], params['ptrans'], params['y1'])
            print (e.args[0])
            print ('***'*20)
            os.mkdir('output/%s' % t.uid)
            with open('output/%s/fail.params.pkl' % t.uid, 'w') as f:
                pickle.dump(params, f)
            return None

    i = 0
    while True: # iterate until tof iterations behave
        try:
            t.relax()
            print ('***'*20)
            print ('success', t.uid, params['z1'], params['z2'], params['ptrans'], params['y1'])
            print ('***'*20)
            break # while

        except models.FitYmeanError as e:
            print ('got FitYmeanError during iterations')
            i = 11

        except models.UnphysicalParameterError as e:
            if 'z inversion' in e.args[0]:
                print ('tried z inversion; no go')
                i = 11
            elif 'y inversion' in e.args[0]:
                print ('tried y inversion; no go')
                i = 11
            elif 'model wants negative core mass' in e.args[0]:
                print ('model wants negative core mass; probably not possible to fit R_eq for these params')
                i = 11
            else:
                print ('UnphysicalParameterError: %s; try increasing mcore guess' % e.args[0])
                params['mcore'] += 4.
                if params['mcore'] > 30.:
                    print ('failed to work around UnphysicalParameterError:')
                    print (e.args[0])
                    i = 11

        except models.EOSError as e:
            print ('EOSError; try increasing mcore')
            params['mcore'] += 3
            if params['mcore'] > 30:
                print ('failed to work around EOSError')
                i = 11
        except models.AtmError as e:
            print ('AtmError; try decreasing mcore')
            params['mcore'] -= 2
            if params['mcore'] < 0.:
                print ('failed to work around AtmError')
                i = 11

        except models.HydroError as e:
            print ('HydroError, probably issue finding molecular-metallic transition')
            i = 11

        except ConvergenceError as e:
            print ('ConvergenceError')
            i = 11

        except RuntimeError as e:
            raise
            print ('RuntimeError: %s' % e.args[0])
            print ('try different core mass')
            params['mcore'] -= 1.

        i += 1
        if i > 10:
            print ('***'*20)
            print ('failure', params['z1'], params['z2'], params['ptrans'], params['y1'])
            print (e.args[0])
            print ('***'*20)
            try:
                os.mkdir('output/%s' % t.uid)
            except OSError:
                print ('dir output/%s already exists; saving fail flag there' % t.uid)
            with open('output/%s/fail.params.pkl' % t.uid, 'w') as f:
                pickle.dump(params, f)
            print()
            return None
            break

    return t

def unpickle(path):
    if path == 'baseline_resample':
        path = '/Users/chris/Dropbox/planet_models/gravity/tof/runs/4096/output/typical'
    import pickle
    with open('%s/tof4_data.pkl' % path, 'rb') as f:
        try:
            return pickle.load(f, encoding='latin1')
        except:
            return pickle.load(f)

def radau(eig, small):
    # gives the first-order ellipticity using radau's approximation
    from scipy.integrate import cumtrapz, trapz
    z = 2. * cumtrapz(eig.rho * eig.r ** 4, x=eig.r, initial=0.) / 3. / eig.r ** 2 / cumtrapz(eig.rho * eig.r ** 2, x=eig.r, initial=0.)
    z[0] = z[1]
    eta = 25. / 4 * (1. - 3. / 2 * z) ** 2. - 1
    eps_surface = 5. * small / 2. / (eta[-1] + 2.)
    eps = np.ones_like(eta) * eps_surface
    for i in np.arange(len(eps)):
        if eig.x[i] == 0.: continue
        eps[i] *= np.exp(-trapz(eta[i:] / eig.x[i:], x=eig.x[i:]))
    return eps
