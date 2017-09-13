# a fourth-order theory of figures modelled closely on the summmary by Nettelmann (2017), https://arxiv.org/abs/1708.06177v1 (appendix B)
import numpy as np
import const
import scipy.optimize
import scipy.integrate
from scipy.special import legendre

class tof:
    def __init__(self, model, qrot, max_iters=10, ax=None):
        
        '''model is nominally an instance of the evolve.Evolver class, but 
        need only be an object having attributes called rho, p, and r.
        qrot is the smallness parameter, aka m.'''
        
        self.ax = ax
        
        self.max_iters = max_iters
        self.rho = np.copy(model.rho[1:])
        self.p = np.copy(model.p[1:])
        self.r = np.copy(model.r[1:]) # radii of the spherical model
        self.mtot = model.mtot
        
        print 'centermost radius %g' % self.r[0]

        self.qrot = qrot
        
        prho_tck = scipy.interpolate.splrep(self.p[::-1], self.rho[::-1], k=1)
        self.rho_of_p = lambda p: scipy.interpolate.splev(p, prho_tck)
        
        assert len(self.rho) == len(self.p), 'rho and p must have same length'
        # qrot :: the smallness parameter (m in N17)
        assert self.qrot >= 0, 'qrot must be positive.'
        assert self.qrot != 0., 'the nonrotating case is exceedingly boring.'
        assert self.qrot < 1., 'no supercritical rotation.'
                
        # l :: the radii of level surfaces
        self.l = self.r
                
        # rm :: mean radius of the outermost level surface. will change with shape
        self.rm = self.r[-1]

        # get the initial masses
        dl = np.diff(self.l)
        dl = np.insert(dl, 0, self.l[0])
        self.dm = 4. * np.pi * self.l ** 2 * self.rho * dl
        self.m = np.cumsum(self.dm)

        # rhobar :: mean density
        self.set_rhobar()
                
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
        self.ss0_alternate = np.zeros(self.nz)
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
        # self.set_f2n_f2np()
                
        
        # legendre polynomials for calculating radii from shape. 
        # provide functions of mu := cos(theta).
        # mu is equal to zero at the equator and unity at the pole.
        self.pp0 = np.poly1d(legendre(0))
        self.pp2 = np.poly1d(legendre(2))
        self.pp4 = np.poly1d(legendre(4))
        self.pp6 = np.poly1d(legendre(6))
        self.pp8 = np.poly1d(legendre(8))
        
        self.iterate()
        
        
    def iterate(self):

        print '%4s %10s %10s %10s %5s %15s %15s %15s %15s %15s %15s' % ('it', 'r_eq', 'r_mean', 'r_pol', 'rhom', 'pc', 'ss0(1)', 'ss2(1)', 'ss4(1)', 'ss6(1)', 'ss8(1)')
        
        for iteration in np.arange(self.max_iters):  
            
            self.iteration = iteration 
                                                               
            # update f_2n and f_2n^'
            self.set_f2n_f2np()

            # integrate for the S_2n and S_2n^'
            self.set_ss2n_ss2np()
            
            # get the total potential and integrate hydrostatic balance
            self.set_aa0()
            self.u = -4. / 3 * np.pi * const.cgrav * self.rhobar * self.l ** 2 * self.aa0
                                    
            self.du_dl = np.diff(self.u) / np.diff(self.l) # + const.cgrav * 4 * np.pi * self.l[1:] * self.rho[1:]
            self.dl = np.diff(self.l)
            
            if False: # check calculation of grad potential
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 5, figsize=(35, 4), gridspec_kw={'wspace':0.4})
                ax[0].plot(self.l, self.ss0, lw=1)
                ax[0].set_ylabel('S0')
                        
                ax[1].semilogy(self.l, self.ss0p, lw=1)
                ax[1].set_ylabel("S0'")
            
                ax[2].semilogy(self.l, self.aa0, lw=1)
                ax[2].set_ylabel('A0')
            
                [z.set_xlabel('l') for z in ax]
            
                ax[3].plot(self.l, self.u, lw=1)
                ax[3].set_xlabel('l')
                ax[3].set_ylabel('u')
                        
                ax[4].plot(self.l[1:], np.diff(self.u) / np.diff(self.l), lw=1)
                ax[4].set_xlabel('l')
                ax[4].set_ylabel('g=du/dl')
                return
            
            for k in np.arange(self.nz)[::-1]:
                if k == self.nz-1: continue
                dp = self.rho[k] * self.du_dl[k] * self.dl[k]
                
                if dp < 0:
                    print 'got negative dp: iter %i, k=%i, l=%g, aa0=%g, p=%g, rho=%g, dp=%g' % (self.iteration, k, self.l[k], self.aa0[k], self.p[k], self.rho[k], dp)
                    raise ValueError
                    
                self.p[k] = self.p[k+1] + dp

            self.rho = self.rho_of_p(self.p)

            self.dm = 4. * np.pi * self.l ** 2 * self.rho * np.insert(self.dl, 0, self.l[0])
            self.m = np.cumsum(self.dm)
            
            # scale up densities by a constant factor to match the total mass
            self.rho *= self.mtot / self.m[-1]
                            
            assert not np.any(self.l is np.inf)
            assert not np.any(self.l is np.nan)
            
            # calculate r(l, theta) on equator and pole; update rm
            self.r_eq = self.l * (1. \
                                        + self.s0 * self.pp0(0.) \
                                        + self.s2 * self.pp2(0.) \
                                        + self.s4 * self.pp4(0.) \
                                        + self.s6 * self.pp6(0.) \
                                        + self.s8 * self.pp8(0.) \
                                        )
            self.r_pol = self.l * (1. \
                                        + self.s0 * self.pp0(1.) \
                                        + self.s2 * self.pp2(1.) \
                                        + self.s4 * self.pp4(1.) \
                                        + self.s6 * self.pp6(1.) \
                                        + self.s8 * self.pp8(1.) \
                                        )

            # self.rm = self.r_eq[-1] ** (2. / 3) * self.r_pol[-1] ** (1. / 3)
            self.l = self.r_eq ** (2. / 3) * self.r_pol ** (1. / 3)
            self.rm = self.l[-1]
            self.set_rhobar() # sets mtot also
                        
            # solve for the figure functions s_2n
            self.set_s2n()
                    
            # update js
            self.set_j2n()
                    
            lw = 1
            alpha = (1. + iteration) / (1. + self.max_iters)
            
            if not self.ax is None:
                # _, = self.ax.plot(self.l, label='l', lw=1)
                # _, = self.ax.plot(self.l, self.r_eq, '--', label='req', lw=1)
                # self.ax.plot(self.l, self.r_pol, ':', label='rpol', lw=1, color=_.get_color())
                self.ax[0, 0].plot(self.l, self.s0, lw=1, label='s0')
                self.ax[0, 1].plot(self.l, self.s2, lw=1, label='s2')
                self.ax[0, 2].plot(self.l, self.s4, lw=1, label='s4')
                self.ax[0, 3].plot(self.l, self.s6, lw=1, label='s6')
                self.ax[0, 4].plot(self.l, self.s8, lw=1, label='s8')
                
                self.ax[1, 0].plot(self.l, self.ss0, lw=1, label='S0')
                self.ax[1, 1].plot(self.l, self.ss2, lw=1, label='S2')
                self.ax[1, 2].plot(self.l, self.ss4, lw=1, label='S4')
                self.ax[1, 3].plot(self.l, self.ss6, lw=1, label='S6')
                self.ax[1, 4].plot(self.l, self.ss8, lw=1, label='S8')

                self.ax[2, 0].plot(self.l, self.ss0p, lw=1, label='S0p')
                self.ax[2, 1].plot(self.l, self.ss2p, lw=1, label='S2p')
                self.ax[2, 2].plot(self.l, self.ss4p, lw=1, label='S4p')
                self.ax[2, 3].plot(self.l, self.ss6p, lw=1, label='S6p')
                self.ax[2, 4].plot(self.l, self.ss8p, lw=1, label='S8p')

                self.ax[3, 0].plot(self.l, self.aa0, lw=1, label='A0')
                self.ax[3, 1].plot(self.l, self.aa2, lw=1, label='A2')
                self.ax[3, 2].plot(self.l, self.aa4, lw=1, label='A4')
                self.ax[3, 3].plot(self.l, self.aa6, lw=1, label='A6')
                self.ax[3, 4].plot(self.l, self.aa8, lw=1, label='A8')
                
                [[z.legend() for z in y] for y in self.ax]
                
                                                
            print '%4i %10.5g %10.5g %10.5g %5.3f %15.5g %15.10f %15.10f %15.10f %15.10f %15.10f' % (iteration, self.r_eq[-1], self.l[-1], self.r_pol[-1], self.rhobar, self.p[0], self.ss0[-1], self.ss2[-1], self.ss4[-1], self.ss6[-1], self.ss8[-1]) 
            print '%59s %15s %15s %15s %15s %15s' % ('', 'mtot', 'j2', 'j4', 'j6', 'j8')
            print '%59s %15.10g %15.10f %15.10f %15.10f %15.10f' % ('', np.sum(self.dm), self.j2, self.j4, self.j6, self.j8) 
            print
            
                                    
    def set_ss2n_ss2np(self):
        '''eq. (B.9)'''
        
        self.z = self.l / self.rm
                
        for k in np.arange(self.nz):
            
            # this form seems less noisy near surface
            self.ss0[k] = self.m[k] / self.mtot / self.z[k] ** 3
            
            # this form doesn't diverge toward innermost point
            # self.ss0[k] = self.rho[k] / self.rhobar * self.f0[k] \
                            # - 1. / self.z[k] ** 3. * \
                            # scipy.integrate.trapz(self.z[:k] ** 3. / self.rhobar, x=self.rho[:k])
        
            self.ss2[k] = self.rho[k] / self.rhobar * self.f2[k] \
                            - 1. / self.z[k] ** (3. + 2) * \
                            scipy.integrate.trapz(self.z[:k] ** (3. + 2) * self.f2[:k] / self.rhobar, x=self.rho[:k])
            self.ss4[k] = self.rho[k] / self.rhobar * self.f4[k] \
                            - 1. / self.z[k] ** (3. + 4) * \
                            scipy.integrate.trapz(self.z[:k] ** (3. + 4) * self.f4[:k] / self.rhobar, x=self.rho[:k])
            self.ss6[k] = self.rho[k] / self.rhobar * self.f6[k] \
                            - 1. / self.z[k] ** (3. + 6) * \
                            scipy.integrate.trapz(self.z[:k] ** (3. + 6) * self.f6[:k] / self.rhobar, x=self.rho[:k])
            self.ss8[k] = self.rho[k] / self.rhobar * self.f8[k] \
                            - 1. / self.z[k] ** (3. + 8) * \
                            scipy.integrate.trapz(self.z[:k] ** (3. + 8) * self.f8[:k] / self.rhobar, x=self.rho[:k])

            self.ss0p[k] = -1. * self.rho[k] / self.rhobar * self.f0p[k] \
                            + 1. / self.z[k] ** 2. * (self.rho[-1] / self.rhobar * self.f0p[-1] \
                            - scipy.integrate.trapz(self.z[k:] ** 2. * self.f0p[k:] / self.rhobar, x=self.rho[k:]))
            self.ss2p[k] = -1. * self.rho[k] / self.rhobar * self.f2p[k] \
                            + (self.rho[-1] / self.rhobar * self.f2p[-1] \
                            - scipy.integrate.trapz(self.f2p[k:] / self.rhobar, x=self.rho[k:]))
            self.ss4p[k] = -1. * self.rho[k] / self.rhobar * self.f4p[k] \
                            + 1. / self.z[k] ** (2. - 4) * (self.rho[-1] / self.rhobar * self.f4p[-1] \
                            - scipy.integrate.trapz(self.z[k:] ** (2. - 4) * self.f4p[k:] / self.rhobar, x=self.rho[k:]))
            self.ss6p[k] = -1. * self.rho[k] / self.rhobar * self.f6p[k] \
                            + 1. / self.z[k] ** (2. - 6) * (self.rho[-1] / self.rhobar * self.f6p[-1] \
                            - scipy.integrate.trapz(self.z[k:] ** (2. - 6) * self.f6p[k:] / self.rhobar, x=self.rho[k:]))
            self.ss8p[k] = -1. * self.rho[k] / self.rhobar * self.f8p[k] \
                            + 1. / self.z[k] ** (2. - 8) * (self.rho[-1] / self.rhobar * self.f8p[-1] \
                            - scipy.integrate.trapz(self.z[k:] ** (2. - 8) * self.f8p[k:] / self.rhobar, x=self.rho[k:]))        
        
    def set_rhobar(self):
        self.mtot = np.sum(self.dm)
        self.rhobar = 3. * self.mtot / 4. / np.pi / self.rm ** 3
        
    def set_f2n_f2np(self):
        """eqs. (B.16) and (B.17)"""
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
                    + self.qrot / 3 * (1. - 2. / 5 * self.s2 - 9. / 35 * self.s2 ** 2 \
                    - 4. / 35 * self.s2 * self.s4 + 22. / 525 * self.s2 ** 3)
                                            
    def set_s2n(self):
        
        def zero_me(s2n, ss0, ss2, ss4, ss6, ss8, ss2p, ss4p, ss6p, ss8p, qrot):

            s2, s4, s6, s8 = s2n

            aa2 = (-1. * s2 + 2. / 7 * s2 ** 2 + 4. / 7 * s2 * s4 \
                    - 29./ 35 * s2 ** 3 + 100. / 693 * s4 ** 2 + 454. / 1155 * s2 ** 4 \
                    - 36. / 77 * s2 ** 2 * s4) * ss0 \
                    + (1. - 6. / 7 * s2 - 6. / 7 * s4 + 111. / 35 * s2 ** 2 \
                    - 1242. * 385 * s2 ** 3 + 144. / 77 * s2 * s4) * ss2 \
                    + (-10. / 7 * s2 - 500. / 693 * s4 + 180. / 77 * s2 ** 2) * ss4 \
                    + (1. + 4. / 7 * s2 + 1. / 35 * s2 ** 2 + 4. / 7 * s4 \
                    - 16. / 105 * s2 ** 3 + 24. / 77 * s2 * s4) * ss2p \
                    + (8. / 7 * s2 + 72. / 77 * s2 ** 2 + 400. / 693 * s4) * ss4p \
                    + qrot / 3 * (-1. + 10. / 7 * s2 + 9. / 35 * s2 ** 2 \
                    - 4. / 7 * s4 + 20. / 77 * s2 * s4 - 26. / 105 * s2 ** 3)
                    
            aa4 = (-1. * s4 + 18. / 35 * s2 ** 2 - 108. / 385 * s2 ** 3 + 40. / 77 * s2 * s4 \
                    + 90. / 143 * s2 * s6 + 162. / 1001 * s4 ** 2 + 16902. / 25025 * s2 ** 4 \
                    - 7369. / 5005 * s2 ** 2 * s4) * ss0 \
                    + (-54. / 35 * s2 - 60. / 77 * s4 + 648. / 385 * s2 ** 2 - 135. / 143 * s6 \
                    + 21468. / 5005 * s2 * s4 - 122688. / 25025 * s2 ** 3) * ss2 \
                    + (1. - 100. / 77 * s2 - 810. / 1001 * s4 + 6368. / 1001 * s2 ** 2) * ss4 \
                    - 315. / 143 * s2 * ss6 \
                    + (36. / 35 * s2 + 108. / 385 * s2 ** 2 + 40. / 77 * s4 + 3578. / 5005 * s2 * s4 \
                    - 36. / 175 * s2 ** 3 + 90. / 143 * s6) * ss2p \
                    + (1. + 80. / 77 * s2 + 1346. / 1001 * s2 ** 2 + 648. / 1001 * s4) * ss4p \
                    + 270. / 143 * s2 * ss6p \
                    + qrot / 3 * (-36. / 35 * s2 + 114. / 77 * s4 + 18. / 77 * s2 ** 2 \
                    - 978. / 5005 * s2 * s4 + 36. / 175 * s2 ** 3 - 90. / 143 * s6)
                    
            aa6 = (-1. * s6 + 10. / 11 * s2 * s4 - 18. / 77 * s2 ** 3 + 28. / 55 * s2 * s6 \
                    + 72. / 385 * s2 ** 4 + 20. / 99 * s4 ** 2 - 54. / 77 * s2 ** 2 * s4) * ss0 \
                    + (-15. / 11 * s4 + 108. / 77 * s2 ** 2 - 42. / 55 * s6 - 144. / 77 * s2 ** 3 \
                    + 216. / 77 * s2 * s4) * ss2 \
                    + (-25. / 11 * s2 - 100. / 99 * s4 + 270. / 77 * s2 ** 2) * ss4 \
                    + (1. - 98. / 55 * s2) * ss6 \
                    + (10. / 11 * s4 + 18. / 77 * s2 ** 2 + 36. / 77 * s2 * s4 + 28. / 55 * s6) * ss2p \
                    + (20. / 11 * s2 + 108. / 77 * s2 ** 2 + 80. / 99 * s4) * ss4p \
                    + (1. + 84. / 55 * s2) * ss6p \
                    + qrot / 3 * (-10. / 11 * s4 - 18. / 77 * s2 ** 2 + 34. / 77 * s2 * s4 + 82. / 55 * s6)
                    
            aa8 = (-1. * s8 + 56. / 65 * s2 * s6 + 72. / 715 * s2 ** 4 + 490. / 1287 * s4 ** 2 \
                    - 84. / 143 * s2 ** 2 * s4) * ss0 \
                    + (-84. / 65 * s6 - 144. / 143 * s2 ** 3 + 336. / 143 * s2 * s4) * ss2 \
                    + (-2450. / 1287 * s4 + 420. / 143 * s2 ** 2) * ss4 \
                    - 196. / 65 * s2 * ss6 \
                    + ss8 \
                    + (56. / 65 * s6 + 56. / 143 * s2 * s4) * ss2p \
                    + (1960. / 1287 * s4 + 168. / 143 * s2 ** 2) * ss4p \
                    + 168. / 65 * s2 * ss6p \
                    + ss8p \
                    + qrot / 3 * (-56. / 65 * s6 - 56. / 143 * s2 * s4)
                
            return np.array([aa2, aa4, aa6, aa8])
            
        # def minimize_me(s2n, ss0, ss2, ss4, ss6, ss8, ss2p, ss4p, ss6p, ss8p, qrot):
        #     aa2, aa4, aa6, aa8 = zero_me(s2n, ss0, ss2, ss4, ss6, ss8, ss2p, ss4p, ss6p, ss8p, qrot)
        #     return aa2 ** 2. + aa4 ** 2. + aa6 ** 2. + aa8 ** 2.
            
        def callback(x, f):
            print x, f
                        
        # if self.iteration == 0:
        #     fatol = 1e-1
        # elif self.iteration == 1:
        #     fatol = 1e-2
        # elif self.iteration == 2:
        #     fatol = 1e-3
        # else:
        #     fatol = 1e-7
            
        import time
        t0 = time.time()
        for k in np.arange(self.nz):
            # x0 = np.array([self.s2[k], self.s4[k], self.s6[k], self.s8[k]])
            x0 = np.zeros(4)
            args = (self.ss0[k], self.ss2[k], self.ss4[k], self.ss6[k], self.ss8[k],
                    self.ss2p[k], self.ss4p[k], self.ss6p[k], self.ss8p[k], self.qrot)
            sol = scipy.optimize.root(zero_me, x0, args=args, method='lm', callback=None)
            # sol = scipy.optimize.minimize(minimize_me, x0, args=args, method='BFGS')
            # solution vector (shape = (4, 1024)) is (s2, s4, s6, s8)

            # store solution
            self.s2[k], self.s4[k], self.s6[k], self.s8[k] = sol.x
            
            # store residuals
            self.aa2[k], self.aa4[k], self.aa6[k], self.aa8[k] = zero_me(sol.x, 
                        self.ss0[k], self.ss2[k], self.ss4[k], self.ss6[k], self.ss8[k],
                        self.ss2p[k], self.ss4p[k], self.ss6p[k], self.ss8p[k], self.qrot)
                        
            if not sol.success:
                print 'failed in s_2n solve in zone', k
                print
                print 'l', self.l[k]
                print '%20s ' * 5 % ('', 'aa2', 'aa4', 'aa6', 'aa8')
                print ('%20s ' + '%20g ' * 4) % ('', self.aa2[k], self.aa4[k], self.aa6[k], self.aa8[k])
                print
                print '%20s ' * 5 % ('ss0', 'ss2', 'ss4', 'ss6', 'ss8')
                print '%20g ' * 5 % (self.ss0[k], self.ss2[k], self.ss4[k], self.ss6[k], self.ss8[k])
                print
                print '%20s ' * 5 % ('', 'ss2p', 'ss4p', 'ss6p', 'ss8p')
                print ('%20s ' + '%20g ' * 4)  % ('', self.ss2p[k], self.ss4p[k], self.ss6p[k], self.ss8p[k])
                print
                print sol
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, 5, figsize=(35, 10), gridspec_kw={'wspace':0.4})
                
                ax[0, 0].semilogx(self.l, self.ss0, '.')
                ax[0, 0].set_ylabel('S0')
                ax[0, 1].semilogx(self.l, self.ss2, '.')
                ax[0, 1].set_ylabel('S2')
                ax[0, 2].semilogx(self.l, self.ss4, '.')
                ax[0, 2].set_ylabel('S4')
                ax[0, 3].semilogx(self.l, self.ss6, '.')
                ax[0, 3].set_ylabel('S6')
                ax[0, 4].semilogx(self.l, self.ss8, '.')
                ax[0, 4].set_ylabel('S8')

                ax[1, 0].semilogx(self.l, self.ss0p, '.')
                ax[1, 0].set_ylabel('S0p')
                ax[1, 1].semilogx(self.l, self.ss2p, '.')
                ax[1, 1].set_ylabel('S2p')
                ax[1, 2].semilogx(self.l, self.ss4p, '.')
                ax[1, 2].set_ylabel('S4p')
                ax[1, 3].semilogx(self.l, self.ss6p, '.')
                ax[1, 3].set_ylabel('S6p')
                ax[1, 4].semilogx(self.l, self.ss8p, '.')
                ax[1, 4].set_ylabel('S8p')
                
                [[z.set_xlabel(r'$l$') for z in y] for y in ax]
                raise RuntimeError('failed in s_2n solve.')

                        
        # eq. (B.2)
        self.s0 = -1. / 5 * self.s2 ** 2 - 2. / 105 * self.s2 ** 3 \
                    - 1. / 9 * self.s4 ** 2 - 2. / 35 * self.s2 ** 2 * self.s4
        # print 's2n loop et %f s' % (time.time() - t0)

        return
        
    def set_j2n(self):
        '''eq. (B.11)'''
        # J_2n :: the harmonic coefficients. zero for the sphere
        self.j2 = - 1. * (self.rm / self.r_eq[-1]) ** 4. * self.ss2[-1]
        self.j4 = - 1. * (self.rm / self.r_eq[-1]) ** 8. * self.ss4[-1]
        self.j6 = - 1. * (self.rm / self.r_eq[-1]) ** 12. * self.ss6[-1]
        self.j8 = - 1. * (self.rm / self.r_eq[-1]) ** 16. * self.ss8[-1]
        return
        