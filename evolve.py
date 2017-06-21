import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d, splrep, splev
from scipy.integrate import trapz, odeint
import sys
import const
import pickle
import utils
import time

# this module is adapted from Daniel P. Thorngren's general-purpose giant planet
# modelling code. the main modifications were the implementation of
# (1) the complete Saumon, Chabrier, & van Horn (1995) equation of state
#     for hydrogen and helium, rather than just solar-Y adiabats;
# (2) the Fortney+2011 model atmospheres for Jupiter and Saturn (could easily be
#     extended to Uranus and Neptune also);
# (3) the Lorenzen+2011 ab initio phase diagram for hydrogen and helium, obtained
#     courtesy of Nadine Nettelmann;
# (4) the Thompson ANEOS for heavy elements, including water, water ice, iron, 
#     serpentine, and dunite;
# (5) the Rostock eos (REOS) for water, also obtained courtesy of Nadine Nettelmann.
#     at present won't work for cool planet models since it only covers T > 1000 K; 
#     blends with aneos water at low T. reference: 2009PhRvB..79e4107F


class Evolver:
    
    def __init__(self,
        hhe_eos_option='scvh', 
        z_eos_option='reos water',
        atm_option='f11_tables',
        nz=1024,
        relative_radius_tolerance=1e-4,
        max_iters_for_static_model=500, 
        min_iters_for_static_model=12,
        mesh_func_type='tanh',
        extrapolate_phase_diagram_to_low_pressure=True):
        # Load in the equations of state
        if hhe_eos_option == 'scvh':
            import scvh; reload(scvh)
            self.hhe_eos = scvh.eos()
                        
            # load isentrope tables so we can use isentropic P-T profiles at solar composition
            # as starting guesses before we tweak Y, Z distribution
            solar_isentropes = np.load("data/scvhIsentropes.npz")
            self.logrho_on_solar_isentrope = RegularGridInterpolator(
                (solar_isentropes['entropy'], solar_isentropes['pressure']), solar_isentropes['density'])
            self.logt_on_solar_isentrope = RegularGridInterpolator(
                (solar_isentropes['entropy'], solar_isentropes['pressure']), np.log10(solar_isentropes['temperature']))             
        elif eos == 'militzer':
            raise NotImplementedError('no militzer EOS yet.')
        else:
            raise NotImplementedError("this EOS name is not recognized.")
        
        # eos for heavy elements
        # rockIceEOS = np.load("data/aneosRockIce.npz")
        # self.getRockIceDensity = interp1d(rockIceEOS['pressure'], rockIceEOS['density']) # returns log10 density   
        
        import aneos; reload(aneos)
        import reos; reload(reos)
        if z_eos_option == 'reos water':
            self.z_eos = reos.eos()
            self.z_eos_low_t = aneos.eos('water')
        elif 'aneos' in z_eos_option:
            material = z_eos_option.split()[1]
            self.z_eos = aneos.eos(material)
            self.z_eos_low_t = None
        else:
            raise ValueError("z eos option '%s' not recognized." % z_eos_option)
        self.z_eos_option = z_eos_option
            
        self.atm_option = atm_option

        # h-he phase diagram
        import lorenzen; reload(lorenzen)
        # self.phase = lorenzen.hhe_phase_diagram(t_offset=phase_t_offset,
                        # extrapolate_to_low_pressure=extrapolate_phase_diagram_to_low_pressure)
        self.phase = lorenzen.hhe_phase_diagram(extrapolate_to_low_pressure=extrapolate_phase_diagram_to_low_pressure)
                                          
        # model atmospheres are now initialized in self.staticModel, so that individual models
        # can be calculated with different model atmospheres within a single Evolver instance.
        # e.g., can run a Jupiter and then a Saturn without making a new Evolver instance.
        
        # initialize structure mesh function
        if mesh_func_type == 'sin':
            self.mesh_func = lambda t: np.sin(t) ** 3
        elif mesh_func_type == 'sin_alt':
            self.mesh_func = lambda t: 1. - np.sin(np.pi / 2 - t) ** 3
        elif mesh_func_type == 'tanh':
            self.mesh_func = lambda t: 0.5 * (1. + np.tanh(10. * (t - np.pi / 4)))
        elif mesh_func_type == 'tanh_9':
            self.mesh_func = lambda t: 0.5 * (1. + np.tanh(9. * (t - np.pi / 4)))
        elif mesh_func_type == 'tanh_8':
            self.mesh_func = lambda t: 0.5 * (1. + np.tanh(8. * (t - np.pi / 4)))
        elif mesh_func_type == 'tanh_7':
            self.mesh_func = lambda t: 0.5 * (1. + np.tanh(7. * (t - np.pi / 4)))
        elif mesh_func_type == 'tanh_5':
            self.mesh_func = lambda t: 0.5 * (1. + np.tanh(5. * (t - np.pi / 4)))
        elif mesh_func_type == 'tanh_3':
            self.mesh_func = lambda t: 0.5 * (1. + np.tanh(3. * (t - np.pi / 4)))
        else:
            raise ValueError('mesh function type %s not recognized' % mesh_func_type)
            
        # Initialize memory
        self.nz = self.nBins = nz
        self.workingMemory = np.zeros(self.nBins)
        
        self.k = np.arange(self.nz)
        self.p = np.zeros(self.nz)
        self.m = np.zeros(self.nz)
        self.r = np.zeros(self.nz)
        self.rho = np.zeros(self.nz)
        self.t = np.zeros(self.nz)
        self.y = np.zeros(self.nz)
        self.z = np.zeros(self.nz)
        
        # hydrostatic model is judged to be converged when the radius has changed by a relative amount less than
        # relative_radius_tolerance over both of the last two iterations.
        self.relative_radius_tolerance = relative_radius_tolerance
        self.max_iters_for_static_model = max_iters_for_static_model
        self.min_iters_for_static_model = min_iters_for_static_model
        
        self.have_rainout = False # until proven guilty
        self.have_rainout_to_core = False
        
        self.nz_gradient = 0
        self.nz_shell = 0
        
        return

    def getMixDensity_daniel(self, entropy, pressure, z):
        '''assumes isentropes at solar Y, and an aneos rock/ice blend for the Z component.'''
        return 1. / ((1. - z) / 10 ** self.logrho_on_solar_isentrope((entropy,np.log10(pressure))) +
                   z / 10 ** self.getRockIceDensity((np.log10(pressure))))
        
    # when building non-isentropic models [for instance, a grad(P, T, Y, Z)=grada(P, T, Y, Z) model
    # with Y, Z functions of depth so that entropy is non-uniform], want to be able to get density 
    # as a function of these same independent variables P, T, Y, Z.
    
    def get_rho_z(self, logp, logt):
        '''helper function to get rho of just the z component. different from self.z_eos.get_logrho because
        this does the switching to aneos water at low T if using reos water as the z eos.'''
        
        if self.z_eos_option == 'reos water':
            # if using reos water, extend down to T < 1000 K using aneos water.
            # for this, mask to find the low T part.            
            logp_high_t = logp[logt >= 3.]
            logt_high_t = logt[logt >= 3.]

            logp_low_t = logp[logt < 3.]
            logt_low_t = logt[logt < 3.]
            
            try:
                rho_z_low_t = 10 ** self.z_eos_low_t.get_logrho(logp_low_t, logt_low_t)
            except:
                print 'off low-t eos tables?'
                raise
            
            try:
                rho_z_high_t = 10 ** self.z_eos.get_logrho(logp_high_t, logt_high_t)
            except:
                print 'off high-t eos tables?'
                raise

            rho_z = np.concatenate((rho_z_high_t, rho_z_low_t))
                
        else:
            try:
                rho_z = 10 ** self.z_eos.get_logrho(logp, logt)
            except ValueError:
                raise ValueError('off z_eos tables.')

        return rho_z

    def get_rho_xyz(self, logp, logt, y, z):
        # only meant to be called when Z is non-zero and Y is not 0 or 1.
                        
        rho_hhe = 10 ** self.hhe_eos.get_logrho(logp, logt, y)
        rho_z = self.get_rho_z(logp, logt)
        rhoinv = (1. - z) / rho_hhe + z / rho_z
        return rhoinv ** -1

    def staticModel_adiabatic_solar(self,mtot=1.,entropy=7.,zenv=0.,mcore=0.):
        '''constructs an isentropic model of specified total mass, entropy (kb per baryon), envelope Z == 1 - fHHe, and dense core mass (earth masses).'''
        # lagrangian grid
        structure = self.workingMemory
        self.m = mass = mtot * const.mjup * \
            np.sin(np.linspace(0, np.pi / 2, self.nz)) ** 5 # this grid gives pretty crummy resolution at low pressure
        kcore = coreBin = int(np.where(mcore * const.mearth <= mass)[0][0])
        dm = dMass = np.diff(mass)
        q = np.zeros(self.nz)        
        # guess initial radius based on an isobar at a ballpark pressure of 1 Mbar
        self.p[:] = 1e12
        # density from eos
        self.rho[:kcore] = 10 ** self.getRockIceDensity((np.log10(self.p[:kcore])))
        self.rho[kcore:] = self.getMixDensity_daniel(entropy, self.p[kcore:], zenv)
        # continuity equation
        q[0] = 0.
        q[1:] = 3. * dm / 4 / np.pi / self.rho[1:] # something like the volume of a constant-density sphere - what is this again, daniel?
        self.r = np.cumsum(q) ** (1. / 3)

        # Get converged structure through relaxation method
        something_like_dp = np.zeros_like(self.p)
        oldRadii = (0, 0, 0)
        for iteration in xrange(500):
            # hydrostatic equilibrium
            something_like_dp[1:] = const.cgrav * self.m[1:] * dm / 4. / np.pi / self.r[1:] ** 4.
            self.p[:] = np.cumsum(something_like_dp[::-1])[::-1] + 10 * 1e6 # hydrostatic balance
            self.rho[:kcore] = 10 ** self.getRockIceDensity((np.log10(self.p[:kcore])))
            self.rho[kcore:] = self.getMixDensity_daniel(entropy, self.p[kcore:], zenv)
            q[0] = 0.
            q[1:] = 3. * dm / 4 / np.pi / self.rho[1:]
            self.r = np.cumsum(q) ** (1. / 3)
            if np.all(np.abs(np.mean((oldRadii / self.r[-1] - 1.))) < self.relative_radius_tolerance):
                break
            if not np.isfinite(self.r[-1]):
                return (np.nan, np.nan)
            oldRadii = (oldRadii[1], oldRadii[2], self.r[-1])
        else:
            return (np.nan, np.nan)
                    
        self.rtot = self.r[-1]
        self.surface_g = const.cgrav * mtot * const.mjup / self.rtot ** 2.
        something_like_dp[1:] = const.cgrav * self.m[1:] * dm / 4. / np.pi / self.r[1:] ** 4.
        self.p[:] = np.cumsum(something_like_dp[::-1])[::-1] + 10 * 1e6
        self.t[:kcore] = 0.
        self.t[kcore:] = 10 ** self.logt_on_solar_isentrope((entropy, np.log10(self.p[kcore:])))
        
        # linear extrapolate in logp to get a 10-bar temperature
        f = ((np.log10(1e7) - np.log10(self.p[-2]))) / (np.log10(self.p[-1]) - np.log10(self.p[-2]))
        logt10 = f * np.log10(self.t[-1]) + (1. - f) * np.log10(self.t[-2])
        self.t10 = 10 ** logt10   
        self.entropy = entropy
        
        dSdE = 1. / trapz(self.t, dx=dm)
        return self.r[-1], dSdE

    def equilibrium_y_profile(self, phase_t_offset, verbose=False, show_timing=False, minimum_y_is_envelope_y=False):
        '''uses the existing p-t profile to find the thermodynamic equilibrium y profile, which
        may require a nearly pure-helium layer atop the core.'''
        p = self.p * 1e-12 # Mbar
        k1 = np.where(p > 1.)[0][-1]
        ymax1 = self.phase.ymax_lhs(p[k1], self.t[k1] - phase_t_offset)
        
        if np.isnan(ymax1) or self.y[k1] < ymax1:
            if verbose: print 'first point at P > 1 Mbar is stable to demixing. Y, Ymax', self.y[k1], ymax1
            self.nz_gradient = 0
            self.nz_shell = 0
            return self.y
            
        self.ystart = np.copy(self.y)
        yout = np.copy(self.y)
        yout[k1:] = ymax1 # homogeneous molecular envelope at this abundance
        if verbose: print 'demix', k1, self.m[k1] / self.m[-1], p[k1], self.t[k1], self.y[k1], '-->', yout[k1]
                
        t0 = time.time()
        rainout_to_core = False
        
        # this can't go here since equilibrium_y_profile is called during iterations until Y gradient stops changing.
        # as far as this routine is concerned, in the last iteration it will find a stable Y configuration and thus count no gradient zones.
        # instead, only initialize nz_gradient back to zero if this routine finds that Y must be redistributed.
        # self.nz_gradient = 0 
        # self.k_shell_top = 0
        self.k_gradient_top = k1
        for k in np.arange(k1-1, self.kcore, -1): # inward from first point where P > 1 Mbar
            t1 = time.time()
            ymax = self.phase.ymax_lhs(p[k], self.t[k] - phase_t_offset) # April 12 2017: phase_t_offset here rather than in Lorenzen module
            if np.isnan(ymax):
                raise ValueError('got nan from ymax_lhs in initial loop over zones. p, t = %f, %f' % (p[k], self.t[k]))
            if show_timing: print 'zone %i: dt %f ms, t0 + %f seconds' % (k, 1e3 * (time.time() - t1), time.time() - t0)
            if yout[k] < ymax:
                if verbose: print 'stable', k, self.m[k] / self.m[-1], p[k], self.t[k], yout[k], ' < ', ymax, -1
                break
            self.nz_gradient = 0
            self.k_shell_top = 0
            ystart = yout[k]
            yout[k] = ymax
            
            if minimum_y_is_envelope_y and yout[k] < yout[k+1]:
                yout[k:] = yout[k]
            
            # difference between initial he mass and current proposed he mass above and including this zone
            # must be in the deeper interior.
            he_mass_missing_above = self.mhe - np.dot(yout[k:], self.dm[k-1:])
            enclosed_envelope_mass = np.sum(self.dm[self.kcore:k])
            if not enclosed_envelope_mass > 0: # at core boundary
                rainout_to_core = True
                yout[self.kcore:k] = 0.95 # since this is < 1., should still have an overall 'missing' helium mass in envelope, to be made up during outward shell iterations.
                # assert this "should"
                assert np.dot(yout[self.kcore:], self.dm[self.kcore-1:]) < self.mhe, 'problem: proposed envelope already has mhe > mhe_initial, even before he shell iterations. case: gradient reaches core'
                kbot = k
                break
            y_interior = he_mass_missing_above / enclosed_envelope_mass
            if y_interior > 1:
                # inner homogeneneous region of envelope would need Y > 1 to conserve global helium mass. thus undissolved droplets on core.
                # set the rest of the envelope to Y = 0.95, then do outward iterations to find how large of a shell is needed to conserve
                # the global helium mass.
                msg = 'would need Y > 1 in inner homog region; rainout to core.'
                if verbose: print msg
                rainout_to_core = True
                yout[self.kcore:k] = 0.95 # since this is < 1., should still have an overall 'missing' helium mass in envelope, to be made up during outward shell iterations.
                # assert this "should"
                # assert np.dot(yout[self.kcore:], self.dm[self.kcore-1:]) < self.mhe, 'problem: proposed envelope already has mhe > mhe_initial, even before he shell iterations. case: y_interior = %f > 1. kcore=%i, k=%i' % (y_interior, self.kcore, k)
                kbot = k
                break
            else:        
                yout[self.kcore:k] = y_interior
                self.nz_gradient += 1
                
        if verbose: print 'he gradient over %i zones. rainout to core %s' % (self.nz_gradient, rainout_to_core)
        if show_timing: print 't0 + %f seconds' % (time.time() - t0)
                
        if verbose: print self.mhe, np.dot(yout[self.kcore:], self.dm[self.kcore-1:]), np.dot(yout[self.kcore:self.nz-1], self.dm[self.kcore:])
                        
        if rainout_to_core:
            # gradient extends down to kbot, below which the rest of the envelope is already set Y=0.95.  since proposed envelope mhe < initial mhe, must grow the 
            # He-rich shell to conserve total mass.
            if verbose: print '%5s %5s %10s %10s %10s' % ('k', 'kcore', 'dm_k', 'mhe_tent', 'mhe')
            for k in np.arange(kbot, self.nz):
                yout[k] = 0.95 # in the future, obtain value from ymax_rhs(p, t)
                # should fix following line for the case where there is no core
                try:
                    tentative_total_he_mass = np.dot(yout[self.kcore:], self.dm[self.kcore-1:])
                except:
                    raise RuntimeError('equilibrium_y_profile fails when have rainout to core and no core. (fixable.)')
                if verbose: print '%5i %5i %10.5e %10.5e %10.5e' % (k, self.kcore, self.dm[k-1], tentative_total_he_mass, self.mhe)
                if tentative_total_he_mass >= self.mhe:
                    if verbose: print 'tentative he mass, initial total he mass', tentative_total_he_mass, self.mhe
                    rel_mhe_error = abs(self.mhe - tentative_total_he_mass) / self.mhe
                    if verbose: print 'satisfied he mass conservation to a relative precision of %f' % rel_mhe_error
                    # yout[k] = (self.mhe - (np.dot(yout[self.kcore:], self.dm[self.kcore-1:]) - yout[k] * self.dm[k-1])) / self.dm[k-1]
                    self.nz_shell = k - self.kcore
                    self.k_shell_top = k
                    break
                    
        self.have_rainout = self.nz_gradient > 0
        self.have_rainout_to_core = rainout_to_core
        if rainout_to_core: assert self.k_shell_top
                 
        return yout


    def staticModel(self, mtot=1., t10=300., yenv=0.27, zenv=0., mcore=0.,
                    zenv_inner=None,
                    include_he_immiscibility=False,
                    phase_t_offset=0,
                    minimum_y_is_envelope_y=False,
                    rrho_where_have_helium_gradient=None,
                    erase_z_discontinuity_from_brunt=False,
                    verbose=False):
        '''build a hydrostatic model with a given total mass mtot, 10-bar temperature t10, envelope helium mass fraction yenv,
            envelope heavy element mass fraction zenv, and ice/rock core mass mcore. returns the number of iterations taken before 
            convergence, or -1 for failure to converge.'''
        
        # model atmospheres
        if 0.9 <= mtot <= 1.1:
            if verbose: print 'using jupiter atmosphere tables'
            if self.atm_option is 'f11_tables':
                import f11_atm
                atm = f11_atm.atm('jup')
            elif self.atm_option is 'f11_fit':
                import f11_atm_fit
                atm = f11_atm_fit.atm('jup')
            else:
                raise ValueError('atm_option %s not recognized.' % self.atm_option)
            self.teq = 109.0
        elif 0.9 <= mtot * const.mjup / const.msat <= 1.1:
            if verbose: print 'using saturn atmosphere tables'
            if self.atm_option is 'f11_tables':
                import f11_atm
                atm = f11_atm.atm('sat')
            elif self.atm_option is 'f11_fit':
                import f11_atm_fit
                atm = f11_atm_fit.atm('sat')
            else:
                raise ValueError('atm_option %s not recognized.' % self.atm_option)
            self.teq = 81.3
        else:
            raise ValueError('model is neither Jupiter- nor Saturn- mass; implement a general model atmosphere option.')        
        
        self.zenv = zenv
        self.zenv_inner = zenv_inner
        
        # t = np.linspace(0, np.pi / 2, self.nz)
        # self.m = mtot * const.mjup * self.mesh_func(t) # grams
        
        # self.dm = np.diff(self.m)
        # self.kcore = kcore = int(np.where(mcore * const.mearth <= self.m)[0][0]) - 1
        # self.mcore = np.sum(self.dm[:kcore])
        # self.mcore = mcore
        
        # hitch: self.mcore not necessarily mcore specified as arg. fix by putting remainder (typically ~tenth of an earth mass) into envelope
        #
        # zenv_remainder_from_core_misfit = (mcore - self.mcore / const.mearth) / (np.sum(self.dm[kcore:]) / const.mearth)
        # print 'specified mcore %2.3f, actual mcore %2.3f. diff %2.3f; will add Z=%1.4f to envelope.' % (mcore, self.mcore / const.mearth, mcore - self.mcore / const.mearth, zenv_remainder_from_core_misfit)
        #
        # solution a: just move the nearest mesh point to the transition to the core mass exactly
        # self.kcore = kcore = np.argmin(abs(self.m - mcore * const.mearth))
        # self.m[kcore] = mcore * const.mearth
        # self.mcore = mcore
        # 
        # solution b: add a new mesh point at mcore (total number of zones is now nz + 1).
        # problematic because structure arrays are initialized in Evolve.__init__ with length nz.
        # so initialize self.m with length < nz before adding zones such that total is nz.
        t = np.linspace(0, np.pi / 2, self.nz - 1)
        self.m = mtot * const.mjup * self.mesh_func(t) # grams
        self.kcore = kcore = np.where(self.m >= mcore * const.mearth)[0][0] # kcore - 1 is last zone with m < mcore
        self.m = np.insert(self.m, self.kcore, mcore * const.mearth)
        self.mcore = mcore

        self.dm = np.diff(self.m)
        
        q = np.zeros(self.nz) # a proxy for dr
        self.grada = np.zeros(self.nz)
        
        logp_surf = np.array([7.])
        logt_surf = np.array([np.log10(t10)])
                
        # start from a solar-Y isentrope matching the specified 10-bar temperature
        entropy_including_smix = 10 ** self.hhe_eos.get_logs(logp_surf, logt_surf, 0.27)[0] * const.mp / const.kb
        # JJF isentropes leave out the entropy of mixing since they were only considering
        # a single mixture, so an additive offset was unimportant. here since we build the initial isentrope from
        # JJF tables, must subtract entropy of mixing from argument to logt_on_solar_isentrope.
        entropy_of_mixing = 10 ** self.hhe_eos.get_logsmix(logp_surf, logt_surf, 0.27)[0] * const.mp / const.kb
        entropy_guess = (entropy_including_smix - entropy_of_mixing)
        
        if np.isnan(entropy_guess):
            raise ValueError('got nan for initial entropy guess. logp, logt, y = %f, %f, %f' % (logp_surf, logt_surf, 0.27))

        if verbose:
            print 'before iterations: solar-composition adiabatic envelope'
            print 'initial guess entropy (not including smix): %f' % entropy_guess
        
        # uniform pressure in the right ballpark, for initial structure guess
        self.p[:] = 1e12
        self.t = 10 ** self.logt_on_solar_isentrope((entropy_guess, np.log10(self.p)))
                
        self.rho[:kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:kcore]), np.log10(self.t[:kcore]))
        self.rho[kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), yenv) # for first pass, no Z in envelope
        # self.rho[kcore:] = self.get_rho_xyz(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y, self.z)
        
        self.y = np.zeros_like(self.p)
        self.y[kcore:] = yenv
                
        self.mhe = np.dot(self.y[1:], self.dm) # initial total he mass. must conserve when later adjusting Y distribution
        self.k_shell_top = 0 # until a shell is found by equilibrium_y_profile
                
        # continuity equation
        q[0] = 0.
        q[1:] = 3. * self.dm / 4 / np.pi / self.rho[1:] # something like the volume of a constant-density sphere - ask daniel
        self.r = np.cumsum(q) ** (1. / 3)
                
        # do a zeroth iteration to get reasonable (again isentropic) P-T profile before
        # going into repeated iterations. this helps to ensure that EOS quantities will
        # be lookupable for all P, T points we encounter. (e.g. better than guessing 1 Mbar
        # for all layers when surface is only 200 K -- would run off eos tables.)
        
        dp = np.zeros_like(self.p)
        dp[1:] = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4
        
        self.p[:] = np.cumsum(dp[::-1])[::-1] + 10 * 1e6 # hydrostatic balance
        self.t = 10 ** self.logt_on_solar_isentrope((entropy_guess, np.log10(self.p)))
        self.t[:kcore] = self.t[kcore] # isothermal at temperature of core-mantle boundary
        
        # before look up density, set composition (z) information
        self.z[:kcore] = 1.
        assert zenv >= 0., 'got negative zenv %g' % zenv
        self.z[kcore:] = zenv
        
        # identify molecular-metallic transition. tentatively 1 Mbar; might realistically be 0.8 to 2 Mbar.
        ktrans = np.where(self.p >= 1e12)[0][-1]

        if zenv_inner: # two-layer envelope in terms of Z distribution. zenv is z of the outer envelope, zenv_inner is z of the inner envelope
            assert zenv_inner > 0, 'if you want a z-free envelope, no need to specify zenv_inner.'
            assert zenv_inner >= zenv, 'no z inversion allowed for now.'
            self.z[kcore:ktrans] = zenv_inner

        self.rho[:kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:kcore]), np.log10(self.t[:kcore])) # core
        if self.z[-1] == 0.: # this check is only valid if envelope is homogeneous in Z
            self.rho[kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:]) # XY envelope
        else:
            self.rho[kcore:] = self.get_rho_xyz(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:], self.z[kcore:]) # XYZ envelope            
            
        # these used to be defined after iterations were completed, but need for calculation of brunt_b to allow superadiabatic
        # regions with grad-grada proportional to brunt_b
        self.gradt = np.zeros_like(self.p)
        self.brunt_b = np.zeros_like(self.p)
        self.chirho = np.zeros_like(self.p)
        self.chit = np.zeros_like(self.p)
        self.chiy = np.zeros_like(self.p)
        self.dlogy_dlogp = np.zeros_like(self.p)        
        
        # relax to hydrostatic
        oldRadii = (0, 0, 0)
        for iteration in xrange(self.max_iters_for_static_model):
            self.iters = iteration + 1
            # hydrostatic equilibrium
            dp[1:] = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4.
            self.p[:] = np.cumsum(dp[::-1])[::-1] + 10 * 1e6 # hydrostatic balance
            
            # identify molecular-metallic transition. tentatively 1 Mbar; might realistically be 0.8 to 2 Mbar.
            ktrans = np.where(self.p >= 1e12)[0][-1]

            if zenv_inner: # two-layer envelope in terms of Z distribution. zenv is z of the outer envelope, zenv_inner is z of the inner envelope
                assert zenv_inner > 0, 'if you want a z-free envelope, no need to specify zenv_inner.'
                assert zenv_inner >= zenv, 'no z inversion allowed for now.'
                self.z[kcore:ktrans] = zenv_inner
                self.z[ktrans:] = zenv
            
            # compute temperature profile by integrating grad_ad from surface
            self.grada[:kcore] = 0.
            self.grada[kcore:] = self.hhe_eos.get_grada(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])
            self.gradt = np.copy(self.grada) # may be modified later if include_he_immiscibility and rrho_where_have_helium_gradient

            # a nan might appear in grada if a p, t point is just outside the original tables.
            # e.g., this was happening at logp, logt = 11.4015234804 3.61913879612, just under
            # the right-hand side "knee" of available data.
            if np.any(np.isnan(self.grada)):
                print '%i nans in grada' % len(self.grada[np.isnan(self.grada)])
                
                with open('output/grada_nans.dat', 'w') as fw:
                    for k, val in enumerate(self.grada):
                        if np.isnan(val):
                            fw.write('%16.8f %16.8f %16.8f\n' % (np.log10(self.p[k]), np.log10(self.t[k]), self.y[k]))
                print 'saved problematic logp, logt, y to output/grada_nans.dat'
                raise ValueError('eos limits')
                # assert False, 'nans in eos quantities; cannot continue.'

            self.dlogp = -1. * np.diff(np.log10(self.p))
            for k in np.arange(self.nz-2, kcore-1, -1):
                logt = np.log10(self.t[k+1]) + self.grada[k+1] * self.dlogp[k]
                self.t[k] = 10 ** logt
                
            self.t[:kcore] = self.t[kcore] # core is isothermal at core-mantle boundary temperature

            # update density from P, T to update hydrostatic r
            self.rho[:kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:kcore]), np.log10(self.t[:kcore])) # core
            if self.z[-1] == 0.: # again, this check is only valid if envelope is homogeneous in Z
                self.rho[kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:]) # XY envelope
            else:
                self.rho[kcore:] = self.get_rho_xyz(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:], self.z[kcore:]) # XYZ envelope
            
            if np.any(np.isnan(self.rho)):
                print 'have one or more nans in rho after eos call'
                return (np.nan, np.nan)
            
            q[0] = 0.
            q[1:] = 3. * self.dm / 4 / np.pi / self.rho[1:]
            self.r = np.cumsum(q) ** (1. / 3)
            # if verbose: print iteration, self.r[-1]
            if np.all(np.abs(np.mean((oldRadii / self.r[-1] - 1.))) < self.relative_radius_tolerance) and iteration >= self.min_iters_for_static_model:
                break
            if not np.isfinite(self.r[-1]):
                print 'found infinite total radius'
                return (np.nan, np.nan)
            oldRadii = (oldRadii[1], oldRadii[2], self.r[-1])

            # if want to dump info during iterations
            # print 'iteration %i: R = %f' % (self.iters, self.r[-1])
            #
            # with open('iter%02i.dat' % self.iters, 'w') as fw:
            #     for pval, tval, rhoval in zip(self.p, self.t, self.rho):
            #         fw.write('%16.8f %16.8f %16.8f\n' % (pval, tval, rhoval))
            
        else:
            return -1
                        
        if verbose: print 'converged homogeneous model after %i iterations.' % self.iters

        if include_he_immiscibility: # repeat iterations, now including the phase diagram calculation
            oldRadii = (0, 0, 0)
            for iteration in xrange(self.max_iters_for_static_model):
                self.iters_immiscibility = iteration + 1
                # hydrostatic equilibrium
                dp[1:] = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4.
                self.p[:] = np.cumsum(dp[::-1])[::-1] + 10 * 1e6 # hydrostatic balance
            
                # compute temperature profile by integrating gradt from surface
                self.grada[:kcore] = 0. # can compute after model is converged, since we take core to be isothermal
                self.grada[kcore:] = self.hhe_eos.get_grada(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:]) # last time grada is set for envelope

                if np.any(np.isnan(self.grada)):
                    print '%i nans in grada' % len(self.grada[np.isnan(self.grada)])
                
                    with open('output/grada_nans.dat', 'w') as fw:
                        for k, val in enumerate(self.grada):
                            if np.isnan(val):
                                fw.write('%16.8f %16.8f %16.8f\n' % (np.log10(self.p[k]), np.log10(self.t[k]), self.y[k]))
                    print 'saved problematic logp, logt, y to output/grada_nans.dat'
                    raise ValueError('eos limits')

                self.gradt = np.copy(self.grada)
                
                if rrho_where_have_helium_gradient and self.have_rainout:
                    # allow helium gradient regions to have superadiabatic temperature stratification.
                    # this is new here -- copied from below where we'd ordinarily only compute this after we have a converged model.
                    # might cost some time because of the additional eos calls.

                    self.chit[kcore:] = self.hhe_eos.get_chit(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])
                    self.chirho[kcore:] = self.hhe_eos.get_chirho(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])
                    self.chiy[kcore:] = self.hhe_eos.get_chiy(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])
                    self.dlogy_dlogp[1:] = np.diff(np.log(self.y)) / np.diff(np.log(self.p)) # actual log rate of change of Y with P
                    self.brunt_b[kcore+1:] = self.chirho[kcore+1:] / self.chit[kcore+1:] * self.chiy[kcore+1:] * self.dlogy_dlogp[kcore+1:]
                    self.brunt_b[self.k_shell_top + 1] = 0.
                    # print 'in internal brunt_b calculation: self.k_shell_top = %i; brunt_b[self.k_shell_top] == %g' % (self.k_shell_top, self.brunt_b[self.k_shell_top+1])
                    assert np.all(self.brunt_b[kcore+1:self.k_shell_top+1] == 0) # he shell itself should be uniform 
                    
                    # rho_this_pt_next_comp = np.zeros_like(self.p)
                    # rho_this_pt_next_comp[self.kcore+1:] = self.get_rho_xyz(np.log10(self.p[self.kcore+1:]), np.log10(self.t[self.kcore+1:]), self.y[self.kcore:-1], self.z[self.kcore:-1])
                    # # for the purposes of getting the T profile, ignore the density discontinuities at top of helium shell and top of core, since we don't want T to jump there
                    # rho_this_pt_next_comp[self.kcore + 1] = self.rho[self.kcore + 1]
                    # if self.k_shell_top > 0:
                    #     rho_this_pt_next_comp[self.k_shell_top+1] = self.rho[self.k_shell_top+1]
                    #
                    # self.brunt_b[self.kcore:-1] = (np.log(rho_this_pt_next_comp[self.kcore:-1]) - np.log(self.rho[self.kcore:-1])) / (np.log(self.rho[self.kcore:-1]) - np.log(self.rho[self.kcore+1:])) / self.chit[self.kcore:-1]
                    #
                    # n_negative_brunt_b = len(self.brunt_b[self.brunt_b < 0])
                    # if n_negative_brunt_b > 0:
                    #     print 'iteration %i.%i: zeroing out %i negative values in brunt_b' % (self.iters, self.iters_immiscibility, n_negative_brunt_b)
                    #     self.brunt_b[self.brunt_b < 0] = 0. # ignore composition inversions for purposes of getting T profile. shouldn't be necessary, but might help during iterations
                    #     print '\t %i positive values remain after pruning b<0' % len(self.brunt_b[self.brunt_b > 0])
                    #
                    # # core, He layer, and uniform envelope should all be homogeneous.
                    # # this should not be necessary! because brunt_b is proportional to rho_this_pt_next_comp - rho,
                    # # and at constant composition the two are equal, brunt_b should be zero in homogeneous regions by construction.
                    # self.brunt_b[:self.kcore] = 0.
                    # self.brunt_b[self.k_gradient_top+1:] = 0.
                    # if self.k_shell_top > 0:
                    #     self.brunt_b[self.kcore:self.k_shell_top + 1] = 0.
                    #
                    # if n_negative_brunt_b > 0:
                    #     print '\t %i positive values remain after pruning zones that should be homogeneous' % len(self.brunt_b[self.brunt_b > 0])
                    #     print '\t self.k_shell_top is %i' % self.k_shell_top
                    #
                    # if n_negative_brunt_b == 0:
                    #     print 'iteration %i.%i: all good' % (self.iters, self.iters_immiscibility)
                    #     print '\t self.k_shell_top is %i' % self.k_shell_top
                    
                    self.gradt += rrho_where_have_helium_gradient * self.brunt_b
                    # self.gradt[kcore:] = self.grada[kcore:] + rrho_where_have_helium_gradient * self.brunt_b[kcore:]
                                                
                # print 'integrating for T profile'
                self.dlogp = -1. * np.diff(np.log10(self.p))
                for k in np.arange(self.nz-2, self.kcore-1, -1): # surface to center
                    logt = np.log10(self.t[k+1]) + self.gradt[k+1] * self.dlogp[k]
                    # print '\t%i %f %f %f' % (k, logt, self.gradt[k+1], self.brunt_b[k+1])
                    if np.isinf(logt):
                        print 'k', k
                        print 'gradt', self.gradt[k+1]
                        print 'logt', logt
                        print 'brunt_b', self.brunt_b[k]
                        raise RuntimeError('got infinite temperature in integration of gradT.')
                    self.t[k] = 10 ** logt

                self.y = self.equilibrium_y_profile(phase_t_offset, minimum_y_is_envelope_y=minimum_y_is_envelope_y)

                # update density from P, T to update hydrostatic r
                self.rho[:kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:kcore]), np.log10(self.t[:kcore])) # core
                if zenv == 0.:
                    self.rho[kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:]) # XY envelope
                else:
                    self.rho[kcore:] = self.get_rho_xyz(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:], self.z[kcore:]) # XYZ envelope
            
                q[0] = 0.
                q[1:] = 3. * self.dm / 4 / np.pi / self.rho[1:]
                self.r = np.cumsum(q) ** (1. / 3)
                if verbose: print iteration, self.r[-1]

                if np.any(np.isnan(self.rho)):
                    with open('output/found_nans_eos.dat', 'w') as f:
                        f.write('%12s %12s %12s %12s %12s %12s %12s %12s %12s\n' % ('core', 'p', 't', 'rho', 'm', 'r', 'y', 'brunt_B', 'gradt-grada'))
                        for k in xrange(self.nz):
                            in_core = k < self.kcore
                            f.write('%12s %12g %12g %12g %12g %12g %12g %12g %12g\n' % (in_core, self.p[k], self.t[k], self.rho[k], self.m[k], self.r[k], self.y[k], self.brunt_b[k], self.gradt[k]-self.grada[k]))
                    print 'saved output/found_nans_eos.dat'
                    raise RuntimeError('found nans in rho on staticModel iteration %i' % self.iters)

                if np.all(np.abs(np.mean((oldRadii / self.r[-1] - 1.))) < self.relative_radius_tolerance):
                    break

                if not np.isfinite(self.r[-1]):
                    with open('output/found_infinite_radius.dat', 'w') as f:
                        f.write('%12s %12s %12s %12s %12s %12s\n' % ('core', 'p', 't', 'rho', 'm', 'r'))
                        for k in xrange(self.nz):
                            in_core = k < self.kcore
                            f.write('%12s %12g %12g %12g %12g %12g\n' % (in_core, self.p[k], self.t[k], self.rho[k], self.m[k], self.r[k]))
                    print 'saved output/found_infinite_radius.dat'
                    raise RuntimeError('found infinite total radius')
                oldRadii = (oldRadii[1], oldRadii[2], self.r[-1])
            else:
                return -1
            
            if verbose: print 'converged with new Y profile after %i iterations.' % self.iters_immiscibility

        # finalize hydrostatic profiles, and calculate lots of auxiliary quantities of interest.
        dp[1:] = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4.
        self.p[:] = np.cumsum(dp[::-1])[::-1] + 10 * 1e6
        self.p[0] *= (1. + 1e-10)
        # update density for the last time
        self.rho[:kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:kcore]), np.log10(self.t[:kcore])) # core
        if zenv == 0.:
            self.rho[kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:]) # XY envelope
        else:
            self.rho[kcore:] = self.get_rho_xyz(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:], self.z[kcore:]) # XYZ envelope
        
        self.rtot = self.r[-1]
        self.mtot = self.m[-1]

        # # extrapolate in logp to get a 10-bar temperature using a cubic spline.
        # npts_get_t10 = 10
        # logp_near_surf = np.log10(self.p[-npts_get_t10:][::-1])
        # t_near_surf = self.t[-npts_get_t10:][::-1]
        # t_surf_spline = splrep(logp_near_surf, t_near_surf, s=0, k=3)
        # self.t10 = splev(7., t_surf_spline, der=0, ext=0) # ext=0 for extrapolate
        
        # better to integrate grada instead. 20 points takes around 50 ms done this way.
        npts_integrate = 20
        logp_ = np.linspace(np.log10(self.p[-1]), 7, npts_integrate)
        logt_ = np.zeros_like(logp_)
        logt_[0] = np.log10(self.t[-1])
        for j, logp in enumerate(logp_):
            if j == 0: continue
            grada = self.hhe_eos.get_grada(logp_[j-1], logt_[j-1], self.y[-1])
            dlogp = logp_[j] - logp_[j-1]
            logt_[j] = logt_[j-1] + grada * dlogp
        self.t10 = 10 ** logt_[-1]
        
        # april 24: try integrating all the way down to 1 bar, starting from the t10 just obtained
        npts_integrate = 20
        logp_ = np.linspace(7., 6., npts_integrate)
        logt_ = np.zeros_like(logp_)
        logt_[0] = np.log10(self.t10)
        for j, logp in enumerate(logp_):
            if j == 0: continue
            grada = self.hhe_eos.get_grada(logp_[j-1], logt_[j-1], self.y[-1])
            dlogp = logp_[j] - logp_[j-1]
            logt_[j] = logt_[j-1] + grada * dlogp
        self.t1 = 10 ** logt_[-1]
        
        
        assert self.t10 > 0., 'surface integration yielded a negative t10 %g' % self.t10
        
        self.surface_g = const.cgrav * mtot * const.mjup / self.r[-1] ** 2
        try:
            self.tint = atm.get_tint(self.surface_g * 1e-2, self.t10) # Fortney+2011 needs g in mks
            self.teff = (self.tint ** 4 + self.teq ** 4) ** (1. / 4)
            self.lint = 4. * np.pi * self.rtot ** 2 * const.sigma_sb * self.tint ** 4
        except ValueError:
            print 'failed to get tint for converged model. g_mks = %f, t10 = %f' % (self.surface_g * 1e-2, self.t10)
            print 'surface p, t', self.p[-1], self.t[-1]
            raise ValueError('atm limits')
            self.tint = 1e20
            self.teff = 1e20
            self.lint = 1e50
        self.entropy = np.zeros_like(self.p)
        # self.entropy[:kcore] = 10 ** self.z_eos.get_logs(np.log10(self.p[:kcore]), np.log10(self.t[:kcore])) * const.mp / const.kb # leave this out, unsure of units on aneos entropy
        # note -- in the envelope, ignoring any contribution that heavies make to the entropy.
        self.entropy[kcore:] = 10 ** self.hhe_eos.get_logs(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:]) * const.mp / const.kb

        # core is isothermal at temperature of the base of the envelope
        self.t[:kcore] = self.t[kcore]
        
        self.g = const.cgrav * self.m / self.r ** 2

        # this is a structure derivative, not a thermodynamic one.
        # wherever the profile is a perfect adiabat, this is also gamma1.
        self.dlogp_dlogrho = np.diff(np.log(self.p)) / np.diff(np.log(self.rho))

        self.gamma1 = np.zeros_like(self.p)
        self.csound = np.zeros_like(self.p)
        self.gradt_direct = np.zeros_like(self.p)
        
        self.r[0] = 1. # 1 cm central radius to keep these things calculable at center zone
        self.gamma1[:kcore] = self.z_eos.get_gamma1(np.log10(self.p[:kcore]), np.log10(self.t[:kcore]))
        self.gamma1[kcore:] = self.hhe_eos.get_gamma1(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])
        self.csound = np.sqrt(self.p / self.rho * self.gamma1)
        self.lamb_s12 = 2. * self.csound ** 2 / self.r ** 2 # lamb freq. squared for l=1
        
        self.delta_nu = (2. * trapz(self.csound ** -1, x=self.r)) ** -1 * 1e6 # the large frequency separation in uHz
        self.delta_nu_env = (2. * trapz(self.csound[kcore:] ** -1, x=self.r[kcore:])) ** -1 * 1e6

        dlnp_dlnr = np.diff(np.log(self.p)) / np.diff(np.log(self.r))
        dlnrho_dlnr = np.diff(np.log(self.rho)) / np.diff(np.log(self.r))
        
        try:
            self.chirho[:kcore] = self.z_eos.get_chirho(np.log10(self.p[:kcore]), np.log10(self.t[:kcore]))
            self.chit[:kcore] = self.z_eos.get_chit(np.log10(self.p[:kcore]), np.log10(self.t[:kcore]))
            self.grada[:kcore] = (1. - self.chirho[:kcore] / self.gamma1[:kcore]) / self.chit[:kcore] # e.g., Unno's equations 13.85, 13.86
        except AttributeError:
            print "warning: z eos option '%s' does not provide methods for get_chirho and get_chit." % self.z_eos_option
            print 'cannot calculate things like grada in core and so this model may not be suited for eigenmode calculations.'
            pass
        
        # ignoring the envelope z component when it comes to calculating chirho and chit
        self.chirho[kcore:] = self.hhe_eos.get_chirho(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])
        self.chit[kcore:] = self.hhe_eos.get_chit(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])
        self.gradt_direct[:kcore] = 0. # was previously self.gradt
        self.gradt_direct[kcore+1:] = np.diff(np.log(self.t[kcore:])) / np.diff(np.log(self.p[kcore:])) # was previously self.gradt
        
        self.brunt_n2_direct = np.zeros_like(self.p)
        self.brunt_n2_direct[1:] = self.g[1:] / self.r[1:] * (dlnp_dlnr / self.gamma1[1:] - dlnrho_dlnr)
        
        # other forms of the BV frequency
        self.homology_v = const.cgrav * self.m * self.rho / self.r / self.p
        self.brunt_n2_unno_direct = np.zeros_like(self.p)
        self.brunt_n2_unno_direct[kcore+1:] = self.g[kcore+1:] * self.homology_v[kcore+1:] / self.r[kcore+1:] * \
            (self.dlogp_dlogrho[kcore:] ** -1. - self.gamma1[kcore+1:] ** -1.) # Unno 13.102
        
        # terms needed for calculation of the composition term brunt_B.
        self.dlogy_dlogp[1:] = np.diff(np.log(self.y)) / np.diff(np.log(self.p)) # structure derivative
        # this is the thermodynamic derivative (const P and T), for H-He.
        self.dlogrho_dlogy = np.zeros_like(self.p)
        self.dlogrho_dlogy[kcore:] = self.hhe_eos.get_dlogrho_dlogy(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])
        
        rho_z = np.zeros_like(self.p)
        rho_hhe = np.zeros_like(self.p)
        # rho_z[self.z > 0.] = 10 ** self.z_eos.get_logrho(np.log10(self.p[self.z > 0.]), np.log10(self.t[self.z > 0.]))
        rho_z[self.z > 0.] = self.get_rho_z(np.log10(self.p[self.z > 0.]), np.log10(self.t[self.z > 0.]))
        rho_hhe[self.y > 0.] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[self.y > 0.]), np.log10(self.t[self.y > 0.]), self.y[self.y > 0.])
        self.dlogrho_dlogz = np.zeros_like(self.p)
        # dlogrho_dlogz is only calculable where all of X, Y, and Z are non-zero.
        self.dlogrho_dlogz[self.z * self.y > 0.] = -1. * self.rho[self.z * self.y > 0.] * self.z[self.z * self.y > 0.] * (rho_z[self.z * self.y > 0.] ** -1 - rho_hhe[self.z * self.y > 0.] ** -1)
        self.dlogz_dlogp = np.zeros_like(self.p)
        self.dlogz_dlogp[1:] = np.diff(np.log(self.z)) / np.diff(np.log(self.p))
        
        
        # this is the form of brunt_n2 that makes use of grad, grada, and the composition term brunt B.
        self.brunt_n2_unno = np.zeros_like(self.p)
        # in core, explicitly ignore the Y gradient.
        self.brunt_n2_unno[:kcore+1] = self.g[:kcore+1] * self.homology_v[:kcore+1] / self.r[:kcore+1] * \
            (self.chit[:kcore+1] / self.chirho[:kcore+1] * (self.grada[:kcore+1] - self.gradt[:kcore+1]) + \
            self.dlogrho_dlogz[:kcore+1] * self.dlogz_dlogp[:kcore+1])
        # in envelope, Y gradient is crucial.
        self.brunt_n2_unno[kcore:] = self.g[kcore:] * self.homology_v[kcore:] / self.r[kcore:] * \
            (self.chit[kcore:] / self.chirho[kcore:] * (self.grada[kcore:] - self.gradt[kcore:]) + \
            self.dlogrho_dlogy[kcore:] * self.dlogy_dlogp[kcore:])
            
        # akin to mike montgomery's form for brunt_B, which is how mesa does it by default (Paxton+2013)
        rho_this_pt_next_comp = np.zeros_like(self.p)
        rho_this_pt_next_comp[self.kcore+1:] = self.get_rho_xyz(np.log10(self.p[self.kcore+1:]), np.log10(self.t[self.kcore+1:]), self.y[self.kcore:-1], self.z[self.kcore:-1])
        # core-mantle point must be treated separately since next cell down is pure Z, and get_rho_xyz is not designed for pure Z. 
        # call z_eos.get_logrho directly instead.
        rho_this_pt_next_comp[self.kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[self.kcore]), np.log10(self.t[self.kcore]))
        # within core, composition is assumed constant so rho_this_pt_next_comp is identical to rho.
        if erase_z_discontinuity_from_brunt:
            rho_this_pt_next_comp[:self.kcore+1] = self.rho[:self.kcore+1]
        else:
            rho_this_pt_next_comp[:self.kcore] = self.rho[:self.kcore]            

        # ignore Y discontinuity at surface of he-rich layer
        # if self.k_shell_top > 0:
            # rho_this_pt_next_comp[self.k_shell_top + 1] = self.rho[self.k_shell_top + 1]
        
        self.brunt_b_mhm = np.zeros_like(self.p)
        self.brunt_b_mhm[:-1] = (np.log(rho_this_pt_next_comp[:-1]) - np.log(self.rho[:-1])) / (np.log(self.rho[:-1]) - np.log(self.rho[1:])) / self.chit[:-1]
        self.brunt_n2_mhm = self.g ** 2 * self.rho / self.p * self.chit / self.chirho * (self.grada - self.gradt + self.brunt_b_mhm)
        self.brunt_n2_mhm[0] = 0. # had nan previously, probably from brunt_b
        
        self.brunt_b = self.brunt_b_mhm
        self.brunt_n2 = self.brunt_n2_mhm
                
        # this is the thermo derivative rho_t in scvh parlance. necessary for gyre, which calls this minus delta.
        # dlogrho_dlogt_const_p = chit / chirho = -delta = -rho_t
        self.dlogrho_dlogt_const_p = np.zeros_like(self.p)
        self.dlogrho_dlogt_const_p[:kcore] = self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[:kcore]), np.log10(self.t[:kcore]))
        if self.z_eos_option == 'reos water' and self.t[-1] < 1e3: # must be calculated separately for low T and high T part of the envelope
            k_t_boundary = np.where(np.log10(self.t) > 3.)[0][-1]
            self.dlogrho_dlogt_const_p[kcore:k_t_boundary+1] = self.rho[kcore:k_t_boundary+1] * \
                (self.z[kcore:k_t_boundary+1] / rho_z[kcore:k_t_boundary+1] * self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[kcore:k_t_boundary+1]), np.log10(self.t[kcore:k_t_boundary+1])) + \
                (1. - self.z[kcore:k_t_boundary+1]) / rho_hhe[kcore:k_t_boundary+1] * self.hhe_eos.get_rhot(np.log10(self.p[kcore:k_t_boundary+1]), np.log10(self.t[kcore:k_t_boundary+1]), self.y[kcore:k_t_boundary+1]))
            self.dlogrho_dlogt_const_p[k_t_boundary+1:] = self.rho[k_t_boundary+1:] * (self.z[k_t_boundary+1:] / rho_z[k_t_boundary+1:] * \
                self.z_eos_low_t.get_dlogrho_dlogt_const_p(np.log10(self.p[k_t_boundary+1:]), np.log10(self.t[k_t_boundary+1:])) + \
                (1. - self.z[k_t_boundary+1:]) / rho_hhe[k_t_boundary+1:] * self.hhe_eos.get_rhot(np.log10(self.p[k_t_boundary+1:]), np.log10(self.t[k_t_boundary+1:]), self.y[k_t_boundary+1:]))

        else: # no need to sweat low vs. high t
            self.dlogrho_dlogt_const_p[kcore:] = self.rho[kcore:] * (self.z[kcore:] / rho_z[kcore:] * self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[kcore:]), np.log10(self.t[kcore:])) + (1. - self.z[kcore:]) / rho_hhe[kcore:] * self.hhe_eos.get_rhot(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:]))
            
        self.mf = self.m / self.mtot
        self.rf = self.r / self.rtot  
        
        # this is written assuming two-layer constant Z: Z uniform in envelope, Z=1 in core. 
        self.mz_env = self.z[-1] * (self.mtot - self.mcore * const.mearth)
        self.mz = self.mz_env + self.mcore * const.mearth
        self.bulk_z = self.mz / self.mtot
        
        self.ysurf = self.y[-1]
        
        # axial moment of inertia, in units of mtot * rtot ** 2. moi of a thin spherical shell is 2 / 3 * m * r ** 2
        self.nmoi = 2. / 3 * trapz(self.r ** 2, x=self.m) / self.mtot / self.rtot ** 2
                        
        return self.iters
                
    def run(self,mtot=1., yenv=0.27, zenv=0., mcore=10., starting_t10=3e3, min_t10=200., nsteps=100, stdout_interval=1,
                include_he_immiscibility=False, phase_t_offset=0., output_prefix=None, minimum_y_is_envelope_y=False, rrho_where_have_helium_gradient=None):
        import time
        assert 0. <= mcore*const.mearth <= mtot*const.mjup,\
            "invalid core mass %f for total mass %f" % (mcore, mtot * const.mjup / const.mearth)
        menv = mtot * const.mjup - mcore * const.mearth
        assert 0. <= zenv <= 1., 'invalid envelope z %f' % zenv

        target_t10s = np.logspace(np.log10(min_t10), np.log10(starting_t10), nsteps)[::-1]

        self.history = {}
        self.history_columns = 'step', 'age', 'radius', 'tint', 't10', 'teff', 'ysurf', 'lint', 'nz_gradient', 'nz_shell', 'iters', 'mz_env', 'mz', 'bulk_z'
        for name in self.history_columns:
            self.history[name] = np.zeros_like(target_t10s)

        keep_going = True
        previous_entropy = 0
        age_gyr = 0
        # these columns are for the realtime (e.g., notebook) output
        columns = 'step', 'iters', 'tgt_t10', 't10', 'teff', 'radius', 's_mean', 'dt_yr', 'age_gyr', 'nz_gradient', 'nz_shell', 'y_surf', 'walltime'
        start_time = time.time()
        print ('%12s ' * len(columns)) % columns
        for step, target_t10 in enumerate(target_t10s):
            self.staticModel(mtot=mtot, t10=target_t10, yenv=yenv, zenv=zenv, mcore=mcore, 
                                include_he_immiscibility=include_he_immiscibility,
                                phase_t_offset=phase_t_offset, 
                                minimum_y_is_envelope_y=minimum_y_is_envelope_y, 
                                rrho_where_have_helium_gradient=rrho_where_have_helium_gradient)
            walltime = time.time() - start_time
            if self.tint > 1e9:
                print 'failed in atmospheric boundary condition. stopping.'
                if output_prefix:
                    with open('output/%s.history' % output_prefix, 'wb') as fw:
                        pickle.dump(self.history, fw, 0)
                    print 'wrote history data to output/%s.history' % output_prefix
                return self.history
            dt_yr = 0.
            if step > 0:
                delta_s = self.entropy - previous_entropy
                delta_s *= const.kb / const.mp # now erg K^-1 g^-1
                assert self.lint > 0, 'found negative intrinsic luminosity.'
                int_tdsdm = trapz(self.t * delta_s, dx=self.dm)
                dt = -1. *  int_tdsdm / self.lint
                if dt < 0:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
                    ax[0].plot(self.mf, self.y - self.ystart, 'k-', lw=1)
                    ax[0].set_ylabel(r'$dY$')
                    integrand = self.t * delta_s
                    ax[1].plot(self.mf, delta_s / max(abs(delta_s)), '.', label=r'$\Delta s$')
                    ax[1].plot(self.mf, integrand / max(abs(integrand)), '.', label=r'$T\,\Delta s$')
                    partial_integral = -1. * np.cumsum(self.t[:-1] * delta_s[:-1] * self.dm)
                    ax[1].plot(self.mf[:-1], partial_integral / max(abs(partial_integral)), '-', lw=1, label=r'$\propto \delta t$')
                    ax[1].legend(fontsize=14, loc='best')
                    print 'error: integral of t*(delta_s)*dm = %g, but should be negative.' % int_tdsdm
                    print 'mean entropy %f' % np.mean(self.entropy[self.kcore:])
                    print 'tint, lint', self.tint, self.lint
                    print '%12i %12i %12.3f %12.3f %12.3f %12.3e %12.3f %12.3e %12.3f %12i %12i %12.3f %12.3f' % \
                        (step, self.iters, target_t10, self.t10, self.teff, self.rtot, np.mean(self.entropy[self.kcore:]), dt_yr, age_gyr, self.nz_gradient, self.nz_shell, self.y[-1], walltime)
                    keep_going = False
                dt_yr = dt / const.secyear
                age_gyr += dt_yr * 1e-9

                self.history['step'][step] = step
                self.history['iters'][step] = self.iters
                self.history['age'][step] = age_gyr
                self.history['radius'][step] = self.rtot
                self.history['tint'][step] = self.tint
                self.history['lint'][step] = self.lint
                self.history['t10'][step] = self.t10
                self.history['teff'][step] = self.teff
                self.history['ysurf'][step] = self.y[-1]
                self.history['nz_gradient'][step] = self.nz_gradient
                self.history['nz_shell'][step] = self.nz_shell
                self.history['mz_env'][step] = self.mz_env
                self.history['mz'][step] = self.mz
                self.history['bulk_z'][step] = self.bulk_z
                
            # history and profile quantities go in dictionaries which are pickled
            if output_prefix:
                assert type(output_prefix) is str, 'output_prefix needs to be a string.'
                self.profile = {}
                
                # profile arrays
                self.profile['p'] = self.p
                self.profile['t'] = self.t
                self.profile['rho'] = self.rho
                self.profile['y'] = self.y
                try:
                    self.profile['dy'] = self.y - self.ystart
                except AttributeError: # no phase separation for this step
                    pass
                self.profile['entropy'] = self.entropy
                self.profile['r'] = self.r
                self.profile['m'] = self.m
                self.profile['g'] = self.g
                self.profile['dlogp_dlogrho'] = self.dlogp_dlogrho
                self.profile['gamma1'] = self.gamma1
                self.profile['csound'] = self.csound
                self.profile['lamb_s12'] = self.lamb_s12
                self.profile['brunt_n2'] = self.brunt_n2
                self.profile['brunt_n2_direct'] = self.brunt_n2_direct
                self.profile['chirho'] = self.chirho
                self.profile['chit'] = self.chit
                self.profile['gradt'] = self.gradt
                self.profile['grada'] = self.grada
                self.profile['rf'] = self.rf
                self.profile['mf'] = self.mf
                
                # profile scalars
                self.profile['step'] = step
                self.profile['age'] = age_gyr
                self.profile['nz'] = self.nz
                self.profile['kcore'] = self.kcore
                self.profile['radius'] = self.rtot
                self.profile['tint'] = self.tint
                self.profile['lint'] = self.lint
                self.profile['t10'] = self.t10
                self.profile['t1'] = self.t1
                self.profile['teff'] = self.teff
                self.profile['ysurf'] = self.y[-1]
                self.profile['nz_gradient'] = self.nz_gradient
                self.profile['nz_shell'] = self.nz_shell
                self.profile['mz_env'] = self.mz_env
                self.profile['mz'] = self.mz
                self.profile['bulk_z'] = self.bulk_z
                

                with open('output/%s%i.profile' % (output_prefix, step), 'w') as f:
                    pickle.dump(self.profile, f, 0) # 0 means dump as text
                                        
                if not keep_going: 
                    print 'stopping.'
                    raise
                
            if step % stdout_interval == 0 or step == nsteps - 1: 
                print '%12i %12i %12.3f %12.3f %12.3f %12.3e %12.3f %12.3e %12.3f %12i %12i %12.3f %12.3f' % (step, self.iters, target_t10, self.t10, self.teff, self.rtot, np.mean(self.entropy[self.kcore:]), dt_yr, age_gyr, self.nz_gradient, self.nz_shell, self.y[-1], walltime)

            previous_entropy = self.entropy
            
        if output_prefix:
            with open('output/%s.history' % output_prefix, 'wb') as fw:
                pickle.dump(self.history, fw, 0)
            print 'wrote history data to output/%s.history' % output_prefix
            
        return self.history
        
    def smooth(self, array, std, type='flat'):
        width = 10 * std
        from scipy.signal import gaussian
        weights = gaussian(width, std=std)
        res = np.zeros_like(array)
        for k in np.arange(len(array)):
            window = array[k-width/2:k+width/2]
            try:
                res[k] = np.dot(weights, window) / np.sum(weights)
            except:
                res[k] = array[k]
        return res
        
    def basic_profile(self, save_prefix=None):
        import os
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'hspace':0.1})
        ax[0].loglog(self.p, self.t); ax[0].set_ylabel(r'$T$')
        ax[1].loglog(self.p, self.rho); ax[1].set_ylabel(r'$\rho$')
        ax[-1].set_xlabel(r'$P$')
        if save_prefix:
            outdir = '%s/figs' % save_prefix
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            plt.savefig('%s/ptrho.pdf' % outdir, bbox_inches='tight')
        fig, ax = plt.subplots(3, 1, figsize=(6, 8), sharex=True, gridspec_kw={'hspace':0.1})
        ax[0].plot(self.rf, self.rho); ax[0].set_ylabel(r'$\rho$')
        ax[1].plot(self.rf, self.y); ax[1].set_ylabel(r'$Y$')
        ax[2].semilogy(self.rf, self.brunt_n2)
        ax[2].semilogy(self.rf, self.lamb_s12)
        ax[2].set_ylabel(r'$N^2,\ \ S_{\ell=1}^2$')
        ax[2].set_ylim(1e-10, 1e-3)
        ax[-1].set_xlabel(r'$r/R$')
        if save_prefix:
            outdir = '%s/figs' % save_prefix
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            plt.savefig('%s/rho_y_prop.pdf' % outdir, bbox_inches='tight')        
        
    def save_gyre_model(self, outfile, 
                    smooth_brunt_n2_std=None, add_rigid_rotation=None, erase_y_discontinuity_from_brunt=False, erase_z_discontinuity_from_brunt=False):
        # outfile = 'output/%s%i.gyre' % (output_prefix, step)
        with open(outfile, 'w') as f:
            header_format = '%6i ' + '%19.12e '*3 + '%5i\n'
            line_format = '%6i ' + '%26.16e '*18 + '\n'
            ncols = 19
            
            # gyre doesn't like the first zone to have zero enclosed mass, so we omit the center point when writing the gyre model.
            
            f.write(header_format % (self.nz - 1, self.mtot, self.rtot, self.lint, ncols))
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
            
            brunt_n2_for_gyre_model = np.copy(self.brunt_n2)
            
            if erase_y_discontinuity_from_brunt:
                assert self.k_shell_top > 0, 'no helium-rich shell, and thus no y discontinuity to erase.'
                brunt_n2_for_gyre_model[self.k_shell_top + 1] = self.brunt_n2[self.k_shell_top + 2]
            
            if erase_z_discontinuity_from_brunt:
                assert self.kcore > 0, 'no core, and thus no z discontinuity to erase.'
                if self.kcore == 1:
                    brunt_n2_for_gyre_model[self.kcore] = 0.
                else:
                    brunt_n2_for_gyre_model[self.kcore] = self.brunt_n2[self.kcore - 1]
            
            if smooth_brunt_n2_std:
                brunt_n2_for_gyre_model = self.smooth(self.brunt_n2, smooth_brunt_n2_std, type='gaussian')
                
            if add_rigid_rotation:
                omega = add_rigid_rotation
            else:
                omega = 0.
            
            for k in np.arange(self.nz):
                if k == 0: continue
                f.write(line_format % (k, self.r[k], w[k], dummy_luminosity, self.p[k], self.t[k], self.rho[k], \
                                       self.gradt[k], brunt_n2_for_gyre_model[k], self.gamma1[k], self.grada[k], -1. * self.dlogrho_dlogt_const_p[k], \
                                       dummy_kappa, dummy_kappa_t, dummy_kappa_rho, 
                                       dummy_epsilon, dummy_epsilon_t, dummy_epsilon_rho, 
                                       omega))

            print 'wrote %i zones to %s' % (k, outfile)
            
        # also write more complete profile info
        outfile_more = outfile.replace('.gyre', '.profile')
        with open(outfile_more, 'w') as f:
                        
            # write scalars
            # these names should match attributes of the Evolver instance
            scalar_names = 'nz', 'iters', 'mtot', 'rtot', 'tint', 'lint', 't1', 't10', 'teff', 'ysurf', 'nz_gradient', 'nz_shell', 'zenv', 'zenv_inner', 'mz_env', 'mz', 'bulk_z', 'delta_nu', 'delta_nu_env'
            n_scalars = len(scalar_names)
            scalar_header_fmt = '%20s ' * n_scalars
            f.write(scalar_header_fmt % scalar_names)
            f.write('\n')
            for name in scalar_names:
                value = getattr(self, name)
                if value is None: # can't substitute None into a string. assign nonphysical numerical value -1.
                    value = -1
                f.write('%20.10g ' % value)
            f.write('\n')
            f.write('\n')
                        
            # write vectors
            vector_names = 'p', 't', 'rho', 'y', 'z', 'entropy', 'r', 'm', 'g', 'gamma1', \
                'csound', 'lamb_s12', 'brunt_n2', 'brunt_n2_direct', 'brunt_b', 'chirho', 'chit', 'gradt', 'grada', 'gradt_direct', 'rf', 'mf'
            n_vectors = len(vector_names)
            vector_header_fmt = '%20s ' * n_vectors
            f.write(vector_header_fmt % vector_names)
            f.write('\n')
            for k in np.arange(self.nz):
                for name in vector_names:
                    try:
                        f.write('%20.10g ' % getattr(self, name)[k])
                    except:
                        print k, name
                f.write('\n')
            f.write('\n')
            
            print 'wrote %i zones to %s' % (k, outfile_more)
            
                                    
