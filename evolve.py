import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d, splrep, splev
from scipy.integrate import trapz, odeint
import sys
import const
import pickle
import utils
import time

class Evolver:
    
    def __init__(self,hhe_eos_option='scvh',z_eos_option='reos water',nz=1000,relative_radius_tolerance=1e-4,max_iters_for_static_model=500, phase_t_offset=0.):
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
        # self.getRockIceDensity = interp1d(rockIceEOS['pressure'],
                                          # rockIceEOS['density']) # returns log10 density   
        import aneos; reload(aneos)
        import reos; reload(reos)
        if z_eos_option == 'reos water':
            self.z_eos = reos.eos()
        elif z_eos_option == 'aneos ice':
            self.z_eos = aneos.eos('ice')
        elif z_eos_option == 'aneos water': # aneos "water" and "ice" seem essentially indistinguishable
            self.z_eos = aneos.eos('water')
        elif z_eos_option == 'aneos iron':
            self.z_eos = aneos.eos('iron')
        elif z_eos_option == 'aneos serpentine':
            self.z_eos = aneos.eos('serpentine')
        else:
            raise ValueError('z eos option %s not recognized.' % z_eos_option)

        # h-he phase diagram
        import lorenzen; reload(lorenzen)
        self.phase = lorenzen.hhe_phase_diagram(t_offset=phase_t_offset)
                                          
        # model atmospheres -- now initialized in self.staticModel        
        # atmos = np.load('data/atmosphereGrid.npz')
        # self.getIntrinsicTemp = RegularGridInterpolator(
        #     (atmos['flux'],
        #      atmos['logGravity'],
        #      atmos['logT990']),
        #     atmos['logTint'],bounds_error=False,fill_value=None)
            
        # Initialize memory
        self.nz = self.nBins = nz
        self.workingMemory = np.zeros(self.nBins)
        
        # cm -- adding separate arrays for the structure variables, so we can access profiles
        # after a model is built.
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
        
        self.have_rainout = False
        self.have_rainout_to_core = False
        
        self.nz_gradient = 0
        self.nz_shell = 0
        
        return

    def getMixDensity_daniel(self, entropy, pressure, z):
        return 1. / ((1. - z) / 10 ** self.logrho_on_solar_isentrope((entropy,np.log10(pressure))) +
                   z / 10 ** self.getRockIceDensity((np.log10(pressure))))
        
    # when building non-isentropic models for instance, a grad(P, T, Y, Z)=grada(P, T, Y, Z) model
    # with Y, Z functions of depth so that entropy is non-uniform, want to be able to get density 
    # as a function of these same independent variables P, T, Y, Z.

    def get_rho_xyz(self, logp, logt, y, z):             
        # old: zero-temperature rock/ice aneos  
        # rhoinv = (1. - z) / 10 ** self.hhe_eos.get_logrho(logp, logt, y) + \
               # z / 10 ** self.getRockIceDensity((logp))
        try:
            rho_z = 10 ** self.z_eos.get_logrho(logp, logt)
        except ValueError:
            print 'out of bounds in z_eos.get_logrho'
            raise
        rho_hhe = 10 ** self.hhe_eos.get_logrho(logp, logt, y)
        rhoinv = (1. - z) / rho_hhe + z / rho_z
        return rhoinv ** -1

    def test(self):
        print "Static Models:"
        print 0.9276, round(self.staticModel(2,6.5,.9,100)[0]/const.rjup,4)
        print 1.1102, round(self.staticModel(10,9,.8,15)[0]/const.rjup,4)
        print 0.5458, round(self.staticModel(.1,7,.4,1)[0]/const.rjup,4)
        print 0.3893, round(self.staticModel(.069849,7,.4,15)[0]/const.rjup,4)
        print "Evolution Models:"
        print 1.2518, round(self.run(4,10,50,1e9,.02,3)[0]/const.rjup,4)
        # print 1.0638, round(self.run(2,100,0,1.3e6,0,1)[0]/const.rjup,4)
        print 1.0249, round(self.run(1,25,20,1.5e7,.01,3.1623)[0]/const.rjup,4)
        # print 0.4089, round(self.run(.1,25,3,1e8,0,10)[0]/const.rjup,4)
        print "Jonathan / Old / New:"
        print 1.010, 1.0252, round(self.run(.406,0,10,1.37e6,0,3.1623)/const.rjup,4)
        print 1.066, 1.0895, round(self.run(1.886,0,0,1.37e6,0,10)/const.rjup,4)
        return

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

    def equilibrium_y_profile(self, verbose=False, show_timing=False):
        '''uses the existing p-t profile to find the thermodynamic equilibrium y profile, which
        may require a nearly pure-helium layer atop the core.'''
        p = self.p * 1e-12 # Mbar
        k1 = np.where(p > 1.)[0][-1]
        ymax1 = self.phase.ymax_lhs(p[k1], self.t[k1])
        
        if np.isnan(ymax1) or self.y[k1] < ymax1:
            if verbose: print 'first point at P > 1 Mbar is stable to demixing. Y, Ymax', self.y[k1], ymax1
            return self.y
            
        yout = np.copy(self.y)
        yout[k1:] = ymax1 # homogeneous molecular envelope at this abundance
        if verbose: print 'demix', k1, self.m[k1] / self.m[-1], p[k1], self.t[k1], self.y[k1], '-->', yout[k1]
                
        t0 = time.time()
        rainout_to_core = False
        
        self.nz_gradient = 0
        for k in np.arange(k1-1, self.kcore, -1):
            t1 = time.time()
            ymax = self.phase.ymax_lhs(p[k], self.t[k])
            if np.isnan(ymax):
                raise ValueError('got nan from ymax_lhs in initial loop over zones. p, t = %f, %f' % (p[k], self.t[k]))
            if show_timing: print 'zone %i: dt %f ms, t0 + %f seconds' % (k, 1e3 * (time.time() - t1), time.time() - t0)
            # sys.stdout.write('.')
            if yout[k] < ymax:
                if verbose: print 'stable', k, self.m[k] / self.m[-1], p[k], self.t[k], yout[k], ' < ', ymax, -1
                break
            ystart = yout[k]
            yout[k] = ymax
            # difference between initial he mass and current proposed he mass above and including this zone
            # must be in the deeper interior.
            he_mass_missing_above = self.mhe - np.dot(yout[k:], self.dm[k-1:])
            enclosed_envelope_mass = np.sum(self.dm[self.kcore:k])
            if not enclosed_envelope_mass > 0: # at core boundary
                msg = 'gradient reaches core boundary; rainout to core.'
                print msg
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
                assert np.dot(yout[self.kcore:], self.dm[self.kcore-1:]) < self.mhe, 'problem: proposed envelope already has mhe > mhe_initial, even before he shell iterations. case: y_interior > 1'
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
                tentative_total_he_mass = np.dot(yout[self.kcore:], self.dm[self.kcore-1:])
                if verbose: print '%5i %5i %10.5e %10.5e %10.5e' % (k, self.kcore, self.dm[k-1], tentative_total_he_mass, self.mhe)
                if tentative_total_he_mass >= self.mhe:
                    if verbose: print 'tentative he mass, initial total he mass', tentative_total_he_mass, self.mhe
                    rel_mhe_error = abs(self.mhe - tentative_total_he_mass) / self.mhe
                    if verbose: print 'satisfied he mass conservation to a relative precision of %f' % rel_mhe_error
                    # yout[k] = (self.mhe - (np.dot(yout[self.kcore:], self.dm[self.kcore-1:]) - yout[k] * self.dm[k-1])) / self.dm[k-1]
                    self.nz_shell = k - self.kcore
                    break
                    
        self.have_rainout = self.nz_gradient > 0
        self.have_rainout_to_core = rainout_to_core
                 
        return yout


    def staticModel(self, mtot=1., t10=300., yenv=0.27, zenv=0., mcore=0.,include_he_immiscibility=False,
        verbose=False):
        '''build a hydrostatic model with a given total mass massTotal, 10-bar temperature t10, envelope helium mass fraction yenv,
            envelope heavy element mass fraction zenv, and ice/rock core mass mcore. returns the number of iterations taken before 
            convergence, or -1 for failure to converge.'''
        
        # model atmospheres
        import f11_atm; reload(f11_atm)
        if 0.9 <= mtot <= 1.1:
            if verbose: print 'using jupiter atmosphere tables'
            atm = f11_atm.atm('jup')
            self.teq = 109.0
        elif 0.9 <= mtot * const.mjup / const.msat <= 1.1:
            if verbose: print 'using saturn atmosphere tables'
            atm = f11_atm.atm('sat')
            self.teq = 81.3
        else:
            raise ValueError('model is neither Jupiter- nor Saturn- mass; implement a general model atmosphere option.')

        def mesh_func(t): # t is a parameter that varies from 0 at center to pi/2 at surface
            # really should work out a function giving better resolution at low P, near surface.
            return np.sin(t) ** (1. / 10) * (1. - np.exp(-t ** 2. / (2 * np.pi / 20)))
        
        t = np.linspace(0, np.pi / 2, self.nz)
        # self.m = mtot * const.mjup * np.sin(np.linspace(0, np.pi / 2, self.nz)) ** 5 # this grid gives pretty crummy resolution at low pressure
        self.m = mtot * const.mjup * mesh_func(t)   
        self.kcore = kcore = int(np.where(mcore * const.mearth <= self.m)[0][0])
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
        self.rho[kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), yenv) # for now, no Z in envelope
        # self.rho[kcore:] = self.get_rho_xyz(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y, self.z)
        
        self.y = np.zeros_like(self.p)
        self.y[kcore:] = yenv
        
        self.z[:kcore] = 1.
        self.z[kcore:] = zenv
        
        self.mhe = np.dot(self.y[1:], self.dm) # initial total he mass. must conserve when later adjusting Y distribution
                
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

        # self.rho[:kcore] = 10 ** self.getRockIceDensity((np.log10(self.p[:kcore]))) # core
        self.rho[:kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:kcore]), np.log10(self.t[:kcore])) # core
        if zenv == 0.:
            self.rho[kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:]) # XY envelope
        else:
            self.rho[kcore:] = self.get_rho_xyz(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:], self.z[kcore:]) # XYZ envelope
        
        # relax to hydrostatic
        oldRadii = (0, 0, 0)
        for iteration in xrange(self.max_iters_for_static_model):
            self.iters = iteration + 1
            # hydrostatic equilibrium
            dp[1:] = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4.
            self.p[:] = np.cumsum(dp[::-1])[::-1] + 10 * 1e6 # hydrostatic balance
            
            # compute temperature profile by integrating grad_ad from surface
            self.grada[:kcore] = 0.
            self.grada[kcore:] = self.hhe_eos.get_grada(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])

            # a nan might appear in grada if a p, t point is just outside the original tables.
            # e.g., this was happening at logp, logt = 11.4015234804 3.61913879612, just under
            # the right-hand side "knee" of available data.
            if np.any(np.isnan(self.grada)):
                print '\t%i nans in grada' % len(self.grada[np.isnan(self.grada)])
                
                with open('output/grada_nans.dat', 'w') as fw:
                    for k, val in enumerate(self.grada):
                        if np.isnan(val):
                            fw.write('%16.8f %16.8f %16.8f\n' % (np.log10(self.p[k]), np.log10(self.t[k]), self.y[k]))
                print 'saved problematic logp, logt, y to grada_nans.dat'
                assert False, 'nans in eos quantities; cannot continue.'

            self.dlogp = -1. * np.diff(np.log10(self.p))
            for k in np.arange(self.nz-2, kcore-1, -1):
                logt = np.log10(self.t[k+1]) + self.grada[k+1] * self.dlogp[k]
                self.t[k] = 10 ** logt
                
            self.t[:kcore] = self.t[kcore] # core is isothermal at core-mantle boundary temperature

            # update density from P, T to update hydrostatic r
            self.rho[:kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:kcore]), np.log10(self.t[:kcore])) # core
            if zenv == 0.:
                self.rho[kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:]) # XY envelope
            else:
                self.rho[kcore:] = self.get_rho_xyz(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:], self.z[kcore:]) # XYZ envelope
            
            q[0] = 0.
            q[1:] = 3. * self.dm / 4 / np.pi / self.rho[1:]
            self.r = np.cumsum(q) ** (1. / 3)
            # if verbose: print iteration, self.r[-1]
            if np.all(np.abs(np.mean((oldRadii / self.r[-1] - 1.))) < self.relative_radius_tolerance):
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

        if include_he_immiscibility: # repeat iterations including the phase diagram calculation
            oldRadii = (0, 0, 0)
            for iteration in xrange(self.max_iters_for_static_model):
                self.iters_immiscibility = iteration + 1
                # hydrostatic equilibrium
                dp[1:] = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4.
                self.p[:] = np.cumsum(dp[::-1])[::-1] + 10 * 1e6 # hydrostatic balance
            
                # compute temperature profile by integrating grad_ad from surface
                self.grada[:kcore] = 0.
                self.grada[kcore:] = self.hhe_eos.get_grada(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])

                if np.any(np.isnan(self.grada)):
                    print '\t%i nans in grada' % len(self.grada[np.isnan(self.grada)])
                
                    with open('grada_nans.dat', 'w') as fw:
                        for k, val in enumerate(self.grada):
                            if np.isnan(val):
                                fw.write('%16.8f %16.8f %16.8f\n' % (np.log10(self.p[k]), np.log10(self.t[k]), self.y[k]))
                    print 'saved problematic logp, logt, y to grada_nans.dat'
                    assert False

                self.dlogp = -1. * np.diff(np.log10(self.p))
                for k in np.arange(self.nz-2, kcore-1, -1):
                    logt = np.log10(self.t[k+1]) + self.grada[k+1] * self.dlogp[k]
                    self.t[k] = 10 ** logt

                self.y = self.equilibrium_y_profile()

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
                if np.all(np.abs(np.mean((oldRadii / self.r[-1] - 1.))) < self.relative_radius_tolerance):
                    break
                if not np.isfinite(self.r[-1]):
                    print 'found infinite total radius'
                    return (np.nan, np.nan)
                oldRadii = (oldRadii[1], oldRadii[2], self.r[-1])
            else:
                return -1
            
            if verbose: print 'converged with new Y profile after %i iterations.' % self.iters_immiscibility



        # finalize profiles
        dp[1:] = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4.
        self.p[:] = np.cumsum(dp[::-1])[::-1] + 10 * 1e6
        
        # extrapolate in logp to get a 10-bar temperature
        # f = ((np.log10(1e7) - np.log10(self.p[-2]))) / (np.log10(self.p[-1]) - np.log10(self.p[-2]))
        # logt10 = f * np.log10(self.t[-1]) + (1. - f) * np.log10(self.t[-2])
        # self.t10 = 10 ** logt10
        npts_get_t10 = 10
        logp_near_surf = np.log10(self.p[-npts_get_t10:][::-1])
        t_near_surf = self.t[-npts_get_t10:][::-1]
        t_surf_spline = splrep(logp_near_surf, t_near_surf, s=0, k=3)
        self.t10 = splev(7., t_surf_spline, der=0, ext=0) # ext=0 for extrapolate
        
        self.surface_g = const.cgrav * mtot * const.mjup / self.r[-1] ** 2
        try:
            self.tint = atm.get_tint(self.surface_g * 1e-2, self.t10) # Fortney+2011 needs g in mks
            self.teff = (self.tint ** 4 + self.teq ** 4) ** (1. / 4)
        except ValueError:
            print 'failed to get tint for converged model. g_mks = %f, t10 = %f' % (self.surface_g * 1e-2, self.t10)
            self.tint = 1e20
            self.teff = 1e20
        self.rtot = self.r[-1]
        self.mtot = self.m[-1]
        self.entropy = np.zeros_like(self.p)
        self.entropy[:kcore] = 10 ** self.z_eos.get_logs(np.log10(self.p[:kcore]), np.log10(self.t[:kcore])) * const.mp / const.kb
        # note -- in the envelope, ignoring any contribution that heavies make to the entropy.
        self.entropy[kcore:] = 10 ** self.hhe_eos.get_logs(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:]) * const.mp / const.kb
        self.t[:kcore] = self.t[kcore]
        
        self.g = const.cgrav * self.m / self.r ** 2

        # this is a structure derivative, not a thermodynamic one.
        # wherever the profile is a perfect adiabat, this is also gamma1.
        self.dlogp_dlogrho = np.diff(np.log(self.p)) / np.diff(np.log(self.rho))

        # this gamma1 is general (for H-He mixtures, at least.)
        self.gamma1 = np.zeros_like(self.p)
        self.csound = np.zeros_like(self.p)
        self.chirho = np.zeros_like(self.p)
        self.chit = np.zeros_like(self.p)
        self.gradt = np.zeros_like(self.p)
        
        self.gamma1[:kcore] = self.z_eos.get_gamma1(np.log10(self.p[:kcore]), np.log10(self.t[:kcore]))
        self.gamma1[kcore:] = self.hhe_eos.get_gamma1(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])
        self.csound = np.sqrt(self.p / self.rho * self.gamma1)
        # self.csound[kcore:] = np.sqrt(self.p[kcore:] / self.rho[kcore:] * self.gamma1[kcore:])
        self.lamb_s12 = 2. * self.csound ** 2 / self.r ** 2 # lamb freq. squared for l=1
        self.lamb_s12[self.r == 0.] = 0.
        dlnp_dlnr = np.diff(np.log(self.p)) / np.diff(np.log(self.r))
        dlnrho_dlnr = np.diff(np.log(self.rho)) / np.diff(np.log(self.r))
        
        self.chirho[:kcore] = self.z_eos.get_chirho(np.log10(self.p[:kcore]), np.log10(self.t[:kcore]))
        self.chit[:kcore] = self.z_eos.get_chit(np.log10(self.p[:kcore]), np.log10(self.t[:kcore]))
        self.grada[:kcore] = (1. - self.chirho[:kcore] / self.gamma1[:kcore]) / self.chit[:kcore] # e.g., Unno's equations 13.85, 13.86
        
        self.chirho[kcore:] = self.hhe_eos.get_chirho(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])
        self.chit[kcore:] = self.hhe_eos.get_chit(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])
        self.gradt[:kcore] = 0.
        self.gradt[kcore+1:] = np.diff(np.log(self.t[kcore:])) / np.diff(np.log(self.p[kcore:]))
        
        self.brunt_n2_direct = np.zeros_like(self.p)
        self.brunt_n2_direct[1:] = self.g[1:] / self.r[1:] * (dlnp_dlnr / self.gamma1[1:] - dlnrho_dlnr)
        
        # other forms of the BV frequency
        self.homology_v = const.cgrav * self.m * self.rho / self.r / self.p
        self.brunt_n2_unno_direct = np.zeros_like(self.p)
        self.brunt_n2_unno_direct[kcore+1:] = self.g[kcore+1:] * self.homology_v[kcore+1:] / self.r[kcore+1:] * \
            (self.dlogp_dlogrho[kcore:] ** -1. - self.gamma1[kcore+1:] ** -1.) # Unno 13.102
        
        self.dlogy_dlogp = np.zeros_like(self.p)
        self.dlogy_dlogp[1:] = np.diff(np.log(self.y)) / np.diff(np.log(self.p)) # structure derivative
        # this is the thermodynamic derivative (const P and T), for H-He.
        self.dlogrho_dlogy = np.zeros_like(self.p)
        self.dlogrho_dlogy[kcore:] = self.hhe_eos.get_dlogrho_dlogy(np.log10(self.p[kcore:]), np.log10(self.t[kcore:]), self.y[kcore:])
        
        self.brunt_n2_unno = np.zeros_like(self.p)
        self.brunt_n2_unno[kcore:] = self.g[kcore:] * self.homology_v[kcore:] / self.r[kcore:] * \
            (self.chit[kcore:] / self.chirho[kcore:] * (self.grada[kcore:] - self.gradt[kcore:]) + \
            self.dlogrho_dlogy[kcore:] * self.dlogy_dlogp[kcore:])
        self.brunt_n2_unno[:kcore] = self.g[:kcore] * self.homology_v[:kcore] / self.r[:kcore] * \
            self.chit[:kcore] / self.chirho[:kcore] * (self.grada[:kcore] - self.gradt[:kcore])
            
        self.brunt_n2 = self.brunt_n2_unno

        self.mf = self.m / self.mtot
        self.rf = self.r / self.rtot                
                        
        return self.iters
                
    def run(self,mtot=1., yenv=0.27, zenv=0., mcore=10., starting_t10=3e3, min_t10=200., nsteps=100,include_he_immiscibility=False,output_prefix=None):
        import time
        assert 0. <= mcore*const.mearth <= mtot*const.mjup,\
            "invalid core mass %f for total mass %f" % (mcore, mtot * const.mjup / const.mearth)
        menv = mtot * const.mjup - mcore * const.mearth
        assert 0. <= zenv <= 1., 'invalid envelope z %f' % zenv

        target_t10s = np.logspace(np.log10(min_t10), np.log10(starting_t10), nsteps)[::-1]

        self.history = {}
        self.history_columns = 'age', 'radius', 'tint', 't10', 'teff', 'ysurf', 'lint', 'nz_gradient', 'nz_shell'
        for name in self.history_columns:
            self.history[name] = np.zeros_like(target_t10s)

        previous_entropy = 0
        age_gyr = 0
        # these columns are for the realtime (e.g., notebook) output
        columns = 'step', 'iters', 'tgt_t10', 't10', 'teff', 'radius', 's_mean', 'dt_yr', 'age_gyr', 'nz_gradient', 'nz_shell', 'walltime'
        start_time = time.time()
        print ('%12s ' * len(columns)) % columns
        for step, target_t10 in enumerate(target_t10s):
            self.staticModel(mtot=mtot, t10=target_t10, yenv=yenv, zenv=zenv, mcore=mcore, include_he_immiscibility=include_he_immiscibility)
            walltime = time.time() - start_time
            if self.tint == 1e20:
                print 'failed in atmospheric boundary condition. stopping.'
                if output_prefix:
                    with open('output/%s.history' % output_prefix, 'wb') as fw:
                        pickle.dump(self.history, fw, 0)
                    print 'wrote history data to output/%s.history' % output_prefix
                return self.history
            lint = 4. * np.pi * self.rtot ** 2 * const.sigma_sb * self.tint ** 4
            dt_yr = 0.
            if step > 0:
                delta_s = self.entropy - previous_entropy
                delta_s *= const.kb / const.mp # now erg K^-1 g^-1
                dt = -1. * trapz(self.t * delta_s, dx=self.dm) / lint
                dt_yr = dt / const.secyear
                age_gyr += dt_yr * 1e-9

                self.history['age'][step] = age_gyr
                self.history['radius'][step] = self.rtot
                self.history['tint'][step] = self.tint
                self.history['lint'][step] = lint
                self.history['t10'][step] = self.t10
                self.history['teff'][step] = self.teff
                self.history['ysurf'][step] = self.y[-1]
                self.history['nz_gradient'][step] = self.nz_gradient
                self.history['nz_shell'][step] = self.nz_shell
                
            # history and profile quantities go in dictionaries which are pickled
            if output_prefix:
                assert type(output_prefix) is str, 'output_prefix needs to be a string.'
                self.profile = {}
                
                # profile arrays
                self.profile['p'] = self.p
                self.profile['t'] = self.t
                self.profile['rho'] = self.rho
                self.profile['y'] = self.y
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
                self.profile['rf'] = self.rf
                self.profile['mf'] = self.mf
                
                # profile scalars
                self.profile['nz'] = self.nz
                self.profile['kcore'] = self.kcore
                # self.profile['age']

                with open('output/%s%i.profile' % (output_prefix, step), 'w') as f:
                    pickle.dump(self.profile, f, 0) # 0 means dump as text
                
            print '%12i %12i %12.3f %12.3f %12.3f %12.3e %12.3f %12.3e %12.3f %12i %12i %12.3f' % (step, self.iters, target_t10, self.t10, self.teff, self.rtot, np.mean(self.entropy[self.kcore:]), dt_yr, age_gyr, self.nz_gradient, self.nz_shell, walltime)

            previous_entropy = self.entropy
            
        if output_prefix:
            with open('output/%s.history' % output_prefix, 'wb') as fw:
                pickle.dump(self.history, fw, 0)
            print 'wrote history data to output/%s.history' % output_prefix
            
        return self.history
        
                                    
# if __name__ == "__main__" and sys.argv[1] != '--pylab':
#     Evolver().makeGrid()
