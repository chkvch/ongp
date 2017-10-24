import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d, splrep, splev
from scipy.integrate import trapz, cumtrapz
import sys
import const
import pickle
import time

# this module is adapted from Daniel P. Thorngren's giant planet evolutionary
# code circa 2016ApJ...831...64T. 
#
# the main modifications were the implementation of
#
# (1) the complete Saumon, Chabrier, & van Horn (SCvH-i; 1995ApJS...99..713S) 
#     equation of state for hydrogen and helium, rather than just solar-Y adiabats;
#
# (2) the Fortney+2011 model atmospheres for Jupiter and Saturn (todo: implement
#     the Uranus and Neptune tables also);
#
# (3) the Lorenzen+2011 ab initio phase diagram for hydrogen and helium, obtained
#     courtesy of Nadine Nettelmann;
#
# (4) the Thompson ANEOS for heavy elements, including water, water ice, iron, 
#     serpentine, and dunite;
#
# (5) the Rostock eos (REOS) for water, also obtained courtesy of Nadine Nettelmann.
#     at present won't work for cool planet models since it only covers T > 1000 K; 
#     blends with aneos water at low T. references: 2008ApJ...683.1217N, 2009PhRvB..79e4107F
#     warn that the entropies 


class evol:
    
    def __init__(self,
        hhe_eos_option='scvh', 
        z_eos_option='reos water',
        atm_option='f11_tables',
        nz=1024,
        relative_radius_tolerance=1e-6, # better than 1 km for a Jupiter radius
        max_iters_for_static_model=500, 
        min_iters_for_static_model=12,
        mesh_func_type='hybrid',
        amplitude_transition_mesh_boost=0.1,
        kf_transition_mesh_boost=None,
        width_transition_mesh_boost=0.06,
        extrapolate_phase_diagram_to_low_pressure=True,
        path_to_data='data'):     
                                 
        self.path_to_data = path_to_data
                
        # initialize hydrogen-helium equation of state
        if hhe_eos_option == 'scvh':
            import scvh; reload(scvh)
            self.hhe_eos = scvh.eos(self.path_to_data)
                        
            # load isentrope tables so we can use isentropic P-T profiles at solar composition
            # as starting guesses before we tweak Y, Z distribution
            solar_isentropes = np.load('%s/scvhIsentropes.npz' % self.path_to_data)
            self.logrho_on_solar_isentrope = RegularGridInterpolator(
                (solar_isentropes['entropy'], solar_isentropes['pressure']), solar_isentropes['density'])
            self.logt_on_solar_isentrope = RegularGridInterpolator(
                (solar_isentropes['entropy'], solar_isentropes['pressure']), np.log10(solar_isentropes['temperature']))
                          
        elif eos == 'militzer':
            # this would only be good for a jupiter-like helium fraction anyway
            raise NotImplementedError('no militzer EOS yet.')
        else:
            raise NotImplementedError("this EOS name is not recognized.")
        
        # initialize z equation of state        
        import aneos; reload(aneos)
        import reos; reload(reos)
        if z_eos_option == 'reos water':
            self.z_eos = reos.eos(self.path_to_data)
            self.z_eos_low_t = aneos.eos(self.path_to_data, 'water')
        elif 'aneos' in z_eos_option:
            material = z_eos_option.split()[1]
            self.z_eos = aneos.eos(self.path_to_data, material)
            self.z_eos_low_t = None
        else:
            raise ValueError("z eos option '%s' not recognized." % z_eos_option)
        self.z_eos_option = z_eos_option
                    
        # model atmospheres are now initialized in self.static, so that individual models
        # can be calculated with different model atmospheres within a single Evolver instance.
        # e.g., can run a Jupiter and then a Saturn without making a new Evolver instance.
        self.atm_option = atm_option

        # h-he phase diagram
        import lorenzen; reload(lorenzen)
        self.phase = lorenzen.hhe_phase_diagram(self.path_to_data, extrapolate_to_low_pressure=extrapolate_phase_diagram_to_low_pressure)
                                   
        # initialize structure variables
        self.nz = nz
        self.k = np.arange(self.nz)
        self.p = np.zeros(self.nz)
        self.m = np.zeros(self.nz)
        self.r = np.zeros(self.nz)
        self.rho = np.zeros(self.nz)
        self.t = np.zeros(self.nz)
        self.y = np.zeros(self.nz)
        self.z = np.zeros(self.nz)
        
        # mesh
        self.mesh_func_type = mesh_func_type
        if not kf_transition_mesh_boost: # sets k / nz where give a boost to resolution (e.g. at the mol-met transition)
            self.kf_transition_mesh_boost = 0.
            self.amplitude_transition_mesh_boost = 0.
        else:
            self.kf_transition_mesh_boost = kf_transition_mesh_boost
            self.amplitude_transition_mesh_boost = amplitude_transition_mesh_boost
        self.width_transition_mesh_boost = width_transition_mesh_boost
        
        
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
    
    # mesh function defining mass enclosed within a given zone number
    def mesh_func(self, t):
        # assumes t runs from 0 at center to 1 at surface
        if self.mesh_func_type == 'tanh': # old type
            return 0.5 * (1. + np.tanh(10. * (t * np.pi / 2 - np.pi / 4)))
        # elif self.mesh_func_type == 'new': # specify "density of samples" by hand and turn it into the mesh function
        #     assert self.amplitude_transition_mesh_boost, 'must specify amplitude amplitude_transition_mesh_boost if mesh_func_type is new.'
        #     assert self.t_transition_mesh_boost, 'must specify fractional mass t_transition_mesh_boost if mesh_func_type is new.'
        #     assert self.width_transition_mesh_boost, 'must specify width width_transition_mesh_boost if mesh_func_type is new.'
        #
        #     assert self.amplitude_surface_mesh_boost, 'must specify amplitude amplitude_surface_mesh_boost if mesh_func_type is new.'
        #     assert self.width_surface_mesh_boost, 'must specify width width_surface_mesh_boost if mesh_func_type is new.'
        #
        #     assert self.amplitude_center_mesh_boost, 'must specify amplitude amplitude_center_mesh_boost if mesh_func_type is new.'
        #     assert self.width_center_mesh_boost, 'must specify width width_center_mesh_boost if mesh_func_type is new.'
        #
        #     dens = self.amplitude_center_mesh_boost * np.exp(-self.width_center_mesh_boost * t) \
        #             + self.amplitude_surface_mesh_boost * np.exp(-self.width_surface_mesh_boost * (1. - t)) \
        #             + self.amplitude_transition_mesh_boost \
        #                 * np.exp(-(t - self.t_transition_mesh_boost) ** 2 \
        #                 / (2 * self.width_transition_mesh_boost ** 2))
        #     out = np.cumsum(1. / dens)
        #     out -= out[0]
        #     out /= out[-1]
        #     return out
        elif self.mesh_func_type == 'hybrid': # start with tanh and add extra resolution around some k/nz
            assert self.amplitude_transition_mesh_boost is not None, 'must specify amplitude amplitude_transition_mesh_boost if mesh_func_type is hybrid.'
            assert self.kf_transition_mesh_boost is not None, 'must specify fractional zone number kf_transition_mesh_boost if mesh_func_type is hybrid.'
            assert self.width_transition_mesh_boost, 'must specify width width_transition_mesh_boost if mesh_func_type is hybrid.'
            
            a = self.amplitude_transition_mesh_boost # nominally 1
            b = self.kf_transition_mesh_boost
            c = self.width_transition_mesh_boost
            
            # our typical tanh mesh for good resolution near surface
            f0 = 0.5 * (1. + np.tanh(10. * (t * np.pi / 2 - np.pi / 4)))
            # convert to density of samples in zone number space
            density_f0 = 1. / np.diff(f0)
            # duplicate inner density to match original length
            density_f0 = np.insert(density_f0, 0, density_f0[0])
            # at present symmetric across middle zone. instead give less weight to inner parts than surface.
            density_f0 *= np.exp(t / 0.1)
            # add a gaussian bump in sample density at k/nz = b = self.kf_transition_mesh_boost
            density_f0 += a * density_f0[0] * np.exp(-(t - b) ** 2 / 2 / c ** 2)
            # flip back into a cumulative fraction in mass space
            out = np.cumsum(1. / density_f0)
            out -= out[0]
            out /= out[-1]
            return out
        else:
            raise ValueError('mesh type %s not recognized.' % self.mesh_func_type)
            
            
    # when building non-isentropic models [for instance, a grad(P, T, Y, Z)=grada(P, T, Y, Z) model
    # with Y, Z functions of depth so that entropy is non-uniform], want to be able to get density 
    # as a function of these same independent variables P, T, Y, Z.
    
    def get_rho_z(self, logp, logt):
        '''helper function to get rho of just the z component. different from self.z_eos.get_logrho because
        this does the switching to aneos water at low T if using reos water as the z eos.'''
        
        if self.z_eos_option == 'reos water':
            # if using reos water, extend down to T < 1000 K using aneos water.
            # for this, mask to find the low-T part.            
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
        # only meant to be called when Z is non-zero and Y is not 0 or 1
        rho_hhe = 10 ** self.hhe_eos.get_logrho(logp, logt, y)
        rho_z = self.get_rho_z(logp, logt)
        rhoinv = (1. - z) / rho_hhe + z / rho_z
        return rhoinv ** -1

    def static(self, mtot=const.mjup, t1=165., yenv=0.27, zenv=0., mcore=0., # mtot specified in Jupiter masses; mcore in Earth masses. yenv and zenv are mass fractions for helium and heavies respectively.
                    zenv_inner=None, # if None, envelope has uniform Z equal to zenv
                    yenv_inner=None, # if None, envelope has uniform Y equal to yenv
                    include_he_immiscibility=False, # whether to move helium around according to a H-He phase diagram
                    phase_t_offset=0, # temperature offset for the phase diagram, only used if include_he_immiscibility
                    minimum_y_is_envelope_y=False, # hack for dealing with non-monotone Y profiles resulting from phase diagram, only used if include_he_immiscibility
                    rrho_where_have_helium_gradient=None, # superadiabaticity (parameterized as the density ratio R_rho) to assume wherever there's a stabilizing composition gradient
                    erase_z_discontinuity_from_brunt=False, # ignore Z discontinuity when calculating buoyancy frequency's composition term
                    include_core_entropy=False, # can include if using a Z eos with entropy information (like REOS water); not necessarily important for every problem
					hydrogen_transition_pressure=1., # pressure to assume for the molecular-metallic interface (where Y or Z discontinuities might exist, or where helium gradient might start)
                    core_prho_relation=None, # if want to use Hubbard + Marley 1989 P(rho) relations instead of a newer Z eos
                    verbose=False):
        '''build a hydrostatic model with a given total mass mtot, 1-bar temperature t1, envelope helium mass fraction yenv,
            envelope heavy element mass fraction zenv, and heavy-element core mass mcore. returns the number of iterations taken before 
            convergence, or -1 for failure to converge.'''

        if type(mtot) is str:
            if mtot[0] == 'j':
                mtot = const.mjup
            elif mtot[0] == 's':
                mtot = const.msat
            else:
                raise ValueError("if type(mtot) is str, first element must be j, s") # or u and n, later
        
        self.mtot = mtot
        
        # model atmospheres
        if 0.9 <= self.mtot / const.mjup <= 1.1:
            if verbose: print 'using jupiter atmosphere tables'
            if self.atm_option is 'f11_tables':
                import f11_atm
                atm = f11_atm.atm(self.path_to_data, 'jup')
            elif self.atm_option is 'f11_fit':
                import f11_atm_fit
                atm = f11_atm_fit.atm(self.path_to_data, 'jup')
            else:
                raise ValueError('atm_option %s not recognized.' % self.atm_option)
            self.teq = 109.0 # make this free parameter!
        elif 0.9 <= self.mtot / const.msat <= 1.1:
            if verbose: print 'using saturn atmosphere tables'
            if self.atm_option is 'f11_tables':
                import f11_atm
                atm = f11_atm.atm(self.path_to_data, 'sat')
            elif self.atm_option is 'f11_fit':
                import f11_atm_fit
                atm = f11_atm_fit.atm('sat')
            else:
                raise ValueError('atm_option %s not recognized.' % self.atm_option)
            self.teq = 81.3 # make this free parameter!
        else:
            raise ValueError('model is not within 10 percent of either Jupiter- or Saturn- mass; surface is bound to run off J or S tables. implement a general model atmosphere option.')
        
        self.core_prho_relation = core_prho_relation # if None (default), use z_eos_option
        
        self.zenv = zenv
        self.zenv_inner = zenv_inner
        self.zenv_outer = self.zenv # in the three-layer case, zenv is effectively an alias for zenv_outer
        
        self.yenv = yenv
        self.yenv_inner = yenv_inner
        self.yenv_outer = self.yenv
        
        # initialize lagrangian mesh.
        # because of discretization, self.mcore not necessarily mcore specified as argument.
        #
        # solution for now: add a new mesh point at mcore (total number of zones is now nz + 1).
        # note: structure arrays are initialized in evol.__init__ with length nz.
        # so initialize self.m with length < nz before adding zones such that total is nz.
        #
        assert mcore * const.mearth < self.mtot, 'core mass must be (well) less than total mass.'
        if mcore > 0.:
            t = np.linspace(0, 1, self.nz - 1)
            self.m = self.mtot * self.mesh_func(t) # grams
            self.m *= self.mtot / self.m[-1] # guarantee surface zone has mtot enclosed
            self.kcore = kcore = np.where(self.m >= mcore * const.mearth)[0][0] # kcore - 1 is last zone with m < mcore
            self.m = np.insert(self.m, self.kcore, mcore * const.mearth) # kcore is the zone where m == mcore. this zone should have z=1.
            self.kcore += 1 # so say self.rho[:kcore] wil encompass all the zones with z==1.
        else: # no need for extra precautions
            t = np.linspace(0, 1, self.nz)
            self.m = self.mtot * self.mesh_func(t) # grams   
            self.m *= self.mtot / self.m[-1] # guarantee surface zone has mtot enclosed
            self.kcore = 0         
        self.mcore = mcore
        
        self.grada = np.zeros_like(self.m)

        self.dm = np.diff(self.m)
        if abs(np.sum(self.dm) - self.mtot) / const.mearth > 0.001:
            print 'warning: total mass is different from requested by more than 0.001 earth masses.', \
                abs(np.sum(self.dm) - self.mtot) / const.mearth  
                                
        # first guess where densities will be calculable
        self.p[:] = 1e12
        self.t[:] = 1e4       
        
        # set initial composition information. for now, only a one-layer envelope
        self.y[:] = 0.
        self.y[self.kcore:] = self.yenv_outer
        
        self.z[:self.kcore] = 1.
        assert zenv >= 0., 'got negative zenv %g' % zenv
        self.z[self.kcore:] = self.zenv_outer

        if self.core_prho_relation:
            self.rho[:self.kcore] = 8 # just an initial guess for root find of hubbard + marley p-rho relations
            if self.core_prho_relation == 'hm89 rock':
                self.rho[:self.kcore] = self.get_rhoz_hm89_rock(self.p[:self.kcore], self.rho[:self.kcore])
            elif self.core_prho_relation == 'hm89 ice':
                self.rho[:self.kcore] = self.get_rhoz_hm89_ice(self.p[:self.kcore], self.rho[:self.kcore])
            else:
                assert ValueError, "core_prho_relation must be one of 'hm89 rock' or 'hm89 ice'."
        else:
            self.rho[:self.kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
        self.rho[self.kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), yenv) # for first pass, no Z in envelope
        
        self.mhe = np.dot(self.y[1:], self.dm) # initial total he mass. for reference if later adjusting Y distribution
        self.k_shell_top = 0 # until a shell is found by equilibrium_y_profile
                
        # continuity equation
        # an approximation
        q = np.zeros_like(self.rho)
        q[0] = 0.
        q[1:] = 3. * self.dm / 4 / np.pi / self.rho[1:]
        self.r = np.cumsum(q) ** (1. / 3)
        
        # hydrostatic balance
        dp = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4
        self.p[-1] = 1e6
        self.p[:-1] = np.cumsum(dp[::-1])[::-1] + 1e6
                       
        # set surface temperature and integrate the adiabat inward to get temperature         
        self.t[-1] = t1         
        self.grada[self.kcore:] = self.hhe_eos.get_grada(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
        for k in np.arange(self.nz)[::-1]:
            if k == self.nz - 1: continue
            if k == self.kcore - 1: break
            dlnp = np.log(self.p[k]) - np.log(self.p[k+1])
            dlnt = self.grada[k] * dlnp
            self.t[k] = self.t[k+1] * (1. + dlnt)

        self.t[:self.kcore] = self.t[self.kcore] # isothermal at temperature of core-mantle boundary
        
        # identify molecular-metallic transition. not relevant for U, N
        self.ktrans = np.where(self.p >= hydrogen_transition_pressure * 1e12)[0][-1]

        if self.zenv_inner: # two-layer envelope in terms of Z distribution. zenv is z of the outer envelope, zenv_inner is z of the inner envelope.
            assert zenv_inner > 0, 'if you want a z-free envelope, no need to specify zenv_inner.'
            assert zenv_inner >= zenv, 'no z inversion allowed.'
            self.z[self.kcore:self.ktrans] = self.zenv_inner
        if self.yenv_inner:
            assert yenv_inner > 0, 'if you want a Y-free envelope, no need to specify yenv_inner.'
            assert yenv_inner >= yenv, 'no y inversion allowed.'
            self.y[self.kcore:self.ktrans] = self.yenv_inner

        # set density in core
        if self.core_prho_relation:
            if self.core_prho_relation == 'hm89 rock':
                self.rho[:self.kcore] = self.get_rhoz_hm89_rock(self.p[:self.kcore], self.rho[:self.kcore])
            elif self.core_prho_relation == 'hm89 ice':
                self.rho[:self.kcore] = self.get_rhoz_hm89_ice(self.p[:self.kcore], self.rho[:self.kcore])
            else:
                assert ValueError, "core_prho_relation must be one of 'hm89 rock' or 'hm89 ice'."
        else:
            self.rho[:self.kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))

        # set density in envelope
        if self.z[-1] == 0.: # XY envelope
            self.rho[self.kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:]) # XY envelope
        else: # XYZ envelope
            self.rho[self.kcore:] = self.get_rho_xyz(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:], self.z[self.kcore:]) # XYZ envelope            
            
                        
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
            dp = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4
            self.p[-1] = 1e6
            self.p[:-1] = self.p[-1] + np.cumsum(dp[::-1])[::-1]
                        
            # identify molecular-metallic transition
            self.ktrans = np.where(self.p >= hydrogen_transition_pressure * 1e12)[0][-1]

            if self.zenv_inner: # two-layer envelope in terms of Z distribution. zenv is z of the outer envelope, zenv_inner is z of the inner envelope
                assert zenv_inner > 0, 'if you want a z-free envelope, no need to specify zenv_inner.'
                assert zenv_inner >= zenv, 'no z inversion allowed.'
                self.z[self.kcore:self.ktrans] = self.zenv_inner
                self.z[self.ktrans:] = self.zenv_outer
            if self.yenv_inner:
                assert yenv_inner > 0, 'if you want a Y-free envelope, no need to specify yenv_inner.'
                assert yenv_inner >= yenv, 'no y inversion allowed.'
                self.y[self.kcore:self.ktrans] = self.yenv_inner
                self.y[self.ktrans:] = self.yenv_outer
            
            # compute temperature profile by integrating grad_ad from surface
            self.grada[:self.kcore] = 0.
            self.grada[self.kcore:] = self.hhe_eos.get_grada(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
            self.gradt = np.copy(self.grada) # may be modified later if include_he_immiscibility and rrho_where_have_helium_gradient
            
            # a nan might appear in grada if a p, t point is just outside the original tables.
            # e.g., this was happening at logp, logt = 11.4015234804 3.61913879612, just under
            # the right-hand side "knee" of available data.
            if np.any(np.isnan(self.grada)):
                num_nans = len(self.grada[np.isnan(self.grada)])

                # raise EOSError('%i nans in grada. first (logT, logP)=(%f, %f); last (logT, logP) = (%f, %f)' % \
                #     (num_nans, np.log10(self.t[np.isnan(self.grada)][0]), np.log10(self.p[np.isnan(self.grada)][0]), \
                #     np.log10(self.t[np.isnan(self.grada)][-1]), np.log10(self.p[np.isnan(self.grada)][-1])))
                
                if iteration < 5 and len(self.grada[np.isnan(self.grada)]) < self.nz / 4:
                    '''early in iterations and fewer than nz/4 nans; attempt to coax grada along.
                    
                    always always always always iteration 2.
                    
                    really not a big deal if we invent some values for grada this early in iterations since
                    many more will follow.
                    '''
                    # print '%i nans in grada for iteration %i, attempt to continue' % (num_nans, iteration)
                    where_nans = np.where(np.isnan(self.grada))
                    k_first_nan = where_nans[0][0]
                    k_last_nan = where_nans[0][-1]
                    last_good_grada = self.grada[k_first_nan - 1]
                    first_good_grada = self.grada[k_last_nan + 1]
                    self.grada[k_first_nan:k_last_nan+1] = (self.r[k_first_nan:k_last_nan+1] - self.r[k_first_nan]) \
                                                            / (self.r[k_last_nan+1] - self.r[k_first_nan]) \
                                                            * (first_good_grada - last_good_grada) + last_good_grada
                else: # abort                
                    print '%i nans in grada for iteration %i, stopping' % (num_nans, iteration)
                    with open('grada_nans.dat', 'w') as fw:
                        for k, val in enumerate(self.grada):
                            if np.isnan(val):
                                fw.write('%16.8f %16.8f %16.8f\n' % (np.log10(self.p[k]), np.log10(self.t[k]), self.y[k]))
                    print 'saved problematic logp, logt, y to grada_nans.dat'
                    raise EOSError('nans in grada.')

            self.dlogp = -1. * np.diff(np.log10(self.p))
            for k in np.arange(self.nz-2, self.kcore-1, -1):
                logt = np.log10(self.t[k+1]) + self.grada[k+1] * self.dlogp[k]
                self.t[k] = 10 ** logt
                
            self.t[:self.kcore] = self.t[self.kcore] # core is isothermal at core-mantle boundary temperature

            # set density in core
            if self.core_prho_relation:
                if self.core_prho_relation == 'hm89 rock':
                    self.rho[:self.kcore] = self.get_rhoz_hm89_rock(self.p[:self.kcore], self.rho[:self.kcore])
                elif self.core_prho_relation == 'hm89 ice':
                    self.rho[:self.kcore] = self.get_rhoz_hm89_ice(self.p[:self.kcore], self.rho[:self.kcore])
                else:
                    assert ValueError, "core_prho_relation must be one of 'hm89 rock' or 'hm89 ice'."
            else:
                self.rho[:self.kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
            if self.z[-1] == 0.: # again, this check is only valid if envelope is homogeneous in Z
                self.rho[self.kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:]) # XY envelope
            else:
                self.rho[self.kcore:] = self.get_rho_xyz(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:], self.z[self.kcore:]) # XYZ envelope
            
            if np.any(np.isnan(self.rho)):
                raise EOSError('have one or more nans in rho after eos call.')
                
            
            # continuity
            q[0] = 0.
            q[1:] = 3. * self.dm / 4 / np.pi / self.rho[1:]
            self.r = np.cumsum(q) ** (1. / 3)
            
            if np.all(np.abs(np.mean((oldRadii / self.r[-1] - 1.))) < self.relative_radius_tolerance) and iteration >= self.min_iters_for_static_model:
                break
            if not np.isfinite(self.r[-1]):
                raise HydroError('found infinite total radius.')
                
            oldRadii = (oldRadii[1], oldRadii[2], self.r[-1])
                        
        else:
            return -1
                        
        if verbose: print 'converged homogeneous model after %i iterations.' % self.iters

        if include_he_immiscibility: # repeat iterations, now including the phase diagram calculation (if modeling helium rain)
            oldRadii = (0, 0, 0)
            for iteration in xrange(self.max_iters_for_static_model):
                self.iters_immiscibility = iteration + 1
                # hydrostatic equilibrium
                dp = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4
                self.p[-1] = 1e6
                self.p[:-1] = np.cumsum(dp[::-1])[::-1] + 1e6
            
                # compute temperature profile by integrating gradt from surface
                self.grada[:self.kcore] = 0. # can compute after model is converged, since we take core to be isothermal
                self.grada[self.kcore:] = self.hhe_eos.get_grada(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:]) # last time grada is set for envelope

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

                    self.chit[self.kcore:] = self.hhe_eos.get_chit(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
                    self.chirho[self.kcore:] = self.hhe_eos.get_chirho(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
                    self.chiy[self.kcore:] = self.hhe_eos.get_chiy(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
                    self.dlogy_dlogp[1:] = np.diff(np.log(self.y)) / np.diff(np.log(self.p)) # actual log rate of change of Y with P
                    self.brunt_b[self.kcore+1:] = self.chirho[self.kcore+1:] / self.chit[self.kcore+1:] * self.chiy[self.kcore+1:] * self.dlogy_dlogp[self.kcore+1:]
                    self.brunt_b[self.k_shell_top + 1] = 0.
                    # print 'in internal brunt_b calculation: self.k_shell_top = %i; brunt_b[self.k_shell_top] == %g' % (self.k_shell_top, self.brunt_b[self.k_shell_top+1])
                    assert np.all(self.brunt_b[self.kcore+1:self.k_shell_top+1] == 0) # he shell itself should be uniform 
                                        
                    self.gradt += rrho_where_have_helium_gradient * self.brunt_b
                                                
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

                print 'WARNING: calling equilibrium_y_profile. check to see that this routine sees the correct total helium mass.'
                print 'changes were made c. september 2017 to support a Y jump at the transition; initial total he mass not necessarily stored correctly.'
                self.y = self.equilibrium_y_profile(phase_t_offset, minimum_y_is_envelope_y=minimum_y_is_envelope_y, hydrogen_transition_pressure=hydrogen_transition_pressure)

                # set density in core
                if core_prho_relation:
                    if self.core_prho_relation == 'hm89 rock':
                        self.rho[:self.kcore] = self.get_rhoz_hm89_rock(self.p[:self.kcore], self.rho[:self.kcore])
                    elif self.core_prho_relation == 'hm89 ice':
                        self.rho[:self.kcore] = self.get_rhoz_hm89_ice(self.p[:self.kcore], self.rho[:self.kcore])
                    else:
                        assert ValueError, "core_prho_relation must be one of 'hm89 rock' or 'hm89 ice'."
                else:
                    self.rho[:self.kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
                    # set density in envelope
                    if zenv == 0.:
                        self.rho[self.kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
                    else:
                        self.rho[self.kcore:] = self.get_rho_xyz(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:], self.z[self.kcore:])
            
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
                    raise RuntimeError('found nans in rho on static iteration %i' % self.iters)

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
        dp = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4
        self.p[-1] = 1e6
        self.p[:-1] = np.cumsum(dp[::-1])[::-1] + self.p[-1]
        # update density for the last time
        # set density in core
        if self.core_prho_relation:
            if self.core_prho_relation == 'hm89 rock':
                self.rho[:self.kcore] = self.get_rhoz_hm89_rock(self.p[:self.kcore], self.rho[:self.kcore])
            elif self.core_prho_relation == 'hm89 ice':
                self.rho[:self.kcore] = self.get_rhoz_hm89_ice(self.p[:self.kcore], self.rho[:self.kcore])
            else:
                assert ValueError, "core_prho_relation must be one of 'hm89 rock' or 'hm89 ice'."
        else:
            self.rho[:self.kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
        if zenv == 0.:
            self.rho[self.kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:]) # XY envelope
        else:
            self.rho[self.kcore:] = self.get_rho_xyz(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:], self.z[self.kcore:]) # XYZ envelope
        
        self.rtot = self.r[-1]
        self.mtot = self.m[-1]

        # identify molecular-metallic transition
        self.ktrans = np.where(self.p >= hydrogen_transition_pressure * 1e12)[0][-1]
                
        self.t1 = self.t[-1]
        
        k10 = np.where(self.p < 1e7)[0][0]
        tck = splrep(self.p[k10-5:k10+5][::-1], self.t[k10-5:k10+5][::-1], k=3)
        self.t10 = splev(1e7, tck)        
        
        assert self.t10 > 0., 'negative t10 %g' % self.t10
        assert self.t1 > 0., 'negative t1 %g' % self.t1
        
        # calculate the gravity and get intrinsic temperature from the model atmospheres.
        self.surface_g = const.cgrav * mtot / self.r[-1] ** 2
        try:
            self.tint = atm.get_tint(self.surface_g * 1e-2, self.t10) # Fortney+2011 needs g in mks
            self.teff = (self.tint ** 4 + self.teq ** 4) ** (1. / 4)
            self.lint = 4. * np.pi * self.rtot ** 2 * const.sigma_sb * self.tint ** 4
        except ValueError:
            if self.zenv_inner:
                print 'z2, z1 = ', self.zenv_inner, self.zenv_outer
            else:
                print 'zenv = ', self.zenv
            raise AtmError('off atm tables: g_mks = %5.2f, t10 = %5.2f. rtot = %10.5g' % (self.surface_g*1e-2, self.t10, self.rtot))
        self.entropy = np.zeros_like(self.p)
        # experimenting with including entropy of core material (don't bother with aneos, it's not a column).
        if include_core_entropy:
            if not self.z_eos_option == 'reos water':
                raise NotImplementedError("including entropy of the core is only possible if z_eos_option == 'reos water'.")
            else:
                self.entropy[:self.kcore] = 10 ** self.z_eos.get_logs(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore])) * const.mp / const.kb
        self.entropy[self.kcore:] = 10 ** self.hhe_eos.get_logs(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:]) * const.mp / const.kb

        # core is isothermal at temperature of the base of the envelope.
        self.t[:self.kcore] = self.t[self.kcore]
        
        self.r[0] = 1. # 1 cm central radius to keep these things calculable at center zone
        self.g = const.cgrav * self.m / self.r ** 2

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
        
        try:
            self.chirho[:self.kcore] = self.z_eos.get_chirho(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
            self.chit[:self.kcore] = self.z_eos.get_chit(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
            self.grada[:self.kcore] = (1. - self.chirho[:self.kcore] / self.gamma1[:self.kcore]) / self.chit[:self.kcore] # e.g., Unno's equations 13.85, 13.86
        except AttributeError:
            print "warning: z eos option '%s' does not provide methods for get_chirho and get_chit." % self.z_eos_option
            print 'cannot calculate things like grada in core and so this model may not be suited for eigenmode calculations.'
            pass
        
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
            self.dlogy_dlogp[self.kcore:] = np.diff(np.log(self.y[self.kcore-1:])) / np.diff(np.log(self.p[self.kcore-1:])) # structure derivative
        else:
            self.dlogy_dlogp[1:] = np.diff(np.log(self.y)) / np.diff(np.log(self.p))
        # this is the thermodynamic derivative (const P and T), for H-He.
        self.dlogrho_dlogy = np.zeros_like(self.p)
        self.dlogrho_dlogy[self.kcore:] = self.hhe_eos.get_dlogrho_dlogy(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
        
        rho_z = np.zeros_like(self.p)
        rho_hhe = np.zeros_like(self.p)
        # rho_z[self.z > 0.] = 10 ** self.z_eos.get_logrho(np.log10(self.p[self.z > 0.]), np.log10(self.t[self.z > 0.]))
        rho_z[self.z > 0.] = self.get_rho_z(np.log10(self.p[self.z > 0.]), np.log10(self.t[self.z > 0.]))
        rho_z[self.z == 0.] = 0
        rho_hhe[self.y > 0.] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[self.y > 0.]), np.log10(self.t[self.y > 0.]), self.y[self.y > 0.])
        rho_hhe[self.y == 0.] = 10 ** self.hhe_eos.get_h_logrho((np.log10(self.p[self.y == 0]), np.log10(self.t[self.y == 0.])))
        self.dlogrho_dlogz = np.zeros_like(self.p)
        # dlogrho_dlogz is only calculable where all of X, Y, and Z are non-zero.
        self.dlogrho_dlogz[self.z * self.y > 0.] = -1. * self.rho[self.z * self.y > 0.] * self.z[self.z * self.y > 0.] * (rho_z[self.z * self.y > 0.] ** -1 - rho_hhe[self.z * self.y > 0.] ** -1)
        self.dlogz_dlogp = np.zeros_like(self.p)
        self.dlogz_dlogp[1:] = np.diff(np.log(self.z)) / np.diff(np.log(self.p))
        
        
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
        self.brunt_n2_thermal = self.g ** 2 * self.rho / self.p * self.chit / self.chirho * (self.grada - self.gradt)
                
        # this is the thermo derivative rho_t in scvh parlance. necessary for gyre, which calls this minus delta.
        # dlogrho_dlogt_const_p = chit / chirho = -delta = -rho_t
        self.dlogrho_dlogt_const_p = np.zeros_like(self.p)
        # print 'at time of calculating rho_t for final static model, log core temperature is %f' % np.log10(self.t[0])
        self.dlogrho_dlogt_const_p[:self.kcore] = self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
        if self.z_eos_option == 'reos water' and self.t[-1] < 1e3: # must be calculated separately for low T and high T part of the envelope
            k_t_boundary = np.where(np.log10(self.t) > 3.)[0][-1]
            try:
                self.dlogrho_dlogt_const_p[self.kcore:k_t_boundary+1] = self.rho[self.kcore:k_t_boundary+1] * \
                    (self.z[self.kcore:k_t_boundary+1] / rho_z[self.kcore:k_t_boundary+1] * self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[self.kcore:k_t_boundary+1]), np.log10(self.t[self.kcore:k_t_boundary+1])) + \
                    (1. - self.z[self.kcore:k_t_boundary+1]) / rho_hhe[self.kcore:k_t_boundary+1] * self.hhe_eos.get_rhot(np.log10(self.p[self.kcore:k_t_boundary+1]), np.log10(self.t[self.kcore:k_t_boundary+1]), self.y[self.kcore:k_t_boundary+1]))
            except:
                print 'failed in dlogrho_dlogt_const_p for hi-T part of envelope'
                raise
            try:
                self.dlogrho_dlogt_const_p[k_t_boundary+1:] = self.rho[k_t_boundary+1:] * (self.z[k_t_boundary+1:] / rho_z[k_t_boundary+1:] * \
                    self.z_eos_low_t.get_dlogrho_dlogt_const_p(np.log10(self.p[k_t_boundary+1:]), np.log10(self.t[k_t_boundary+1:])) + \
                    (1. - self.z[k_t_boundary+1:]) / rho_hhe[k_t_boundary+1:] * self.hhe_eos.get_rhot(np.log10(self.p[k_t_boundary+1:]), np.log10(self.t[k_t_boundary+1:]), self.y[k_t_boundary+1:]))
            except:
                print 'failed in dlogrho_dlogt_const_p for lo-T part of envelope'
                raise

        else: # no need to sweat low vs. high t
            self.dlogrho_dlogt_const_p[self.kcore:] = self.rho[self.kcore:] * (self.z[self.kcore:] / rho_z[self.kcore:] * self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:])) + (1. - self.z[self.kcore:]) / rho_hhe[self.kcore:] * self.hhe_eos.get_rhot(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:]))
            
        self.mf = self.m / self.mtot
        self.rf = self.r / self.rtot  
                
        if self.zenv_inner: # two-layer envelope in terms of Z
            self.mz_env_outer = np.sum(self.dm[self.ktrans:]) * self.z[self.ktrans + 1]
            self.mz_env_inner = np.sum(self.dm[self.kcore:self.ktrans]) * self.z[self.ktrans - 1]
            self.mz_core = np.dot(self.z[:self.kcore], self.dm[:self.kcore])
            self.mz = self.mz_env_outer + self.mz_env_inner + self.mz_core
        else:
            # Z uniform in envelope, Z=1 in core. 
            self.mz_env = self.z[-1] * (self.mtot - self.mcore * const.mearth)
            self.mz = self.mz_env + self.mcore * const.mearth
        
        self.bulk_z = self.mz / self.mtot
        self.ysurf = self.y[-1]
        
        # axial moment of inertia, in units of mtot * rtot ** 2. moi of a thin spherical shell is 2 / 3 * m * r ** 2
        self.nmoi = 2. / 3 * trapz(self.r ** 2, x=self.m) / self.mtot / self.rtot ** 2
        
        self.pressure_scale_height = self.p / self.rho / self.g

                        
        return self.iters
        
    # these implement the analytic p(rho) relations for "rock" and "ice" mixtures from Hubbard & Marley 1989

    def p_of_rho_hm89_rock(self, rho):
        return rho ** 4.406 * np.exp(-6.579 - 0.176 * rho + 0.00202 * rho ** 2.) # Mbar

    def p_of_rho_hm89_ice(self, rho):
        return rho ** 3.719 * np.exp(-2.756 - 0.271 * rho + 0.00701 * rho ** 2.) # Mbar
        
    def get_rhoz_hm89_rock(self, p, rho_now):
        from scipy.optimize import root
        zero_me = lambda rho: self.p_of_rho_hm89_rock(rho) - p * 1e-12
        res = root(zero_me, rho_now)
        assert res['success'], 'failed root find in get_rhoz_hm89_rock'
        return res.x

    def get_rhoz_hm89_ice(self, p, rho_now):
        from scipy.optimize import root
        zero_me = lambda rho: self.p_of_rho_hm89_ice(rho) - p * 1e-12
        res = root(zero_me, rho_now)
        assert res['success'], 'failed root find in get_rhoz_hm89_ice'
        return res.x

    def equilibrium_y_profile(self, phase_t_offset, verbose=False, show_timing=False, minimum_y_is_envelope_y=False, hydrogen_transition_pressure=None):
        '''uses the existing p-t profile to find the thermodynamic equilibrium y profile, which
        may require a nearly pure-helium layer atop the core.'''
        p = self.p * 1e-12 # Mbar
        k1 = np.where(p > hydrogen_transition_pressure)[0][-1]
        ymax1 = self.phase.ymax_lhs(p[k1], self.t[k1] - phase_t_offset)
        
        if np.isnan(ymax1) or self.y[k1] < ymax1:
            if verbose: print 'first point at P > P_trans = %1.1f Mbar is stable to demixing. Y = %1.4f, Ymax = %1.4f' % (hydrogen_transition_pressure, self.y[k1], ymax1)
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
        for k in np.arange(k1-1, self.kcore, -1): # inward from first point where P > P_trans
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
            # gradient extends down to kbot, below which the rest of the envelope is already set Y=0.95.  
            # since proposed envelope mhe < initial mhe, must grow the He-rich shell to conserve total mass.
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
                
    def run(self, mtot=1., yenv=0.27, zenv=0., mcore=10., starting_t1=2e3, min_t1=160., nsteps=100,
                stdout_interval=1, # output controls
                output_prefix=None,
                include_he_immiscibility=False, # helium rain 
                phase_t_offset=0., 
                minimum_y_is_envelope_y=False, 
                rrho_where_have_helium_gradient=None,
                max_age=None, # general
                include_core_entropy=False, 
                gammainv_erosion=1e-1, # for core erosion rate estimates
                luminosity_erosion_option='first_convective_cell',
                timesteps_ease_in=None):
                
        '''builds a sequence of static models with different surface temperatures and calculates the delta time between each pair
        using the energy equation dL/dm = -T * ds/dt where L is the intrinsic luminosity, m is the mass coordinate, T is the temperature,
        s is the specific entropy, and t is time.
        important inputs are starting_t1 (say 1, 2, 3 thousand K) and min_t1 (put just beneath a realistic t1 for J/S/U/N or else it'll run off F11 atm tables.)
        for now ignores the possibility of having two-layer envelope in terms of Y or Z'''
                
        import time
        assert 0. <= zenv <= 1., 'invalid envelope z %f' % zenv

        # set vector of t1s to compute
        if timesteps_ease_in:
            # make the first steps_to_ease_in steps more gradual in terms of d(dt)/d(step).
            # only important if you're interested in resolving evolution at very early times (1e3 or 1e4 years).
            i = np.arange(nsteps)
            t1s = 0.5 * (starting_t1 - min_t1) * (1. + np.tanh(np.pi * (i - 3. * nsteps / 5.) / (2. / 3) / nsteps))[::-1] + min_t1
        else:
            t1s = np.logspace(np.log10(min_t1), np.log10(starting_t1), nsteps)[::-1]
        # return t1s

        self.history = {}
        self.history_columns = 'step', 'age', 'dt_yr', 'radius', 'tint', 't1', 't10', 'teff', 'ysurf', 'lint', 'nz_gradient', 'nz_shell', 'iters', \
            'mz_env', 'mz', 'bulk_z', 'dmcore_dt_guillot', 'int_dmcore_dt_guillot', 'dmcore_dt_garaud', 'int_dmcore_dt_garaud', \
            'dmcore_dt_guillot_alternate', 'int_dmcore_dt_guillot_alternate'
        for name in self.history_columns:
            # allocate history arrays with the length of steps we expect.
            # bad design because if the run terminates early, rest of history
            # will be filled with zeroes. keep in mind.
            self.history[name] = np.zeros_like(t1s)

        keep_going = True
        previous_entropy = 0
        age_gyr = 0
        # these columns are for the realtime (e.g., notebook) output
        stdout_columns = 'step', 'iters', 't1', 'teff', 'radius', 's_mean', 'dt_yr', 'age_gyr', 'nz_gradient', 'nz_shell', 'y_surf', 'walltime'
        start_time = time.time()
        print ('%12s ' * len(stdout_columns)) % stdout_columns
        for step, t1 in enumerate(t1s):
            try:
                self.static(mtot=mtot, t1=t1, yenv=yenv, zenv=zenv, mcore=mcore, 
                                include_he_immiscibility=include_he_immiscibility,
                                phase_t_offset=phase_t_offset, 
                                minimum_y_is_envelope_y=minimum_y_is_envelope_y, 
                                rrho_where_have_helium_gradient=rrho_where_have_helium_gradient,
                                include_core_entropy=include_core_entropy)
                walltime = time.time() - start_time
            except ValueError:
                print('failed in building static model -- likely off eos or atm tables')
                # don't update any history info; save history to this point if output_prefix is specified
                if output_prefix:
                    with open('%s.history' % output_prefix, 'wb') as fw:
                        pickle.dump(self.history, fw, 0)
                    print 'wrote history data to %s.history' % output_prefix
                return self.history
            dt_yr = 0.
            # this is the piece daniel now uses scipy.integrate.odeint to integrate instead
            if step > 0: # there is a timestep to speak of
                delta_s = self.entropy - previous_entropy
                delta_s *= const.kb / const.mp # now erg K^-1 g^-1
                assert self.lint > 0, 'found negative intrinsic luminosity.'
                int_tdsdm = trapz(self.t * delta_s, dx=self.dm)
                dt = -1. *  int_tdsdm / self.lint
                if dt < 0: raise RuntimeError('got negative timestep %f for step %i' % (dt, step))
                # now that have timestep based on total intrinsic luminosity, 
                # calculate eps_grav to see distribution of energy generation
                eps_grav = -1. * self.t * delta_s / dt
                luminosity = np.insert(cumtrapz(eps_grav, dx=self.dm), 0, 0.) # inserting a zero at center point to match length of other profile quantities
                dt_yr = dt / const.secyear
                age_gyr += dt_yr * 1e-9
                
                self.delta_s = delta_s # erg K^-1 g^-1
                self.eps_grav = eps_grav
                self.luminosity = luminosity

                self.history['step'][step] = step
                self.history['iters'][step] = self.iters
                self.history['age'][step] = age_gyr
                self.history['dt_yr'][step] = dt_yr
                self.history['radius'][step] = self.rtot
                self.history['tint'][step] = self.tint
                self.history['lint'][step] = self.lint
                self.history['t1'][step] = self.t1
                self.history['t10'][step] = self.t10
                self.history['teff'][step] = self.teff
                self.history['ysurf'][step] = self.y[-1]
                self.history['nz_gradient'][step] = self.nz_gradient
                self.history['nz_shell'][step] = self.nz_shell
                self.history['mz_env'][step] = self.mz_env
                self.history['mz'][step] = self.mz
                self.history['bulk_z'][step] = self.bulk_z
                
                # estimate of core erosion rate following Guillot+2003 chapter eq. 14. see Moll, Garaud, Mankovich, Fortney ApJ 2017
                pomega = 0.3 # the order-unity factor from integration
                hp_core_top = self.pressure_scale_height[self.kcore + 1]
                r_first_convective_cell = self.r[self.kcore + 1] + hp_core_top # first convective cell extends ~ from core top to this radius
                l1 = self.luminosity[self.r > r_first_convective_cell][0]
                l_core_top = self.luminosity[self.kcore]
                r_core_top = self.r[self.kcore]
                m_core_top = self.m[self.kcore]
                leff = {'first_convective_cell':l1, 'core_top':l_core_top}

                dmcore_dt_guillot = - gammainv_erosion / pomega * self.rtot * leff[luminosity_erosion_option] / const.cgrav / self.mtot # g s^-1
                dmcore_dt_guillot_alternate = - gammainv_erosion / pomega * r_core_top * leff[luminosity_erosion_option] / const.cgrav / m_core_top # g s^-1

                alpha_core_top = - self.hhe_eos.get_rhot(np.log10(self.p[self.kcore]), np.log10(self.t[self.kcore]), self.y[self.kcore]) / self.t[self.kcore] # K^-1
                cp_hhe_core_top = self.hhe_eos.get_cp(np.log10(self.p[self.kcore]), np.log10(self.t[self.kcore]), self.y[self.kcore])
                cp_z_core_top = self.z_eos.get_cp(np.log10(self.p[self.kcore]), np.log10(self.t[self.kcore]))
                cp_core_top = 0.5 *  cp_hhe_core_top + 0.5 * cp_z_core_top # erg g^-1 K^-1
                # print 'cp for gas, z, 50/50 mix', cp_hhe_core_top, cp_z_core_top, cp_core_top
                dmcore_dt_garaud = - gammainv_erosion * alpha_core_top * leff[luminosity_erosion_option] / cp_core_top # g s^-1
                
                self.history['dmcore_dt_guillot'][step] = dmcore_dt_guillot # g s^-1
                self.history['int_dmcore_dt_guillot'][step] = trapz(self.history['dmcore_dt_guillot'][:step], dx=self.history['dt_yr'][1:step]*const.secyear) # g
                self.history['dmcore_dt_guillot_alternate'][step] = dmcore_dt_guillot_alternate
                self.history['int_dmcore_dt_guillot_alternate'][step] = trapz(self.history['dmcore_dt_guillot_alternate'][:step], dx=self.history['dt_yr'][1:step]*const.secyear) # g
                
                self.history['dmcore_dt_garaud'][step] = dmcore_dt_garaud # g s^-1
                self.history['int_dmcore_dt_garaud'][step] = trapz(self.history['dmcore_dt_garaud'][:step], dx=self.history['dt_yr'][1:step]*const.secyear) # g
                
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
                self.profile['z'] = self.z
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
                self.profile['pressure_scale_height'] = self.pressure_scale_height
                
                # these only work if step > 0
                if step > 0:
                    self.profile['delta_s'] = self.delta_s
                    self.profile['eps_grav'] = self.eps_grav
                    self.profile['luminosity'] = self.luminosity
                
                # profile scalars
                self.profile['step'] = step
                self.profile['age'] = age_gyr
                self.profile['nz'] = self.nz
                self.profile['kcore'] = self.kcore
                self.profile['radius'] = self.rtot
                self.profile['tint'] = self.tint
                self.profile['lint'] = self.lint
                self.profile['t1'] = self.t1
                self.profile['t10'] = self.t10
                self.profile['teff'] = self.teff
                self.profile['ysurf'] = self.y[-1]
                self.profile['nz_gradient'] = self.nz_gradient
                self.profile['nz_shell'] = self.nz_shell
                self.profile['mz_env'] = self.mz_env
                self.profile['mz'] = self.mz
                self.profile['bulk_z'] = self.bulk_z
                

                with open('%s%i.profile' % (output_prefix, step), 'w') as f:
                    pickle.dump(self.profile, f, 0) # 0 means dump as text
                                        
                if not keep_going: 
                    print 'stopping.'
                    raise
                
            if step % stdout_interval == 0 or step == nsteps - 1: 
                print '%12i %12i %12.3f %12.3f %12.3e %12.3f %12.3e %12.3f %12i %12i %12.3f %12.3f' % \
                    (step, self.iters, self.t1, self.teff, self.rtot, np.mean(self.entropy[self.entropy > 0]), dt_yr, age_gyr, self.nz_gradient, self.nz_shell, self.y[-1], walltime)

            previous_entropy = self.entropy
            
        if output_prefix:
            with open('%s.history' % output_prefix, 'wb') as fw:
                pickle.dump(self.history, fw, 0)
            print 'wrote history data to %s.history' % output_prefix
            
        return self.history
        
    def smooth(self, array, std, type='flat'):
        '''moving gaussian filter to smooth an array. wrote this to artificially smooth brunt composition term for seismology purposes.'''
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
        
    def basic_profile_plot(self, save_prefix=None):
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
        
    def save_profile(self, outfile, save_gyre_model_with_profile=True,
                    smooth_brunt_n2_std=None, add_rigid_rotation=None, erase_y_discontinuity_from_brunt=False, erase_z_discontinuity_from_brunt=False,
                    omit_brunt_composition_term=False):
            
        # try to be smart about output filenames        
        if '.profile' in outfile and '.gyre' in outfile:
            print 'please pass save_profile an output path with either .gyre or .profile extensions, or neither. not both. please.'
        elif '.profile' in outfile:
            gyre_outfile = outfile.replace('.profile', '.gyre')
        elif '.gyre' in outfile:
            gyre_outfile = outfile
            outfile = gyre_outfile.replace('.gyre', '.profile')
        else:
            gyre_outfile = '%s.gyre' % outfile
            outfile = '%s.profile' % outfile
                        
        if save_gyre_model_with_profile:
            with open(gyre_outfile, 'w') as f:
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
            
                
                if omit_brunt_composition_term:
                    assert not erase_y_discontinuity_from_brunt, 'set omit_brunt_composition_term or erase_y_discontinuity_from_brunt, not both'
                    assert not erase_z_discontinuity_from_brunt, 'set omit_brunt_composition_term or erase_z_discontinuity_from_brunt, not both'
                    brunt_n2_for_gyre_model = np.copy(self.brunt_n2_thermal)
                else:
                    brunt_n2_for_gyre_model = np.copy(self.brunt_n2)                    
            
                if erase_y_discontinuity_from_brunt:
                    assert self.k_shell_top > 0, 'no helium-rich shell, and thus no y discontinuity to erase.'
                    brunt_n2_for_gyre_model[self.k_shell_top + 1] = self.brunt_n2[self.k_shell_top + 2]
            
                if erase_z_discontinuity_from_brunt:
                    assert self.kcore > 0, 'no core, and thus no z discontinuity to erase.'
                    brunt_n2_for_gyre_model = np.copy(self.brunt_n2)
                    if self.kcore == 1:
                        brunt_n2_for_gyre_model[self.kcore] = 0.
                    else:
                        brunt_n2_for_gyre_model[self.kcore] = self.brunt_n2[self.kcore - 1]
            
                if smooth_brunt_n2_std:
                    brunt_n2_for_gyre_model = self.smooth(brunt_n2_for_gyre_model, smooth_brunt_n2_std, type='gaussian')
                
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

                print 'wrote %i zones to %s' % (k, gyre_outfile)
            
        # also write more complete profile info
        with open(outfile, 'w') as f:
                        
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
            
            print 'wrote %i zones to %s' % (k, outfile)
            
            
class EOSError(Exception):
    pass
    
class AtmError(Exception):
    pass
    
class HydroError(Exception):
    pass