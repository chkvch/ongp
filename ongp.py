import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d, splrep, splev
from scipy.integrate import trapz, cumtrapz
import sys
import const
import pickle
import time
try:
    from importlib import reload
except:
    pass
import gp_configs.app_config as app_cfg
import gp_configs.model_config as model_cfg
import logging
import config_const as conf

log = logging.getLogger(__name__)
logging.basicConfig(filename=app_cfg.logfile, filemode='w', format=conf.FORMAT)
log.setLevel(conf.log_level)

class evol:

    def __init__(self, params, **mesh_params):

        if not 'path_to_data' in params.keys():
            raise ValueError('must specify path_to_data indicating path to eos/atm data')

        if not 'hhe_eos_option' in params.keys():
            params['hhe_eos_option'] = 'scvh'
        # initialize hydrogen-helium equation of state
        if params['hhe_eos_option'] == 'scvh':
            import scvh
            self.hhe_eos = scvh.eos(params['path_to_data'])
        elif params['hhe_eos_option'] == 'reos3b':
            import reos3b
            self.hhe_eos = reos3b.eos(params['path_to_data'])
        elif params['hhe_eos_option'] == 'mh13_scvh':
            import mh13_scvh; reload(mh13_scvh)
            self.hhe_eos = mh13_scvh.eos(params['path_to_data'])
        else:
            print('hydrogen-helium eos option {} not recognized'.format(params['hhe_eos_option']))

        if params['z_eos_option']:
            # initialize z equation of state
            import aneos
            import reos_water
            if params['z_eos_option'] == 'reos water':
                self.z_eos = reos_water.eos(params['path_to_data'])
                self.z_eos_low_t = aneos.eos(params['path_to_data'], 'water')
            elif 'aneos' in params['z_eos_option']:
                material = z_eos_option.split()[1]
                self.z_eos = aneos.eos(params['path_to_data'], material)
                self.z_eos_low_t = None
            else:
                raise ValueError("z eos option '%s' not recognized." % params['z_eos_option'])

        # if you're wondering, model atmospheres are initialized in self.static.
        # that way we can run, e.g., a Jupiter and then a Saturn without invoking a new evol instance.

        if 'hhe_phase_diagram' in params.keys():
            if not 'extrapolate_phase_diagram_to_low_pressure' in params.keys():
                params['extrapolate_phase_diagram_to_low_pressure'] = False
            if params['hhe_phase_diagram'] == 'lorenzen':
                import lorenzen
                reload(lorenzen)
                self.phase = lorenzen.hhe_phase_diagram(
                                        params['path_to_data'],
                                        extrapolate_to_low_pressure=params['extrapolate_phase_diagram_to_low_pressure']
                                        )
            elif params['hhe_phase_diagram'] == 'schoettler':
                import schoettler
                reload(schoettler)
                self.phase = schoettler.hhe_phase_diagram()
            else:
                raise ValueError('hydrogen-helium phase diagram option {} is not recognized.'.format(params['hhe_phase_diagram']))

        # defaults for other params
        self.evol_params = {
            'nz':1024,
            'radius_rtol':1e-5,
            'max_iters_static':500,
            'min_iters_static':12
        }
        # overwrite with any passed by user
        for key, value in params.items():
            self.evol_params[key] = value

        # default mesh
        self.mesh_params = {
            'mesh_func_type':'flat_with_surface_exponential_core_gaussian',
            'width_transition_mesh_boost':6e-2,
            'amplitude_transition_mesh_boost':0.1,
            'amplitude_surface_mesh_boost':1e5,
            'width_surface_mesh_boost':1e-2,
            'amplitude_core_mesh_boost':1e1,
            'width_core_mesh_boost':1e-2,
            'fmean_core_bdy_mesh_boost':2e-1
            }
        # overwrite with any passed by user
        for key, value in mesh_params.items():
            self.mesh_params[key] = value

        self.have_rainout = False # until proven guilty
        self.have_rainout_to_core = False

        self.nz_gradient = 0
        self.nz_shell = 0

        # initialize structure variables
        self.nz = self.evol_params['nz']
        self.k = np.arange(self.nz)
        self.p = np.zeros(self.nz)
        self.r = np.zeros(self.nz)
        self.rho = np.zeros(self.nz)
        self.t = np.zeros(self.nz)
        self.y = np.zeros(self.nz)
        self.z = np.zeros(self.nz)


    # mesh function defining mass enclosed within a given zone number
    def mesh_func(self, t, mcore=None):
        # assumes t runs from 0 at center to 1 at surface
        if self.mesh_params['mesh_func_type'] == 'tanh': # old type
            return 0.5 * (1. + np.tanh(10. * (t * np.pi / 2 - np.pi / 4)))
        elif self.mesh_params['mesh_func_type'] == 'flat':
            return t
        elif self.mesh_params['mesh_func_type'] == 'flat_with_surface_exponential':
            f0 = t
            density_f0 = 1. / np.diff(f0)
            density_f0 = np.insert(density_f0, 0, density_f0[0])
            density_f0 += self.mesh_params['amplitude_surface_mesh_boost'] * f0 * np.exp((f0 - 1.) / self.mesh_params['width_surface_mesh_boost']) * np.mean(density_f0)
            out = np.cumsum(1. / density_f0)
            out -= out[0]
            out /= out[-1]
            return out
        elif self.mesh_params['mesh_func_type'] == 'flat_with_surface_exponential_core_gaussian':
            f0 = t
            density_f0 = 1. / np.diff(f0)
            density_f0 = np.insert(density_f0, 0, density_f0[0])
            norm = np.mean(density_f0)
            density_f0 += self.mesh_params['amplitude_surface_mesh_boost'] * f0 * np.exp((f0 - 1.) / self.mesh_params['width_surface_mesh_boost']) * norm
            density_f0 += self.mesh_params['amplitude_core_mesh_boost'] * np.exp(-(f0 - self.mesh_params['fmean_core_bdy_mesh_boost']) ** 2 / self.mesh_params['width_core_mesh_boost']) * norm
            out = np.cumsum(1. / density_f0)
            out -= out[0]
            out /= out[-1]
            return out

        elif self.mesh_params['mesh_func_type'] == 'tanh_with_surface_exponential':
            f0 = 0.5 * (1. + np.tanh(5. * (t * np.pi / 2 - np.pi / 4)))
            density_f0 = 1. / np.diff(f0)
            density_f0 = np.insert(density_f0, 0, density_f0[0])
            density_f0 += self.mesh_params['amplitude_surface_mesh_boost'] * f0 * np.exp((f0 - 1.) / self.mesh_params['width_surface_mesh_boost']) * np.mean(density_f0)
            out = np.cumsum(1. / density_f0)
            out -= out[0]
            out /= out[-1]
            return out
        else:
            raise ValueError('mesh type %s not recognized.' % self.mesh_params['mesh_func_type'])

    def get_rho_z(self, logp, logt):
        '''helper function to get rho of just the z component. different from self.z_eos.get_logrho because
        this does the switching to aneos water at low T if using reos water as the z eos.'''

        assert self.evol_params['z_eos_option'], 'cannot calculate rho_z with no z eos specified.'

        if self.evol_params['z_eos_option'] == 'reos water':
            # if using reos water, extend down to T < 1000 K using aneos water.
            # for this, mask to find the low-T part.
            logp_high_t = logp[logt >= 3.]
            logt_high_t = logt[logt >= 3.]

            logp_low_t = logp[logt < 3.]
            logt_low_t = logt[logt < 3.]

            try:
                rho_z_low_t = 10 ** self.z_eos_low_t.get_logrho(logp_low_t, logt_low_t)
            except:
                print('off low-t eos tables?')
                raise

            try:
                rho_z_high_t = 10 ** self.z_eos.get_logrho(logp_high_t, logt_high_t)
            except:
                print('off high-t eos tables?')
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
        if np.any(np.isnan(logp)):
            raise EOSError('have %i nans in logp' % len(logp[np.isnan(logp)]))
        elif np.any(np.isnan(logt)):
            raise EOSError('have %i nans in logt' % len(logt[np.isnan(logt)]))
        elif np.any(y <= 0.):
            raise UnphysicalParameterError('one or more bad y')
        elif np.any(y >= 1.):
            raise UnphysicalParameterError('one or more bad y')
        elif np.any(z < 0.):
            raise UnphysicalParameterError('one or more bad z')
        elif np.any(z > 1.):
            raise UnphysicalParameterError('one or more bad z')
        rho_hhe = 10 ** self.hhe_eos.get_logrho(logp, logt, y)
        rho_z = self.get_rho_z(logp, logt)
        rhoinv = (1. - z) / rho_hhe + z / rho_z
        return rhoinv ** -1

    def static(self, params):

        '''build a hydrostatic model with a given total mass mtot, 1-bar temperature t1, envelope helium mass fraction yenv,
            envelope heavy element mass fraction zenv, and heavy-element core mass mcore. returns the number of iterations taken before
            convergence, or -1 for failure to converge.'''

        if type(params['mtot']) is str:
            if params['mtot'][0] == 'j':
                params['mtot'] = const.mjup
            elif params['mtot'][0] == 's':
                params['mtot'] = const.msat
            elif params['mtot'][0] == 'u':
                params['mtot'] = const.mura
            elif params['mtot'][0] == 'n':
                params['mtot'] = const.mnep
            else:
                raise ValueError("if type(params['mtot']) is str, first element must be j, s, u, n")
        self.mtot = params['mtot']
        if ('t1' in params.keys()) and not ('t10' in params.keys()):
            self.atm_which_t = 't1'
            self.t1 = params['t1']
        elif ('t10' in params.keys()) and not ('t1' in params.keys()):
            self.atm_which_t = 't10'
            self.t10 = params['t10']
        else:
            raise ValueError('must specify one and only one of t1 or t10.')

        # model atmospheres
        atm_type, atm_planet = self.evol_params['atm_option'].split() # e.g., 'f11_tables u'
        if 'teq' in params.keys():
            self.teq = params['teq']
        # else:
            # use a default value.
            # J is a mean of Hanel et al. 1981 and Pearl & Conrath 1991.
            # S i'll need to check.
            # U, N come from Pearl & Conrath 1991.
            # self.teq = {'jup':109., 'sat':81.3, 'u':58.2, 'n':46.6}[atm_planet]

            # better: don't set. in set_atm, if no self.teq, then instead of calculating
            # teff from teff^4 = tint^4 + teq^4, just get teff directly from model
            # atmospheres.

        if atm_type == 'f11_tables':
            import f11_atm; reload(f11_atm)
            if 'atm_jupiter_modified_teq' in list(self.evol_params):
                self.atm = f11_atm.atm(self.evol_params['path_to_data'], atm_planet,
                    jupiter_modified_teq=self.evol_params['atm_jupiter_modified_teq'])
            else:
                self.atm = f11_atm.atm(self.evol_params['path_to_data'], atm_planet)
        elif atm_type == 'f11_fit':
            import f11_atm_fit
            self.atm = f11_atm_fit.atm(atm_planet)
        else:
            raise ValueError('atm_type %s not recognized.' % atm_type)

        if 'yenv' in params.keys():
            params['y1'] = params.pop('yenv')
        if 'zenv' in params.keys():
            params['z1'] = params.pop('zenv')

        self.z1 = params['z1']
        if 'z2' in params.keys():
            self.z2 = params['z2']
        else:
            self.z2 = None

        self.y1 = params['y1']
        if 'y2' in params.keys():
            self.y2 = params['y2']
        else:
            self.y2 = None

        self.mz_env = None
        self.mz = None
        self.bulk_z = None

        if not 'phase_t_offset' in params.keys():
            params['phase_t_offset'] = 0.
        if not 'minimum_y_is_envelope_y' in params.keys():
            minimum_y_is_envelope_y = params['minimum_y_is_envelope_y'] = False
        if not 'erase_z_discontinuity_from_brunt' in params.keys():
            erase_z_discontinuity_from_brunt = params['erase_z_discontinuity_from_brunt'] = False
        if not 'include_core_entropy' in params.keys():
            params['include_core_entropy'] = False
        if not 'core_prho_relation' in params.keys():
            params['core_prho_relation'] = None
        if not 'verbose' in params.keys():
            params['verbose'] = False
        if not 'rainout_verbosity' in params.keys():
            params['rainout_verbosity'] = 0

        # save params passed to static
        self.static_params = params

        # initialize lagrangian mesh.
        if not 'mcore' in params.keys(): params['mcore'] = 0.
        mcore = params['mcore']
        assert mcore * const.mearth < self.mtot, 'core mass must be (well) less than total mass.'
        if mcore > 0.:
            t = np.linspace(0, 1, self.nz - 1)
            if self.mesh_params['mesh_func_type'] == 'flat_with_surface_exponential_core_gaussian':
                self.m = self.mtot * self.mesh_func(t, mcore=mcore) # passes mcore so that core-boundary-mesh-boost coincides with core boundary.
            else:
                self.m = self.mtot * self.mesh_func(t)
            self.kcore = kcore = np.where(self.m >= mcore * const.mearth)[0][0] # kcore - 1 is last zone with m < mcore
            self.m = np.insert(self.m, self.kcore, mcore * const.mearth) # kcore is the zone where m == mcore. this zone should have z=1.
            self.kcore += 1 # so say self.rho[:kcore] wil encompass all the zones with z==1.
        elif mcore == 0.: # no need for extra precautions
            t = np.linspace(0, 1, self.nz)
            self.m = self.mtot * self.mesh_func(t) # grams
            self.m *= self.mtot / self.m[-1] # guarantee surface zone has mtot enclosed
            self.kcore = 0
        else:
            raise UnphysicalParameterError('bad core mass %g' % mcore)
        self.mcore = mcore
        self.dm = np.diff(self.m)

        self.grada = np.zeros_like(self.m)

        # first guess, values chosen just so that densities will be calculable
        self.p[:] = 1e12
        self.t[:] = 1e4

        # set initial composition information. for now, envelope is homogeneous
        self.y[:] = 0.
        self.y[self.kcore:] = self.y1

        self.z[:self.kcore] = 1.
        assert self.z1 >= 0., 'got negative z1 %g' % self.z1
        self.z[self.kcore:] = self.z1

        if not 'transition_pressure' in params.keys():
            params['transition_pressure'] = 1.

        self.iters = 0

        # get density everywhere based on primitive guesses
        self.set_core_density()
        self.set_envelope_density(ignore_z=True) # ignore Z for first pass at densities

        self.integrate_continuity() # rho, dm -> r
        self.integrate_hydrostatic() # m, r -> p # p[-1] hardcoded to 1 bar
        self.integrate_temperature(brute_force_loop=True) # p, t, y -> grada -> t

        # now that we have pressure, try and locate transition pressure
        self.locate_transition_pressure()
        # set y2 and z2 if applicable
        self.set_yz()

        # recalculate densities based on possibly three-layer y, z
        self.set_core_density()
        self.set_envelope_density()

        # these used to be defined after iterations were completed, but they are needed for calculation
        # of brunt_b to allow superadiabatic regions with grad-grada proportional to brunt_b.
        self.gradt = np.zeros_like(self.p)
        self.brunt_b = np.zeros_like(self.p)
        self.chirho = np.zeros_like(self.p)
        self.chit = np.zeros_like(self.p)
        self.chiy = np.zeros_like(self.p)
        self.grady = np.zeros_like(self.p)

        # helium rain bookkeeping
        self.mhe = np.dot(self.y[1:], self.dm) # initial total he mass
        self.k_shell_top = 0 # until a shell is found by equilibrium_y_profile

        # relax to hydrostatic
        last_three_radii = 0, 0, 0
        for iteration in range(self.evol_params['max_iters_static']):
            self.iters += 1
            self.integrate_hydrostatic() # integrate momentum equation to get pressure
            self.locate_transition_pressure() # find point that should be discontinuous in y and z, if any
            # set y and z profile assuming three-layer homogeneous. if doing helium rain (self.phase is set),
            # Y profile wile be set appropriately later in equilibrium_Y iterations.
            self.set_yz()
            if self.iters < 5:
                self.integrate_temperature(brute_force_loop=True) # integrate gradt (usually grada) for envelope temp; core isothermal
            else:
                self.integrate_temperature(brute_force_loop=False)
            self.set_core_density()
            self.set_envelope_density()
            self.integrate_continuity() # get zone radii from their densities via continuity equation

            if 'debug_iterations' in params.keys():
                if params['debug_iterations']:
                    print('iter=', self.iters, 'rtot=%g'%self.r[-1])

            # hydrostatic model is judged to be converged when the radius has changed by a relative amount less than
            # radius_rtol over both of the last two iterations.
            if np.all(np.abs(np.mean((last_three_radii / self.r[-1] - 1.))) < self.evol_params['radius_rtol']):
                if iteration >= self.evol_params['min_iters_static']:
                    break
            if not np.isfinite(self.r[-1]):
                raise HydroError('found infinite total radius.')

            last_three_radii = last_three_radii[1], last_three_radii[2], self.r[-1]

        else:
            return -1

        if params['verbose']: print('converged homogeneous model after %i iterations.' % self.iters)

        if hasattr(self, 'phase'): # repeat hydro iterations, now including the phase diagram calculation (if modeling helium rain)
            last_three_radii = 0, 0, 0
            for iteration in range(self.evol_params['max_iters_static']):
                self.iters_immiscibility = iteration + 1

                self.integrate_hydrostatic()
                self.integrate_temperature() # sets grada for the last time

                if 'rrho_where_have_helium_gradient' in params.keys() and self.have_rainout:
                    # allow helium gradient regions to have superadiabatic temperature stratification.
                    # this is new here -- copied from below where we'd ordinarily only compute this after we have a converged model.
                    # might cost some time because of the additional eos calls.

                    # as is, brunt_b only accounts for hydrogen/helium gradients. extending to include more
                    # species is straightforward: chiy*grady becomes sum_i^{N-1}(chi_i * grad X_i) where sum_i^N(X_i)=1
                    res = self.hhe_eos.get(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
                    self.chit[self.kcore:] = res['chit']
                    self.chirho[self.kcore:] = res['chirho']
                    self.chiy[self.kcore:] = res['chiy']
                    self.grady[self.kcore+1:] = np.diff(np.log(self.y[self.kcore:])) / np.diff(np.log(self.p[self.kcore:]))
                    self.grady[self.k_shell_top+1] = 0.
                    self.brunt_b[self.kcore+1:] = self.chirho[self.kcore+1:] / self.chit[self.kcore+1:] * self.chiy[self.kcore+1:] * self.grady[self.kcore+1:]
                    # self.brunt_b[self.k_shell_top + 1] = 0.

                    assert np.all(self.brunt_b[self.kcore+1:self.k_shell_top+1] == 0) # he shell itself should be uniform

                    self.gradt += params['rrho_where_have_helium_gradient'] * self.brunt_b

                    self.integrate_temperature(adiabatic=False)

                self.y = self.equilibrium_y_profile(params['phase_t_offset'],
                            verbosity=params['rainout_verbosity'],
                            minimum_y_is_envelope_y=params['minimum_y_is_envelope_y'],
                            ptrans=params['transition_pressure'])

                self.set_core_density()
                self.set_envelope_density()
                self.integrate_continuity()

                if np.all(np.abs(np.mean((last_three_radii / self.r[-1] - 1.))) < self.evol_params['radius_rtol']):
                    break

                if not np.isfinite(self.r[-1]):
                    with open('output/found_infinite_radius.dat', 'w') as f:
                        f.write('%12s %12s %12s %12s %12s %12s\n' % ('core', 'p', 't', 'rho', 'm', 'r'))
                        for k in range(self.nz):
                            in_core = k < self.kcore
                            f.write('%12s %12g %12g %12g %12g %12g\n' % (in_core, self.p[k], self.t[k], self.rho[k], self.m[k], self.r[k]))
                    print('saved output/found_infinite_radius.dat')
                    raise RuntimeError('found infinite total radius')
                last_three_radii = last_three_radii[1], last_three_radii[2], self.r[-1]
            else:
                return -1

            # if verbose: print('converged with new Y profile after %i iterations.' % self.iters_immiscibility)

        # finalize hydrostatic profiles (maybe not necessary since we just did 20 iterations)
        self.integrate_hydrostatic()
        self.set_core_density()
        self.set_envelope_density()

        # finally, calculate lots of auxiliary quantities of interest
        self.rtot = self.r[-1]

        self.set_atm() # make sure t10 is set; use (t10, g) to get (tint, teff) from model atmosphere
        self.set_entropy() # set entropy profile (necessary for an evolutionary calculation)
        # try:
            # self.set_derivatives_etc() # calculate thermo derivatives, seismology quantities, g, etc.
        # except:
            # pass
        self.set_derivatives_etc() # calculate thermo derivatives, seismology quantities, g, etc.

        return self.rtot, self.t[-1], self.teff






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

    def equilibrium_y_profile(self, phase_t_offset, verbosity=0, show_timing=False, minimum_y_is_envelope_y=False, ptrans=None):
        '''uses the existing p-t profile to find the thermodynamic equilibrium y profile, which
        may require a helium-rich layer atop the core.'''
        p = self.p * 1e-12 # Mbar
        t = self.t * 1e-3 # kK
        k1 = np.where(p > ptrans)[0][-1]

        # the phase diagram came from a model for a system of just H and He.
        # if xp and Yp represent the helium number fraction and mass fraction from the phase diagram,
        # this would correspond to xp > x and Yp > Y where x, Y are the fractions in a real model
        # that also includes a Z component.
        # given Z, set Y from Yp according to
        #   X + Y + Z = 1, and
        #   Y / X = Yp / Xp.
        # these combine to
        #   Y = (1 - Z) / (1 + (1 - Yp) / Yp).

        get_xp = lambda yp: 1. / (1. + 4. * (1. - yp) / yp)
        get_yp = lambda xp: 1. / (1. + (1. - xp) / 4. / xp)

        get_y = lambda z, yp: (1. - z) / (1. + (1. - yp) / yp)

        if verbosity > 0: print('rainout iteration {}'.format(self.iters_immiscibility))

        # if t_offset is positive, immiscibility sets in sooner.
        # i.e., query phase diagram with a lower T than true planet T.
        xplo, xphi = self.phase.miscibility_gap(p[k1], t[k1] - phase_t_offset * 1e-3)
        if type(xplo) is str:
            if xplo == 'stable':
                if verbosity > 0: print('first zone with p>ptrans=%1.1f Mbar is stable to demixing\n' % (ptrans))
                return self.y
            elif xplo == 'failed':
                raise ValueError('failed to get miscibility gap in initial loop over zones. off phase diagram? p=%.1f, t = %.1f, t-t_offset=%.1f' % (p[k1], self.t[k1], self.t[k1] - phase_t_offset))
        elif type(xplo) is np.float64:
            ymax1 = get_y(self.z[k1-1], get_yp(xplo)) # self.z[k1] is z1. self.z[k1-1] is z2.
            if self.y[k1] < ymax1:
                if verbosity > 0: print('first zone with p>ptrans=%1.1f Mbar is stable to demixing. t_offset=%f K, y=%1.4f, ymax=%1.4f\n' % (ptrans, phase_t_offset,  self.y[k1], ymax1))
                return self.y

        self.ystart = np.copy(self.y)
        yout = np.copy(self.y)
        yout[k1:] = get_y(self.z[k1], get_yp(xplo)) # homogeneous molecular envelope at this abundance
        if verbosity > 0: print('demix', k1, self.m[k1] / self.m[-1], p[k1], self.t[k1], 'env', self.y[k1], '-->', yout[k1])

        t0 = time.time()
        rainout_to_core = False

        self.k_gradient_top = k1
        for k in np.arange(k1-1, self.kcore, -1): # inward from first point where p > ptrans
            t1 = time.time()
            # note that phase_t_offset is applied here; phase diagram simply sees different temperature
            xplo, xphi = self.phase.miscibility_gap(p[k], t[k] - phase_t_offset * 1e-3)
            if type(xplo) is str:
                if xplo == 'stable':
                    if verbosity > 1: print('stable', k, self.m[k] / self.m[-1], p[k], self.t[k], yout[k])
                    break
                elif xplo == 'failed':
                    raise ValueError('failed to get miscibility gap in initial loop over zones. p, t = %f Mbar, %f K' % (p[k], self.t[k]))
            elif type(xplo) is np.float64:
                ymax = get_y(self.z[k], get_yp(xplo))
                if yout[k] < ymax:
                    if verbosity > 1: print('stable', k, self.m[k] / self.m[-1], p[k], self.t[k], yout[k], ' < ', ymax)
                    break
            else:
                raise TypeError('got unexpected type for xplo from phase diagram', type(xplo))
            if show_timing: print('zone %i: dt %f ms, t0 + %f seconds' % (k, 1e3 * (time.time() - t1), time.time() - t0))

            self.k_shell_top = 0
            ystart = yout[k]
            yout[k] = ymax

            if minimum_y_is_envelope_y and yout[k] < yout[k+1]:
                yout[k:] = yout[k]

            # difference between initial he mass and current proposed he mass above and including this zone
            # must be deposited into the deeper interior.
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
                if verbosity > 1: print('would need Y > 1 in inner homog region; rainout to core.')
                rainout_to_core = True
                yout[self.kcore:k] = 0.95 # since this is < 1., should still have an overall 'missing' helium mass in envelope, to be made up during outward shell iterations.
                # assert this "should"
                # assert np.dot(yout[self.kcore:], self.dm[self.kcore-1:]) < self.mhe, 'problem: proposed envelope already has mhe > mhe_initial, even before he shell iterations. case: y_interior = %f > 1. kcore=%i, k=%i' % (y_interior, self.kcore, k)
                kbot = k
                break
            else:
                # all good; uniformly distribute all helium that has rained out from above into deeper interior
                if verbosity > 2: print('{:>5n} {:>8.4f} {:>10.6f} {:>10.6f}'.format(k, p[k], yout[k], y_interior))
                yout[self.kcore:k] = y_interior



        if verbosity > 0: print('rainout to core %s' % rainout_to_core)
        if show_timing: print('t0 + %f seconds' % (time.time() - t0))

        if rainout_to_core:
            # gradient extends down to kbot, below which the rest of the envelope is already set Y=0.95.
            # since proposed envelope mhe < initial mhe, must grow the He-rich shell to conserve total mass.
            if verbosity > 1: print('%5s %5s %10s %10s %10s' % ('k', 'kcore', 'dm_k', 'mhe_tent', 'mhe'))
            for k in np.arange(kbot, self.nz):
                yout[k] = 0.95 # in the future, obtain value from ymax_rhs(p, t)
                tentative_total_he_mass = np.dot(yout[self.kcore:-1], self.dm[self.kcore:])
                if verbosity > 1: print('%5i %5i %10.4e %10.4e %10.4e' % (k, self.kcore, self.dm[k-1], tentative_total_he_mass, self.mhe))
                if tentative_total_he_mass >= self.mhe:
                    if verbosity > 1: print('tentative he mass, initial total he mass', tentative_total_he_mass, self.mhe)
                    rel_mhe_error = abs(self.mhe - tentative_total_he_mass) / self.mhe
                    if verbosity > 1: print('satisfied he mass conservation to a relative precision of %f' % rel_mhe_error)
                    # yout[k] = (self.mhe - (np.dot(yout[self.kcore:], self.dm[self.kcore-1:]) - yout[k] * self.dm[k-1])) / self.dm[k-1]
                    self.k_shell_top = k
                    break

        self.have_rainout = self.nz_gradient > 0
        self.have_rainout_to_core = rainout_to_core
        if rainout_to_core: assert self.k_shell_top
        if verbosity > 0: print()

        self.nz_gradient = len(np.where(np.diff(yout) < 0)[0])
        self.nz_shell = max(0, self.k_shell_top - self.kcore)

        return yout

    def set_core_density(self):
        if self.kcore == 0:
            return

        if self.static_params['core_prho_relation']:
            if self.static_params['core_prho_relation'] == 'hm89 rock':
                self.rho[:self.kcore] = 8 # just initial guess for root find
                self.rho[:self.kcore] = self.get_rhoz_hm89_rock(self.p[:self.kcore], self.rho[:self.kcore])
            elif self.static_params['core_prho_relation'] == 'hm89 ice':
                self.rho[:self.kcore] = 8
                self.rho[:self.kcore] = self.get_rhoz_hm89_ice(self.p[:self.kcore], self.rho[:self.kcore])
            else:
                raise ValueError("core_prho_relation must be one of 'hm89 rock' or 'hm89 ice'.")
        else:
            assert self.evol_params['z_eos_option'], 'cannot calculate rho_z if no z_eos_option specified'
            self.rho[:self.kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))

    def set_envelope_density(self, ignore_z=False):
        if ignore_z or self.z[-1] == 0.: # XY envelope
            self.rho[self.kcore:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
        else: # XYZ envelope
            self.rho[self.kcore:] = self.get_rho_xyz(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:], self.z[self.kcore:])
        self.rho_check_nans()

    def integrate_continuity(self):
        q = np.zeros_like(self.rho)
        q[0] = 0.
        q[1:] = 3. * self.dm / 4 / np.pi / self.rho[1:]
        self.r = np.cumsum(q) ** (1. / 3)

    def integrate_hydrostatic(self):
        dp = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4
        if self.atm_which_t == 't1':
            psurf = 1e6
        elif self.atm_which_t == 't10':
            psurf = 1e7
        else:
            raise ValueError('atm_which_t option %s not recognized.' % self.atm_which_t)
        self.p[-1] = psurf
        self.p[:-1] = psurf + np.cumsum(dp[::-1])[::-1]

        if np.any(np.isnan(self.p)):
            raise EOSError('%i nans in pressure after integrate hydrostatic on static iteration %i.' % (len(self.p[np.isnan(self.p)]), self.iters))


    def integrate_temperature(self, adiabatic=True, brute_force_loop=False):
        '''
        set surface temperature and integrate the adiabat inward to get temperature.
        adiabatic gradient here (as well as later) is ignoring any z contribution.
        if not adiabatic, then just integrate existing gradt
        '''
        if self.atm_which_t == 't1':
            self.t[-1] = self.t1
        elif self.atm_which_t == 't10':
            self.t[-1] = self.t10
        else:
            raise ValueError("atm_which_t must be one of 't1', 't10'")
        if adiabatic:
            self.grada[:self.kcore] = 0. # ignore because doesn't play a role in the temperature structure (core is isothermal).
            # grada will in general still be set inside the core from the Z eos if possible, after a static model is converged.

            # this call to get grada is slow.
            # next round of optimization should make sure we are carrying out the minimum number of eos calls because each one
            # relies so heavily on interpolation in the hhe eos.
            self.grada[self.kcore:] = self.hhe_eos.get_grada(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
            self.grada_check_nans()
            self.gradt = np.copy(self.grada) # may be modified later if include_he_immiscibility and rrho_where_have_helium_gradient
            if brute_force_loop:
                for k in np.arange(self.nz)[::-1]: # surface to center
                    if k == self.nz - 1: continue
                    if k == self.kcore - 1: break
                    dlnp = np.log(self.p[k]) - np.log(self.p[k+1])
                    dlnt = self.grada[k] * dlnp
                    self.t[k] = self.t[k+1] * (1. + dlnt)
            else: # slices + cumsum does at least a factor of a few faster
                dlnp = np.diff(np.log(self.p))
                dlnt = self.grada[:-1] * dlnp
                dt = dlnt * self.t[1:]
                self.t[self.kcore:-1] = self.t[-1] - np.cumsum(dt[self.kcore:][::-1])[::-1]

        else:
            for k in np.arange(self.nz)[::-1]:
                if k == self.nz - 1: continue
                if k == self.kcore - 1: break
                dlnp = np.log(self.p[k]) - np.log(self.p[k+1])
                dlnt = self.gradt[k] * dlnp # gradt, not grada
                self.t[k] = self.t[k+1] * (1. + dlnt)
                if np.isinf(self.t[k]):
                    raise RuntimeError('got infinite temperature in integration of non-ad gradt. k %i, gradt[k+1] %i, brunt_b[k] %g' % (k, self.gradt[k+1], self.brunt_b[k]))

        self.t[:self.kcore] = self.t[self.kcore] # core is isothermal at temperature of core-mantle boundary

        if np.any(np.isnan(self.t)):
            raise EOSError('%i nans in temperature after integrate gradt on static iteration %i.' % (len(self.t[np.isnan(self.t)]), self.iters))

    def locate_transition_pressure(self):
        '''
            identify some transition pressure between regions in the envelope.
            this could be the H molecular-metallic transition for mtot comparable to a Saturn mass.
            this could be the gaseous-icy transition for ice giant masses.
        '''
        try:
            # add + 1 such that self.z[self.ktrans:] = self.z1, consistent with self.set_yz below
            self.ktrans = np.where(self.p >= self.static_params['transition_pressure'] * 1e12)[0][-1] + 1
        except IndexError:
            if self.mtot > 0.5 * const.msat:
                # might fail to find this transition if current pressure guess is still poor.
                # the model will have this transition in the end; just make a rough guess for ktrans for now.
                # not too important where it goes for first couple iterations.
                if self.iters < 2:
                    self.ktrans = int(2. * self.nz / 3)
                else:
                    raise HydroError('found no molecular-metallic transition. ptrans %.3e, max p %.3e, iters %i ' \
                        % (self.static_params['transition_pressure']*1e12, max(self.p), self.iters))
            else: # skip for U, N
                self.ktrans = -1

    def set_yz(self):
        '''
        set y and z of inner and outer envelope.
        '''
        if self.ktrans == 0:
            print('warning: got ktrans == 0')
            return
        elif self.ktrans == -1: # no transition found (yet)
            return
        if self.z2: # two-layer envelope in terms of Z distribution. zenv is z of the outer envelope, z2 is z of the inner envelope
            try:
                assert self.z2 > 0, 'if you want a z-free envelope, no need to specify z2.'
                assert self.z2 < 1., 'set_yz got bad z %f' % self.z2
                assert self.z2 >= self.z1, 'no z inversion allowed.'
            except AssertionError as e:
                raise UnphysicalParameterError(e.args[0])
            self.z[self.kcore:self.ktrans] = self.z2
            self.z[self.ktrans:] = self.z1
        if self.y2:
            try:
                assert self.y2 > 0, 'if you want a Y-free envelope, no need to specify y2.'
                assert self.y2 < 1., 'set_yz got bad y %f' % self.y2
                assert self.y2 >= self.y1, 'no y inversion allowed.'
            except AssertionError as e:
                raise UnphysicalParameterError(e.args[0])
            self.y[self.kcore:self.ktrans] = self.y2
            self.y[self.ktrans:] = self.y1

        self.envelope_mean_y = np.dot(self.dm[self.kcore:], self.y[self.kcore:-1]) / np.sum(self.dm[self.kcore:])


    def grada_check_nans(self):
        # a nan might appear in grada if a p, t point is just outside the original tables.
        # e.g., this was happening at logp, logt = 11.4015234804 3.61913879612, just under
        # the right-hand side "knee" of available data.
        if np.any(np.isnan(self.grada)):
            num_nans = len(self.grada[np.isnan(self.grada)])

            # raise EOSError('%i nans in grada. first (logT, logP)=(%f, %f); last (logT, logP) = (%f, %f)' % \
            #     (num_nans, np.log10(self.t[np.isnan(self.grada)][0]), np.log10(self.p[np.isnan(self.grada)][0]), \
            #     np.log10(self.t[np.isnan(self.grada)][-1]), np.log10(self.p[np.isnan(self.grada)][-1])))

            if self.iters < 5 and len(self.grada[np.isnan(self.grada)]) < self.nz / 4:
                '''early in iterations and fewer than nz/4 nans; attempt to coax grada along.

                seems more of a problem with large transition_pressure.

                really not a big deal if we invent some values for grada this early in iterations since
                many more iterations will follow.
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
                print('%i nans in grada for iteration %i, stopping' % (num_nans, self.iters))
                with open('grada_nans.dat', 'w') as fw:
                    for k, val in enumerate(self.grada):
                        if np.isnan(val):
                            fw.write('%16.8f %16.8f %16.8f\n' % (np.log10(self.p[k]), np.log10(self.t[k]), self.y[k]))
                print('saved problematic logp, logt, y to grada_nans.dat')
                raise EOSError('%i nans in grada after eos call on static iteration %i.' % (len(self.grada[np.isnan(self.grada)]), self.iters))

    def rho_check_nans(self):
        if np.any(np.isnan(self.rho)):
            if self.iters < 5: # try and coax along by connecting the dots
                where_nans = np.where(np.isnan(self.rho))
                k_first_nan = where_nans[0][0]
                k_last_nan = where_nans[0][-1]
                last_good_rho = self.rho[k_first_nan - 1]
                first_good_rho = self.rho[k_last_nan + 1]
                self.rho[k_first_nan:k_last_nan+1] = (self.r[k_first_nan:k_last_nan+1] - self.r[k_first_nan]) \
                                                        / (self.r[k_last_nan+1] - self.r[k_first_nan]) \
                                                        * (first_good_rho - last_good_rho) + last_good_rho
            else:
                with open('rho_nans.dat', 'w') as fw:
                    for k, val in enumerate(self.rho):
                        if np.isnan(val):
                            fw.write('%16.8f %16.8f %16.8f %16.8f\n' % (np.log10(self.p[k]), np.log10(self.t[k]), self.y[k], self.z[k]))
                print('saved problematic logp, logt, y, z to rho_nans.dat')
                raise EOSError('%i nans in rho after eos call on static iteration %i.' % (len(self.rho[np.isnan(self.rho)]), self.iters))

    def set_atm(self):

        if self.atm_which_t == 't1': # interpolate in existing t profile to find t10
            assert self.t1 == self.t[-1] # this should be true by construction (see integrate_temperature)
            k10 = np.where(self.p < 1e7)[0][0]
            tck = splrep(self.p[k10-5:k10+5][::-1], self.t[k10-5:k10+5][::-1], k=3) # cubic
            self.t10 = np.float64(splev(1e7, tck))
            assert self.t1 > 0., 'bad t1 %g' % self.t1
        elif self.atm_which_t == 't10':
            assert self.t10 == self.t[-1] # this should be true by construction (see integrate_temperature)
            self.t1 = -1
        assert self.t10 > 0., 'bad t10 %g' % self.t10

        # look up intrinsic temperature, effective temperature, intrinsic luminosity from atm module. that module takes g in mks.
        self.surface_g = const.cgrav * self.mtot / self.r[-1] ** 2 # in principle different for 1-bar vs. 10-bar surface, but negligible
        if self.evol_params['atm_option'].split()[0] == 'f11_tables':
            if self.surface_g * 1e-2 > max(self.atm.g_grid):
                raise AtmError('surface gravity too high for atm tables. value = %g, maximum = %g' % (self.surface_g*1e-2, max(self.atm.g_grid)))
            elif self.surface_g * 1e-2 < min(self.atm.g_grid):
                raise AtmError('surface gravity too low for atm tables. value = %g, minimum = %g' % (self.surface_g*1e-2, min(self.atm.g_grid)))
        try:
            if hasattr(self, 'teq'):
                self.tint = self.atm.get_tint(self.surface_g * 1e-2, self.t10) # Fortney+2011 needs g in mks
                self.teff = (self.tint ** 4 + self.teq ** 4) ** (1. / 4)
            else:
                self.tint, self.teff = self.atm.get_tint_teff(self.surface_g * 1e-2, self.t10)
            self.lint = 4. * np.pi * self.rtot ** 2 * const.sigma_sb * self.tint ** 4
        except ValueError as e:
            if self.z2:
                print('z2, z1 = ', self.z2, self.z1)
            else:
                print('z1 = ', self.z1)
            if 'f(a) and f(b) must have different signs' in e.args[0]:
                raise AtmError('atm.get_tint failed to bracket solution for root find. g=%g, t10=%g' % (self.surface_g*1e-2, self.t10))
            else:
                raise AtmError('unspecified atm error for g=%g, t10=%g: %s' % (self.surface_g*1e-2, self.t10, e.args[0]))

    def set_derivatives_etc(self):
        self.r[0] = 1. # 1 cm central radius to keep these things at least calculable at center zone
        self.g = const.cgrav * self.m / self.r ** 2
        self.g[0] = self.g[1] # hack so that we don't get infs in, e.g., pressure scale height. won't effect anything

        # this is a structure derivative, not a thermodynamic one. wherever the profile is a perfect adiabat, this is also gamma1.
        self.dlogp_dlogrho = np.diff(np.log(self.p)) / np.diff(np.log(self.rho))

        self.gamma1 = np.zeros_like(self.p)
        self.csound = np.zeros_like(self.p)
        self.gradt_direct = np.zeros_like(self.p)

        if self.kcore > 0 and self.evol_params['z_eos_option']: # compute gamma1 in core
            self.gamma1[:self.kcore] = self.z_eos.get_gamma1(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
        self.gamma1[self.kcore:] = self.hhe_eos.get_gamma1(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
        self.csound = np.sqrt(self.p / self.rho * self.gamma1)
        self.lamb_s12 = 2. * self.csound ** 2 / self.r ** 2 # lamb freq. squared for l=1

        self.delta_nu = (2. * trapz(self.csound ** -1, x=self.r)) ** -1 * 1e6 # the large frequency separation in uHz
        self.delta_nu_env = (2. * trapz(self.csound[self.kcore:] ** -1, x=self.r[self.kcore:])) ** -1 * 1e6

        dlnp_dlnr = np.diff(np.log(self.p)) / np.diff(np.log(self.r))
        dlnrho_dlnr = np.diff(np.log(self.rho)) / np.diff(np.log(self.r))

        if self.kcore > 0:
            try:
                self.chirho[:self.kcore] = self.z_eos.get_chirho(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
                self.chit[:self.kcore] = self.z_eos.get_chit(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
                self.grada[:self.kcore] = (1. - self.chirho[:self.kcore] / self.gamma1[:self.kcore]) / self.chit[:self.kcore] # e.g., Unno's equations 13.85, 13.86
            except AttributeError:
                print("warning: z_eos_option '%s' does not provide methods for get_chirho and get_chit." % self.evol_params['z_eos_option'])
                print('cannot calculate things like grada in core and so this model may not be suited for eigenmode calculations.')
                pass

        # ignoring the envelope z component when it comes to calculating chirho and chit
        res = self.hhe_eos.get(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:]) # a final H/He eos call
        self.chirho[self.kcore:] = res['chirho']
        self.chit[self.kcore:] = res['chit']
        self.gamma1[self.kcore:] = res['gamma1']
        self.gradt_direct[:self.kcore] = 0. # was previously self.gradt
        self.gradt_direct[self.kcore+1:] = np.diff(np.log(self.t[self.kcore:])) / np.diff(np.log(self.p[self.kcore:])) # was previously self.gradt

        # terms needed for calculation of the composition term brunt_B.
        self.grady[self.kcore+1:] = np.diff(np.log(self.y[self.kcore:])) / np.diff(np.log(self.p[self.kcore:]))
        if self.k_shell_top:
            self.grady[self.k_shell_top+1] = 0.
        self.grady[np.where(np.isnan(self.grady))] = 0.
        # this is the thermodynamic derivative (const P and T), for H-He.
        self.dlogrho_dlogy = np.zeros_like(self.p)
        try:
            self.dlogrho_dlogy[self.kcore:] = res['chiy']
        except:
            pass

        self.brunt_n2_direct = np.zeros_like(self.p)
        self.brunt_n2_direct[1:] = self.g[1:] / self.r[1:] * (dlnp_dlnr / self.gamma1[1:] - dlnrho_dlnr)

        # other forms of the BV frequency
        self.homology_v = const.cgrav * self.m * self.rho / self.r / self.p
        self.brunt_n2_unno_direct = np.zeros_like(self.p)
        self.brunt_n2_unno_direct[self.kcore+1:] = self.g[self.kcore+1:] * self.homology_v[self.kcore+1:] / self.r[self.kcore+1:] * \
            (self.dlogp_dlogrho[self.kcore:] ** -1. - self.gamma1[self.kcore+1:] ** -1.) # Unno 13.102

        rho_z = np.zeros_like(self.p)
        rho_hhe = np.zeros_like(self.p)
        if np.any(self.z > 0.) and self.evol_params['z_eos_option']:
            rho_z[self.z > 0.] = self.get_rho_z(np.log10(self.p[self.z > 0.]), np.log10(self.t[self.z > 0.]))
        # rho_hhe[self.y > 0.] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[self.y > 0.]), np.log10(self.t[self.y > 0.]), self.y[self.y > 0.])
        # rho_hhe[self.y == 0.] = 10 ** self.hhe_eos.get_h_logrho((np.log10(self.p[self.y == 0]), np.log10(self.t[self.y == 0.])))
        rho_hhe[self.kcore:] = 10 ** res['logrho']
        self.dlogrho_dlogz = np.zeros_like(self.p)
        # dlogrho_dlogz is only calculable where all of X, Y, and Z are non-zero.
        self.dlogrho_dlogz[self.z * self.y > 0.] = -1. * self.rho[self.z * self.y > 0.] * self.z[self.z * self.y > 0.] * (rho_z[self.z * self.y > 0.] ** -1 - rho_hhe[self.z * self.y > 0.] ** -1)
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
            self.dlogrho_dlogy[self.kcore:] * self.grady[self.kcore:])

        # akin to mike montgomery's form for brunt_B, which is how mesa does it by default (Paxton+2013)
        rho_this_pt_next_comp = np.zeros_like(self.p)
        if np.all(self.z[self.kcore:-1] > 0.):
            rho_this_pt_next_comp[self.kcore+1:] = self.get_rho_xyz(np.log10(self.p[self.kcore+1:]), np.log10(self.t[self.kcore+1:]), self.y[self.kcore:-1], self.z[self.kcore:-1])
        else:
            rho_this_pt_next_comp[self.kcore+1:] = 10 ** self.hhe_eos.get_logrho(np.log10(self.p[self.kcore+1:]), np.log10(self.t[self.kcore+1:]), self.y[self.kcore:-1])
        # core-mantle point must be treated separately since next cell down is pure Z, and get_rho_xyz is not designed for pure Z.
        # call z_eos.get_logrho directly instead.
        if self.kcore > 0:
            if self.evol_params['z_eos_option']:
                rho_this_pt_next_comp[self.kcore] = 10 ** self.z_eos.get_logrho(np.log10(self.p[self.kcore]), np.log10(self.t[self.kcore]))
            elif self.static_params['core_prho_relation']:
                if self.static_params['core_prho_relation'] == 'hm89 rock':
                    rho_this_pt_next_comp[self.kcore] = self.get_rhoz_hm89_rock(self.p[self.kcore], self.rho[self.kcore])
                elif self.static_params['core_prho_relation'] == 'hm89 ice':
                    rho_this_pt_next_comp[self.kcore] = self.get_rhoz_hm89_ice(self.p[self.kcore], self.rho[self.kcore])
                else:
                    raise ValueError ('if using core_prho_relation, only options are hm89 rock and hm89 ice.')
            else:
                raise ValueError('cannot include a core unless either z_eos_option or  are set.')
        # within core, composition is assumed constant so rho_this_pt_next_comp is identical to rho.
        if self.static_params['erase_z_discontinuity_from_brunt']:
            rho_this_pt_next_comp[:self.kcore+1] = self.rho[:self.kcore+1]
        else:
            rho_this_pt_next_comp[:self.kcore] = self.rho[:self.kcore]

        self.brunt_b_mhm = np.zeros_like(self.p)
        self.brunt_b_mhm[:-1] = (np.log(rho_this_pt_next_comp[1:]) - np.log(self.rho[:-1])) / (np.log(self.rho[1:]) - np.log(self.rho[:-1])) / self.chit[:-1]
        # self.brunt_b_mhm[:-1] = (np.log(rho_this_pt_next_comp[:-1]) - np.log(self.rho[:-1])) / (np.log(self.rho[:-1]) - np.log(self.rho[1:])) / self.chit[:-1]
        self.brunt_n2_mhm = self.g ** 2 * self.rho / self.p * self.chit / self.chirho * (self.grada - self.gradt + self.brunt_b_mhm)
        self.brunt_n2_mhm[0] = 0. # had nan previously, probably from brunt_b

        # self.brunt_b = self.brunt_b_mhm
        # self.brunt_n2 = self.brunt_n2_mhm
        self.brunt_n2_thermal = self.g ** 2 * self.rho / self.p * self.chit / self.chirho * (self.grada - self.gradt)

        self.brunt_n2 = self.brunt_n2_unno

        # this is the thermo derivative rho_t in scvh parlance. necessary for gyre, which calls this minus delta.
        # dlogrho_dlogt_const_p = chit / chirho = -delta = -rho_t
        self.dlogrho_dlogt_const_p = np.zeros_like(self.p)
        # print 'at time of calculating rho_t for final static model, log core temperature is %f' % np.log10(self.t[0])
        if self.kcore > 0 and self.evol_params['z_eos_option']:
            self.dlogrho_dlogt_const_p[:self.kcore] = self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
        if self.evol_params['z_eos_option'] == 'reos water' and self.t[-1] < 1e3: # must be calculated separately for low T and high T part of the envelope
            k_t_boundary = np.where(np.log10(self.t) > 3.)[0][-1]
            try:
                if self.z1 > 0.: # use eq. (16) in ms.pdf for this derivative from additive volume mixture
                    self.dlogrho_dlogt_const_p[self.kcore:k_t_boundary+1] = \
                            self.rho[self.kcore:k_t_boundary+1] \
                            * (self.z[self.kcore:k_t_boundary+1] / rho_z[self.kcore:k_t_boundary+1] \
                                * self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[self.kcore:k_t_boundary+1]), \
                                                                        np.log10(self.t[self.kcore:k_t_boundary+1])) \
                            + (1. - self.z[self.kcore:k_t_boundary+1]) / rho_hhe[self.kcore:k_t_boundary+1] \
                                * res['rhot'][:k_t_boundary+1-self.kcore]) # this funny slice is because res[...] only runs kcore to surface
                else: # pure H/He, just ask SCvH
                    self.dlogrho_dlogt_const_p[self.kcore:k_t_boundary+1] = res['rhot'][:k_t_boundary+1]
            except:
                print('failed in dlogrho_dlogt_const_p for hi-T part of envelope')
                raise
            try: # fix for z==0
                if self.z1 > 0.:
                    self.dlogrho_dlogt_const_p[k_t_boundary+1:] = self.rho[k_t_boundary+1:] \
                                                    * (self.z[k_t_boundary+1:] / rho_z[k_t_boundary+1:] \
                                                    * self.z_eos_low_t.get_dlogrho_dlogt_const_p(np.log10(self.p[k_t_boundary+1:]), \
                                                                                                np.log10(self.t[k_t_boundary+1:])) \
                                                    + (1. - self.z[k_t_boundary+1:]) / rho_hhe[k_t_boundary+1:] \
                                                        * res['rhot'][k_t_boundary+1-self.kcore:])
                else:
                    self.dlogrho_dlogt_const_p[k_t_boundary+1:] = res['rhot'][k_t_boundary+1-self.kcore:]
            except:
                print('failed in dlogrho_dlogt_const_p for lo-T part of envelope')
                raise

        else: # no need to sweat low vs. high t (only an REOS-H2O limitation)
            if self.evol_params['z_eos_option']:
                self.dlogrho_dlogt_const_p[self.kcore:] = self.rho[self.kcore:] * \
                    (self.z[self.kcore:] / rho_z[self.kcore:] \
                    * self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:])) \
                    + (1. - self.z[self.kcore:]) / rho_hhe[self.kcore:] \
                    * res['rhot'])
            else:
                assert np.all(self.z[self.kcore:] == 0.), 'consistency check failed: z_eos_option is None, but have non-zero z in envelope'
                self.dlogrho_dlogt_const_p[self.kcore:] = res['rhot']

        if self.z2: # two-layer envelope in terms of Z
            self.mz_env_outer = np.sum(self.dm[self.ktrans:]) * self.z[self.ktrans + 1]
            self.mz_env_inner = np.sum(self.dm[self.kcore:self.ktrans]) * self.z[self.ktrans - 1]
            self.mz_env = self.mz_env_outer + self.mz_env_inner
            self.mz_core = np.dot(self.z[:self.kcore], self.dm[:self.kcore])
            self.mz = self.mz_env_outer + self.mz_env_inner + self.mz_core
        else:
            # Z uniform in envelope, Z=1 in core.
            self.mz_env = self.z[-1] * (self.mtot - self.mcore * const.mearth)
            self.mz = self.mz_env + self.mcore * const.mearth

        self.bulk_z = self.mz / self.mtot
        self.ysurf = self.y[-1]
        self.envelope_mean_y = np.dot(self.dm[self.kcore:], self.y[self.kcore:-1]) / np.sum(self.dm[self.kcore:])

        # axial moment of inertia if spherical, in units of mtot * rtot ** 2. moi of a thin spherical shell is 2 / 3 * m * r ** 2
        self.nmoi = 2. / 3 * trapz(self.r ** 2, x=self.m) / self.mtot / self.rtot ** 2

        self.pressure_scale_height = self.p / self.rho / self.g
        self.mf = self.m / self.mtot
        self.rf = self.r / self.rtot

    def set_entropy(self):
        # set entropy in envelope (ignore z contribution in envelope)
        self.entropy = np.zeros_like(self.p)
        self.entropy[self.kcore:] = 10 ** self.hhe_eos.get_logs(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:]) * const.mp / const.kb
        # experimenting with including entropy of core material (don't bother with aneos, it's not a column).
        if self.static_params['include_core_entropy']:
            if not self.evol_params['z_eos_option'] == 'reos water':
                raise NotImplementedError("including entropy of the core is only possible if z_eos_option == 'reos water'.")
            else:
                self.entropy[:self.kcore] = 10 ** self.z_eos.get_logs(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore])) * const.mp / const.kb



    def evolve(self, params):

        '''builds a sequence of static models with different surface temperatures and calculates the delta time between each pair
        using the energy equation dL/dm = -T * ds/dt where L is the intrinsic luminosity, m is the mass coordinate, T is the temperature,
        s is the specific entropy, and t is time.
        important inputs are start_t (say t1 = 1, 2, 3 thousand K) and end_t (put just beneath a realistic t1 for J/S/U/N or else it'll run off F11 atm tables.)
        for now ignores the possibility of having two-layer envelope in terms of Y or Z.'''

        # necessaries
        assert 'mtot' in list(params), 'must specify total mass.'
        if 'yenv' in list(params): params['y1'] = params.pop('yenv')
        if 'zenv' in list(params): params['z1'] = params.pop('zenv')
        assert 'y1' in list(params), 'must specify y1.'
        assert 'z1' in list(params), 'must specify z1.'

        # defaults
        if not 'mcore' in params.keys():
            params['mcore'] = 0.
        if not 'start_t' in params.keys():
            params['start_t'] = 2e3
        if not 'which_t' in params.keys():
            params['which_t'] = 't1'

        try:
            stdout_interval = params['stdout_interval']
        except KeyError:
            stdout_interval = 1

        self.evolve_params = params

        start_t = params['start_t']
        end_t = params['end_t']
        which_t = params['which_t']

        # old way: set array of surface temperatures to compute, logarithmically distributed from the end
        # nsteps = params['nsteps']
        # ts = np.logspace(np.log10(end_t), np.log10(start_t), nsteps)[::-1]

        previous_entropy = None
        # these columns are for the realtime (e.g., notebook) output
        stdout_columns = 'step', 'iters', 'limit', which_t, 'teff', 'radius', 'dt_yr', 'age_gyr', 'nz_grady', 'nz_shell', 'y1', 'et_s'
        start_time = time.time()
        header_format = '{:>6s} {:>6s} {:>6s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>6s} {:>8s}'
        stdout_format = '{:>6n} {:>6n} {:>6s} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10n} {:>10n} {:>6.3f} {:>8.1f}'
        print(header_format.format(*stdout_columns))

        # the evolve loop
        self.step = 0
        self.age_gyr = 0
        self.dt_yr = 0
        self.status = 'okay'
        delta_t = 2 # nominal temperature step in K, can scale as we go
        prev_y1 = -1
        done = False
        limit = 'ok'
        while not done:

            params[which_t] = start_t
            other_t = {'t1':'t10', 't10':'t1'}[params['which_t']]
            params.pop(other_t, None)

            self.static(params) # pass the full evolve params; many won't be used
            # normally we also catch AtmError (end gracefully with done=True) and raise others

            if self.step > 0:
                accept_step = False
                titers = 0
                limit = 'ok'
                while not accept_step:
                    params[which_t] = previous_t - delta_t
                    other_t = {'t1':'t10', 't10':'t1'}[params['which_t']]
                    params.pop(other_t, None)

                    self.static(params) # pass the full evolve params; many won't be used, but shouldn't cause any problems
                    # normally we also catch AtmError (end gracefully with done=True) and raise others

                    delta_s = self.entropy - previous_entropy
                    delta_s *= const.kb / const.mp # now erg K^-1 g^-1
                    int_tdsdm = trapz(self.t * delta_s, dx=self.dm)
                    dt = -1. * int_tdsdm / self.lint
                    # now that have timestep based on total intrinsic luminosity,
                    # calculate eps_grav to see distribution of energy generation
                    eps_grav = -1. * self.t * delta_s / dt
                    luminosity = np.insert(cumtrapz(eps_grav, dx=self.dm), 0, 0.)
                    self.dt_yr = dt / const.secyear
                    if self.dt_yr < 0:
                        self.status = EnergyError('got negative timestep %f for step %i' % (dt, self.step))
                        done = True
                        break

                    if self.dt_yr < params['max_timestep']: # okay on timestep
                        if prev_y1 - self.y[-1] < params['max_dy1']: # okay on dy1
                            delta_t *= (params['target_timestep'] / self.dt_yr) ** 0.5
                            accept_step = True
                        else:
                            delta_t *= (params['max_dy1'] / (prev_y1 - self.y[-1])) ** 1.5
                            # msg = 'y1 change {:.2e} exceeds limit {:.2e}; retry'.format(prev_y1-self.y[-1], params['max_dy1'])
                            limit = 'dy1'
                    else: # retry with smaller step
                        delta_t *= (params['target_timestep'] / self.dt_yr) ** 0.5
                        # msg = 'timestep {:e} exceeds limit {:e}; retry'.format(self.dt_yr, params['max_timestep'])
                        limit = 'dt'

                    titers += 1

                self.age_gyr += self.dt_yr * 1e-9
                self.delta_s = delta_s # erg K^-1 g^-1
                self.eps_grav = eps_grav
                self.luminosity = luminosity
                prev_y1 = self.y[-1]

            if 'full_profiles' in list(params):
                if params['full_profiles']:
                    if not hasattr(self, 'profiles'): self.profiles = {}
                    assert self.step not in list(self.profiles), 'profiles dict already has entry for step {}'.format(self.step)
                    self.profiles[self.step] = self.get_profile()
            self.append_history()

            if self.status != 'okay':
                print('stopping:', self.status)
                break

            if self.step % stdout_interval == 0: # or self.step == nsteps - 1:
                walltime = time.time() - start_time
                stdout_data = self.step, self.iters, limit, params[which_t], self.teff, \
                    self.rtot, self.dt_yr, self.age_gyr, self.nz_gradient, self.nz_shell, self.y[-1], walltime
                print(stdout_format.format(*stdout_data))

            # catch stopping condition
            if params[which_t] < end_t:
                done = True
            else:
                previous_entropy = self.entropy
                previous_t = params[which_t]
                self.step += 1

        if 'output_prefix' in list(params):
            self.save_history(params['output_prefix'])

    def append_history(self):
            history_qtys = {
                'step': self.step,
                'iters': self.iters,
                'age': self.age_gyr,
                'dt_yr': self.dt_yr,
                'radius': self.rtot,
                'tint': self.tint,
                'lint': self.lint,
                't1': self.t1,
                't10': self.t10,
                'teff': self.teff,
                'y1': self.y[-1],
                'nz_gradient': self.nz_gradient,
                'nz_shell': self.nz_shell,
                'mz_env': self.mz_env,
                'mz': self.mz,
                'bulk_z': self.bulk_z,
                'status': self.status,
                'kcore':self.kcore,
                'ktrans':self.ktrans,
                'k_shell_top':self.k_shell_top
                }

            if not hasattr(self, 'history'):
                self.history = {}

            for key, qty in history_qtys.items():
                if not key in list(self.history):
                    self.history[key] = np.array([]) # initialize ndarray for this column
                self.history[key] = np.append(self.history[key], qty)

    def get_profile(self):
        profile = {}
        profile['p'] = np.copy(self.p)
        profile['t'] = np.copy(self.t)
        profile['rho'] = np.copy(self.rho)
        profile['y'] = np.copy(self.y)
        profile['z'] = np.copy(self.z)
        profile['entropy'] = np.copy(self.entropy)
        profile['r'] = np.copy(self.r)
        profile['m'] = np.copy(self.m)
        profile['g'] = np.copy(self.g)
        profile['dlogp_dlogrho'] = np.copy(self.dlogp_dlogrho)
        profile['gamma1'] = np.copy(self.gamma1)
        profile['csound'] = np.copy(self.csound)
        # profile['lamb_s12'] = np.copy(self.lamb_s12)
        profile['brunt_n2'] = np.copy(self.brunt_n2)
        # profile['brunt_n2_direct'] = self.brunt_n2_direct
        profile['chirho'] = np.copy(self.chirho)
        profile['chit'] = np.copy(self.chit)
        profile['gradt'] = np.copy(self.gradt)
        profile['grada'] = np.copy(self.grada)
        profile['rf'] = np.copy(self.rf)
        profile['mf'] = np.copy(self.mf)
        # profile['pressure_scale_height'] = self.pressure_scale_height
        try:
            profile['dy'] = np.copy(self.y - self.ystart)
        except AttributeError: # no phase separation for this step
            pass
        if self.step > 0:
            profile['delta_s'] = np.copy(self.delta_s)
            profile['eps_grav'] = np.copy(self.eps_grav)
            profile['luminosity'] = np.copy(self.luminosity)

        return profile

    def dump_history(self, prefix):
        with open('{}.history'.format(prefix), 'wb') as fw:
            pickle.dump(self.history, fw, 0) # 0 means save as text
        print('wrote history data to {}.history'.format(prefix))

    def dump_profile(self, prefix):
        assert type(prefix) is str, 'output_prefix needs to be a string.'
        with open('{}{}.profile' % (prefix, step), 'w') as f:
            pickle.dump(profile, f, 0) # 0 means dump as text
        print('wrote profile data to {}{}.profile'.format(prefix, step))

    def smooth(self, array, std, type='flat'):
        '''
        moving gaussian filter to smooth an array.
        wrote this to artificially smooth brunt composition term for seismology diagnostic purposes.
        '''
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

    def save_profile(self, outfile,
                    save_gyre_model_with_profile=True,
                    smooth_brunt_n2_std=None,
                    add_rigid_rotation=None,
                    erase_y_discontinuity_from_brunt=False,
                    erase_z_discontinuity_from_brunt=False,
                    omit_brunt_composition_term=False):

        # try to be smart about output filenames
        if '.profile' in outfile and '.gyre' in outfile:
            print('please pass save_profile an output path with either .gyre or .profile extensions, or neither. not both. please.')
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

                print('wrote %i zones to %s' % (k, gyre_outfile))

        # also write more complete profile info
        with open(outfile, 'w') as f:

            # write scalars
            # these names should match attributes of the Evolver instance
            scalar_names = 'nz', 'iters', 'mtot', 'rtot', 'mcore', 'tint', 'lint', 't1', 't10', 'teff', 'ysurf', 'nz_gradient', 'nz_shell', 'zenv', 'z2', 'mz_env', 'mz', 'bulk_z', 'delta_nu', 'delta_nu_env'
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
                        print(k, name)
                f.write('\n')
            f.write('\n')

            print('wrote %i zones to %s' % (k, outfile))

class EOSError(Exception):
    pass

class AtmError(Exception):
    pass

class HydroError(Exception):
    pass

class EnergyError(Exception):
    pass

class UnphysicalParameterError(Exception):
    pass
