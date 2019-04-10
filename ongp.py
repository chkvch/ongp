import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d, splrep, splev
from scipy.integrate import trapz, cumtrapz
import const
import pickle
import time
import os
try:
    from importlib import reload
except:
    pass

class evol:

    def __init__(self, params, mesh_params={}):

        if not 'path_to_data' in params.keys():
            params['path_to_data'] = os.environ['ongp_data_path']
            if not os.path.exists(params['path_to_data']):
                raise ValueError('must specify path_to_data indicating path to eos/atm data. best way is to set environment variable ongp_data_path.')

        if not 'hhe_eos_option' in params.keys():
            params['hhe_eos_option'] = 'scvh'
        # initialize hydrogen-helium equation of state
        if params['hhe_eos_option'] == 'scvh':
            import scvh; reload(scvh)
            self.hhe_eos = scvh.eos(params['path_to_data'])
        elif params['hhe_eos_option'] == 'reos3b':
            import reos3b
            self.hhe_eos = reos3b.eos(params['path_to_data'])
        elif params['hhe_eos_option'] == 'mh13_scvh':
            import mh13_scvh; reload(mh13_scvh)
            self.hhe_eos = mh13_scvh.eos(params['path_to_data'])
        elif params['hhe_eos_option'] == 'chabrier':
            import chabrier; reload(chabrier)
            self.hhe_eos = chabrier.eos(params['path_to_data'])
        else:
            print('hydrogen-helium eos option {} not recognized'.format(params['hhe_eos_option']))

        if 'z_eos_option' in list(params):
            # initialize z equation of state
            import aneos
            import reos_water
            if params['z_eos_option'] == 'reos water':
                self.z_eos = reos_water.eos(params['path_to_data'])
                self.z_eos_low_t = aneos.eos(params['path_to_data'], 'water')
            elif 'aneos' in params['z_eos_option']:
                material = params['z_eos_option'].split()[1]
                self.z_eos = aneos.eos(params['path_to_data'], material)
                self.z_eos_low_t = None
            else:
                raise ValueError("z eos option '%s' not recognized." % params['z_eos_option'])

        # if you're wondering, model atmospheres are initialized in self.static.
        # that way we can run, e.g., a Jupiter and then a Saturn without invoking a new evol instance.

        if 'hhe_phase_diagram' in params.keys():
            if not 'extrapolate_phase_diagram_to_low_pressure' in params.keys():
                params['extrapolate_phase_diagram_to_low_pressure'] = False
            if not 'phase_p_interpolation' in params.keys():
                params['phase_p_interpolation'] = 'log'
            if not 't_shift_p1' in params.keys():
                params['t_shift_p1'] = None
            if not 'x_transform' in params.keys():
                params['x_transform'] = None
            if not 'y_transform' in params.keys():
                params['y_transform'] = None

            if params['hhe_phase_diagram'] == 'lorenzen':
                import lorenzen
                reload(lorenzen)
                self.phase = lorenzen.hhe_phase_diagram(
                                        params['path_to_data'],
                                        extrapolate_to_low_pressure=params['extrapolate_phase_diagram_to_low_pressure'],
                                        t_shift_p1=params['t_shift_p1'],
                                        p_interpolation=params['phase_p_interpolation']
                                        )
            elif params['hhe_phase_diagram'] == 'schoettler':
                import schoettler
                reload(schoettler)
                if 'extrapolate_to_low_pressure' in list(params):
                    raise NotImplementedError('extrapolate_to_low_pressure not implemented for schoettler phase diagram')
                if params['t_shift_p1'] is not None:
                    raise NotImplementedError('t_shift_p1 not implemented for schoettler phase diagram')
                if 'schoettler_add_knots' in list(params):
                    self.phase = schoettler.hhe_phase_diagram(params['path_to_data'],
                                                p_interpolation=params['phase_p_interpolation'],
                                                add_knots=params['schoettler_add_knots']
                                                )
                else:
                    self.phase = schoettler.hhe_phase_diagram(params['path_to_data'],
                                                p_interpolation=params['phase_p_interpolation']
                                                )
            else:
                raise ValueError('hydrogen-helium phase diagram option {} is not recognized.'.format(params['hhe_phase_diagram']))

        # defaults for other params
        self.evol_params = {
            'nz':1024,
            'radius_rtol':1e-4,
            'y1_rtol':1e-4,
            'max_iters_static':30,
            'min_iters_static':3,
            'max_iters_static_before_rain':3
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
            'width_core_mesh_boost':1e-1,
            'fmean_core_bdy_mesh_boost':3e-1
            }
        # overwrite with any passed by user
        for key, value in mesh_params.items():
            self.mesh_params[key] = value

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

        '''
        build a single hydrostatic model.
        the crucial independent variables are total mass, core mass, y1 and z1 (maybe y2 & z2 also),
        and either t1 or t10.
        can pass an ongp.evol instance to guess to use its current structure as a starting point for
        hydro iterations.
        '''

        if not 'phase_t_offset' in params.keys():
            params['phase_t_offset'] = 0.
        if not 'allow_y_inversions' in params.keys():
            allow_y_inversions = params['allow_y_inversions'] = False
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

        if ('t1' in params.keys()) and not ('t10' in params.keys()):
            self.atm_which_t = 't1'
            self.t1 = params['t1']
        elif ('t10' in params.keys()) and not ('t1' in params.keys()):
            self.atm_which_t = 't10'
            self.t10 = params['t10']
        else:
            raise ValueError('must specify one and only one of t1 or t10.')

        self.iters = 0

        if not hasattr(self, 'mtot'):
            '''initialize model: mesh, atm, etc'''
            assert not hasattr(self, 'atm')
            assert not hasattr(self, 'y1')
            assert not hasattr(self, 'y2')
            assert not hasattr(self, 'z1')
            assert not hasattr(self, 'z2')
            assert not hasattr(self, 'mcore')
            assert not hasattr(self, 'kcore')

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

            # initialize model atmospheres
            if 'teq' in params.keys():
                self.teq = params['teq']
            if self.evol_params['atm_option'] == 'f11_tables':
                import f11_atm; reload(f11_atm)
                if 'force_teq' in list(self.evol_params):
                    self.atm = f11_atm.atm(self.evol_params['path_to_data'], self.evol_params['atm_planet'],
                        force_teq=self.evol_params['force_teq'])
                else:
                    self.atm = f11_atm.atm(self.evol_params['path_to_data'], self.evol_params['atm_planet'])
            elif self.evol_params['atm_option'] == 'f11_fit':
                import f11_atm_fit
                self.atm = f11_atm_fit.atm(self.evol_params['atm_planet'])
            else:
                raise ValueError('atm_type %s not recognized.' % atm_type)

            if 'yenv' in list(params):
                params['y1'] = params.pop('yenv')
            if 'zenv' in list(params):
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

            # initialize lagrangian mesh.
            if not 'mcore' in params.keys(): params['mcore'] = 0.
            mcore = params['mcore']
            assert mcore * const.mearth < self.mtot, 'core mass must be less than total mass.'
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

            # if not 'transition_pressure' in params.keys():
            #     params['transition_pressure'] = 1.

            # set initial composition information. for now, envelope is homogeneous
            self.y[:] = 0.
            self.y[self.kcore:] = self.y1

            self.z[:self.kcore] = 1.
            assert self.z1 >= 0., 'got negative z1 %g' % self.z1
            self.z[self.kcore:] = self.z1

            # these used to be defined after iterations were completed, but they are needed for calculation
            # of brunt_b to allow superadiabatic regions with grad-grada proportional to brunt_b.
            self.gradt = np.zeros_like(self.p)
            self.brunt_b = np.zeros_like(self.p)
            self.chirho = np.zeros_like(self.p)
            self.chit = np.zeros_like(self.p)
            self.chiy = np.zeros_like(self.p)
            self.grady = np.zeros_like(self.p)

            # helium rain bookkeeping
            self.mhe = np.dot(self.y[:-1], self.dm) # initial total he mass
            self.k_shell_top = None # until a shell is found by equilibrium_y_profile

            self.grada = np.zeros_like(self.m)
            # first guess, values chosen just so that densities will be calculable
            self.p[:] = 1e12
            self.t[:] = 1e4

            # get density everywhere based on primitive guesses
            self.set_core_density()
            self.set_envelope_density(ignore_z=True) # ignore Z for first pass at densities
            self.integrate_continuity() # rho, dm -> r
        else:
            # self.y[:] = 0.
            # self.y[self.kcore:] = self.y1
            self.p_start = np.copy(self.p)
            self.t_start = np.copy(self.t)
            self.y_start = np.copy(self.y)
            pass

        if 'debug_iterations' in params.keys():
            if params['debug_iterations']:
                t0 = time.time()

        # relax to hydrostatic
        last_three_radii = 0, 0, 0
        for iteration in range(self.evol_params['max_iters_static']):
            self.iters += 1
            self.integrate_hydrostatic() # integrate momentum equation to get pressure
            self.locate_transition_pressure() # find point that should be discontinuous in y and z, if any
            # set y and z profile assuming three-layer homogeneous. if doing helium rain (self.phase is set),
            # Y profile wile be set appropriately later in equilibrium_Y iterations.
            self.set_yz()
            self.integrate_temperature()
            self.set_core_density()
            self.set_envelope_density()
            self.integrate_continuity() # get zone radii from their densities via continuity equation

            if 'debug_iterations' in params.keys():
                if params['debug_iterations']:
                    et = time.time() - t0
                    print('iter {:>2n}, rtot {:.5e}, ktrans {}, et_ms {:5.2f}'.format(self.iters, self.r[-1], self.ktrans, et*1e3))

            # if going to repeat hydro iterations with rainout calculation, quit early even if
            # radius is still changing, because iterations with rainout will take care of it
            if hasattr(self, 'phase') and iteration >= self.evol_params['max_iters_static_before_rain']:
                last_three_radii = last_three_radii[1], last_three_radii[2], self.r[-1]
                break

            # hydrostatic model is judged to be converged when the radius has changed by a relative amount less than
            # radius_rtol over both of the last two iterations.
            # if np.all(np.abs((last_three_radii / self.r[-1] - 1.)) < self.evol_params['radius_rtol']):
            if np.all(np.abs(np.mean((last_three_radii / self.r[-1] - 1.))) < self.evol_params['radius_rtol']):
                if iteration >= self.evol_params['min_iters_static']:
                    last_three_radii = last_three_radii[1], last_three_radii[2], self.r[-1]
                    break
            # else:
                # print('radius change exceeds tolerance; continue')
            if not np.isfinite(self.r[-1]):
                raise HydroError('found infinite total radius.')

            last_three_radii = last_three_radii[1], last_three_radii[2], self.r[-1]

        else:
            raise ConvergenceError('{} exceeded max iterations {}'.format(iteration, self.evol_params['max_iters_static']))

        if params['verbose']: print('converged homogeneous model after %i iterations.' % self.iters)

        self.t_before_he = np.copy(self.t)
        self.y_before_he = np.copy(self.y)
        self.p_before_he = np.copy(self.p)

        last_three_y1 = 0, 0, 0
        # repeat hydro iterations, now including the phase diagram calculation (if helium rain)
        if hasattr(self, 'phase'):
            # last_three_radii = 0, 0, 0

            # if hasattr(self, 'prev_y'):
            #     self.y[:] = self.prev_y[:]
            #     del(self.prev_y)
            #     self.set_envelope_density()
            #     self.integrate_continuity()
            self.k1 = 0
            for iteration in range(self.evol_params['max_iters_static']):
                self.iters_rain = iteration + 1

                self.integrate_hydrostatic()
                self.locate_transition_pressure() # find point that should be discontinuous in y and z, if any
                # self.integrate_temperature()
                self.k_gradient_bot = None
                if 'rrho_where_have_helium_gradient' in params.keys() and self.nz_gradient > 0:
                    # allow helium gradient regions to have superadiabatic temperature stratification.
                    # this is new here -- copied from below where we'd ordinarily only compute this
                    # after we have a converged model.
                    # will cost some time because of the additional eos calls.

                    # as is, brunt_b only accounts for hydrogen/helium gradients. extending to include more
                    # species is straightforward: chiy*grady becomes sum_i^{N-1}(chi_i * grad X_i) where sum_i^N(X_i)=1.

                    # new: moved these to self.integrate_temperature, since an eos call happens there anyway
                    # res = self.hhe_eos.get(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
                    # self.chit[self.kcore:] = res['chit']
                    # self.chirho[self.kcore:] = res['chirho']
                    # self.chiy[self.kcore:] = res['chiy']

                    self.grady[self.kcore+1:] = np.diff(np.log(self.y[self.kcore:])) / np.diff(np.log(self.p[self.kcore:]))
                    if self.k_shell_top:
                        self.grady[self.k_shell_top+1] = 0. # ignore discontinuous step up to Y=0.95
                        if 'adjust_shell_top_abundance' in list(self.static_params):
                            if self.static_params['adjust_shell_top_abundance']:
                                # also zero large grady resulting from intermediate abundance step
                                self.grady[self.k_shell_top] = 0.

                    # more general way of doing it treats the similar discontinuity that may exist
                    # even before there is a a Y=0.95 shell.
                    k_gradient_bottom = np.where(np.diff(np.diff(self.y[self.kcore:])) < -0.1)[0]
                    if np.any(k_gradient_bottom):
                        self.k_gradient_bot = k_gradient_bottom[0] + 1 + self.kcore
                        if self.static_params['adjust_shell_top_abundance']:
                            self.k_gradient_bot += 1 # shift to intermediate-abundance zone
                    # k_gradient_bottom = np.where(np.diff(self.y[self.kcore:]) / self.y[self.kcore:-1] < -0.1)[0]
                    # if np.any(k_gradient_bottom):
                    #     if len(k_gradient_bottom) > 2:
                    #         raise ValueError('more than two zones with dlny < -0.1')
                    #     elif len(k_gradient_bottom) == 2:
                    #         assert self.static_params['adjust_shell_top_abundance'] # only reason why this should happen
                    #         self.k_gradient_bot = k_gradient_bottom[-1] + self.kcore
                    #     else: # normal situation
                    #         self.k_gradient_bot = k_gradient_bottom[0] + self.kcore

                    self.brunt_b[self.kcore+1:] = self.chirho[self.kcore+1:] / self.chit[self.kcore+1:] * self.chiy[self.kcore+1:] * self.grady[self.kcore+1:]

                    # self.gradt += params['rrho_where_have_helium_gradient'] * self.brunt_b
                    self.gradt = self.grada + params['rrho_where_have_helium_gradient'] * self.brunt_b

                    self.integrate_temperature(adiabatic=False)
                else:
                    self.integrate_temperature()

                self.y = self.equilibrium_y_profile(params['phase_t_offset'],
                            verbosity=params['rainout_verbosity'],
                            allow_y_inversions=params['allow_y_inversions'])
                # these bookkeeping numbers are not accurate if y inversions allowed;
                # not a big deal but do fix later
                if np.any(np.diff(self.y) < 0):
                    self.nz_gradient = len(np.where(np.diff(self.y) < 0)[0])
                    self.k_gradient_top = np.where(np.diff(self.y) < 0)[0][-1] + 1
                    self.k_gradient_bot = np.where(np.diff(self.y) < 0)[0][0]
                else:
                    self.nz_gradient = 0
                    self.k_gradient_top = None
                    self.k_gradient_bot = None

                self.set_core_density() # z eos call, fast (not many zones)
                self.set_envelope_density() # full-on eos call; could try skipping and using rho from last eos call (integrate_temperature)
                self.integrate_continuity() # just an integral, super fast

                if 'debug_iterations' in params.keys():
                    if params['debug_iterations']:
                        et = time.time() - t0
                        qtys = self.iters, self.iters_rain, self.r[-1], self.y[-1], self.k1, self.p[self.k1]*1e-12, et*1e3
                        print('iter {:>2n}, he_iter {:>2n}, rtot {:.5e}, y1 {:>.5f}, k1 {}, p[k1] {:.5f}, et_ms {:5.2f}'.format(*qtys))
                        if 'debug_iterations' in list(params):
                            if type(params['debug_iterations']) is str:
                                step = int(params['debug_iterations'].split()[1])
                                self.step = -1 # banana
                                if self.step == step:
                                    self.rtot = self.r[-1]
                                    self.set_derivatives_etc()
                                    with open('{:03n}.dump.pkl'.format(self.iters_rain), 'wb') as fw:
                                        pickle.dump(self, fw)
                # print('dr={:10.5e} (rtol={:10.5e}) dy1={:10.5e} (rtol={:10.5e})'.format(np.abs(np.mean((last_three_radii / self.r[-1] - 1.))), self.evol_params['radius_rtol'], np.abs(np.mean((last_three_y1 / self.y[-1] - 1.))), self.evol_params['y1_rtol']))
                if np.all(np.abs(np.mean((last_three_radii / self.r[-1] - 1.))) < self.evol_params['radius_rtol']):
                    # print('rain iter {}: radius ok'.format(self.iters_rain))
                    if np.all(np.abs(np.mean((last_three_y1 / self.y[-1] - 1.))) < self.evol_params['y1_rtol']):
                        # print('rain iter {}: y1 okay'.format(self.iters_rain))
                        break
                    # else:
                        # print('y1 change exceeds tolerance; keep going')
                # else:
                    # print('radius change exceeds tolerance; keep going')

                if not np.isfinite(self.r[-1]):
                    with open('output/found_infinite_radius.dat', 'w') as f:
                        f.write('%12s %12s %12s %12s %12s %12s\n' % ('core', 'p', 't', 'rho', 'm', 'r'))
                        for k in range(self.nz):
                            f.write('%12s %12g %12g %12g %12g %12g\n' % (k < self.kcore, self.p[k], self.t[k], self.rho[k], self.m[k], self.r[k]))
                    print('saved output/found_infinite_radius.dat')
                    raise ValueError('found infinite total radius')
                last_three_radii = last_three_radii[1], last_three_radii[2], self.r[-1]
                last_three_y1 = last_three_y1[1], last_three_y1[2], self.y[-1]
                # include a similar check on y1?
            else:
                raise ConvergenceError('static model iters {} exceeded max_iters_static {}'.format(self.iters_rain, self.evol_params['max_iters_static']))
        else:
            self.k_gradient_top = None
            self.k_gradient_bot = None
        # finalize hydrostatic profiles (maybe not necessary since we just did 20 iterations)
        # self.integrate_hydrostatic()
        # self.set_core_density()
        # self.set_envelope_density()

        # finally, calculate lots of auxiliary quantities of interest
        self.rtot = self.r[-1]
        self.set_atm() # make sure t10 is set; use (t10, g) to get (tint, teff) from model atmosphere
        self.set_entropy() # set entropy profile (necessary for an evolutionary calculation)
        self.set_derivatives_etc() # calculate thermo derivatives, seismology quantities, g, etc.

        return

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

    def equilibrium_y_profile(self, phase_t_offset, verbosity=0, show_timing=False, allow_y_inversions=False):
        '''uses the existing p-t profile to find the thermodynamic equilibrium y profile, which
        may require a helium-rich layer atop the core.'''
        p = self.p * 1e-12 # Mbar
        t = self.t * 1e-3 # kK
        self.ymax = np.ones_like(self.p)

        # the phase diagram came from a model for a system of just H and He.
        # if xp and Yp represent the helium number fraction and mass fraction from the phase diagram,
        # this would correspond to xp > x and Yp > Y where x, Y are the fractions in a real model
        # that also includes a Z component.
        # given Z, set Y from Yp according to
        #   X + Y + Z = 1, and
        #   Y / X = Yp / Xp.
        # these combine to
        #   Y = (1 - Z) / (1 + (1 - Yp) / Yp).

        from schoettler import get_xp, get_y # get_xp(z, y); get_y(z, xp)

        if verbosity > 0: print('iters {}, iters rain {:2n}, rtot {:.5e}, y1 {:.4f} '.format(self.iters, self.iters_rain, self.r[-1], self.y[-1]))

        ymax = {}
        for pval in self.phase.pvals:
            # start by checking just nodes (p==0.5, 1, 1.5, 2, 4, 10...).
            # if not allow_y_inversions, then the minimum ymax reached by the planet P-T profile
            # on any of the known phase curves will establish the envelope abundance.
            # that will probably be 2 Mbar, or 4 Mbar if model goes to low y1 (high phase_t_offset).

            k = np.where(p < pval)[0][0]-1 # e.g., 2.001 Mbar
            if k < self.kcore: continue
            try:
                # interpolate to get t value corresponding to pval, e.g., 2.0000 Mbar
                ti = splev(pval, splrep(p[k+3:k-3:-1], t[k+3:k-3:-1], k=3))
            except:
                # print('failed to get model {} Mbar temperature'.format(pval))
                # print(k, self.kcore)
                # print(p[k+3:k-3:-1])
                # print(t[k+3:k-3:-1])
                # raise
                continue

            xlo = splev(ti - phase_t_offset*1e-3, self.phase.tck_xlo[pval])
            ylo = get_y(self.z[k], xlo)
            if ylo < 0 or ylo > 1:
                # print('got bad ymax {} for {} Mbar'.format(ylo, pval))
                continue
            ymax[pval] = {'k':k, 'y':ylo}

        if 10 not in list(ymax):
            t10M = splev(10, splrep(p[self.kcore:][::-1], t[self.kcore:][::-1], k=1))
            xlo = splev(t10M - phase_t_offset*1e-3, self.phase.tck_xlo[10])
            ylo = get_y(self.z[-1], xlo)
            if ylo < 0 or ylo > 1:
                pass
            else:
                ymax[10.] = {'k':0, 'y':ylo}

        ps = list(ymax)
        ys = [ymax[pval]['y'] for pval in ps]
        ks = [ymax[pval]['k'] for pval in ps]

        if hasattr(self, 'p_start'):
            # get this static model's starting y vector, interpolated onto current pressures
            ystart = splev(self.p, splrep(self.p_start[::-1], self.y_start[::-1], k=1))
        else:
            ystart = self.y

        if not np.any(ys):
            # print('no rainout')
            return self.y
        if not np.any(ystart > min(ys)):
            # print('no rainout')
            return self.y

        if verbosity > 1:
            print(ps, ys)
        fymax = lambda p: splev(p, splrep(ps, ys, k=1))
        self.tck_ymax = splrep(ps, ys, k=1)

        if allow_y_inversions:
            # gradient loop starts at first zone with y > ymax as given by fymax (linear p-y).
            try:
                k1 = np.where(ystart - fymax(p) > 0)[0][-1]
            except IndexError:
                return self.y
            # k1 = np.where(self.y - fymax(p) > 0)[0][-1]
            # if k1 outside lowest pval in list(ymax), then abundance would be set by
            # extrapolating from above.
            # instead, k1 will be at the lowest pressure for which ymax was calculable.
            if k1 > max(ks):
                # print ('k1={}>max(ks)={}; p[k1]={}'.format(k1, max(ks), p[k1]))
                k1 = max(ks)
            elif k1 == max(ks)-1:
                # print ('k1={}<max(ks)={}; p[k1]={}'.format(k1, max(ks), p[k1]))
                pass
            if self.iters_rain > 1:
                if k1 > self.k1: # self.k1 is k1 from previous iteration
                    # make sure boundary doesn't drift outwards; tends to cause one-zone oscillations
                    # that prevent the static model from converging
                    k1 = self.k1
            # for pval in list(ymax):
                # print(pval, ymax[pval], p[ymax[pval]['k']])
            if verbosity > 1:
                print('inversion k1={:5n} p={:8.4f} ystart={:8.4f} {:4s} fymax={:8.4f}'.format(k1, p[k1], ystart[k1], '->', fymax(p[k1])))
        else:
            # gradient loop starts at pressure with minimum maxy in order to guarantee monotone y.
            # this is always a node (e.g., p=2, p=4) by construction of fymax (linear p-y).
            ys = np.array([ymax[pval]['y'] for pval in list(ymax)])
            ks = np.array([ymax[pval]['k'] for pval in list(ymax)])
            k1 = ks[ys==min(ys)][0]
            # even if fictitious (interpolated) 2-Mbar temperature does dip below tphase for 0.27,
            # the first zone with P > 2 Mbar might still find ymax of, e.g., 0.2705. in that case
            # we've obviously found no rainout.
            # gradient loop below assumes that k1 will set envelope abundance for sure.
            # so handle this case by breaking here.
            if ystart[k1] < fymax(p[k1]):
                return self.y
            if verbosity > 1:
                print('noinv k1', k1, p[k1], ystart[k1], fymax(p[k1]))

        self.k1 = k1

        t0 = time.time()
        rainout_to_core = False

        # self.k_shell_top = None # leave alone; may exist already even if the current iteration doesn't enter shell loop
        yout = np.zeros_like(self.y)
        for k in np.arange(k1, self.kcore, -1): # inward from first point below phase curve
            t1 = time.time()
            if k == k1:
                yout[k] = fymax(p[k])
                if allow_y_inversions: # not necessarily hugging one of the phase curves; set simply
                    yout[k:] = yout[k]
                else:
                    # abundance should be hugging phase curve (usually 2 Mbar). since p[k] possibly
                    # much larger than pval of relevant phase curve, yout[k] may change discontinuously
                    # in a way that the whole envelope shouldn't.
                    # thus, set envelope abundance by evaluating ymax for p==pval exactly.
                    pval = -1
                    for pval in sorted(list(ymax))[::-1]: # high to low
                        if p[k] > pval:
                            break
                    assert pval > 0
                    assert p[k] > pval
                    assert p[k+1] < pval
                    yout[k+1:] = fymax(pval)

                    # print('noinv: fymax(pval)={}, fymax(p[k+1])={}, fymax(p[k])={}'.format(fymax(pval), fymax(p[k+1]), fymax(p[k])))
                    # print('noinv: p[k+1]={} < pval={} < p[k]={}'.format(p[k+1], pval, p[k]))
                    # assert False
            else:
                if fymax(p[k]) < yout[k]:
                    yout[k] = fymax(p[k])
                else:
                    break

            # difference between initial he mass and current proposed he mass above and including this zone
            # must be deposited into the deeper interior.
            he_mass_missing_above = self.mhe - np.dot(yout[k:], self.dm[k-1:])
            enclosed_envelope_mass = np.sum(self.dm[self.kcore:k-1]) # was previously :k
            if not enclosed_envelope_mass > 0: # at core boundary
                if verbosity > 1:
                    print('i={:n} ir={:n} rainout to core because gradient reaches core.'.format(self.iters, self.iters_rain))
                    # print(yout)
                rainout_to_core = True
                yout[self.kcore:k] = 0.
                # double-check that we have overall "missing" helium, to be made up for in
                # outward shell iterations
                assert np.dot(yout[self.kcore:], self.dm[self.kcore-1:]) < self.mhe
                kbot = k
                break
            y_interior = he_mass_missing_above / enclosed_envelope_mass
            # if verbosity > 1: print('grdient(i={:n} ir={:n} k={:n} p={:.4f} t={:.4f} yout={:.4f}  yint={:.4f})'.format(self.iters, self.iters_rain, k, p[k], t[k], yout[k], y_interior))
            if y_interior > 1:
                # inner homogeneneous region of envelope would need Y > 1 to conserve global helium mass. thus undissolved droplets on core.
                # set the rest of the envelope to Y = 0.95, then do outward iterations to find how large of a shell is needed to conserve
                # the global helium mass.
                if verbosity > 1: print('rainout to core because would need Y > 1 in inner homog region')
                rainout_to_core = True
                yout[self.kcore:k] = 0.
                # double-check that we have overall "missing" helium, to be made up for in
                # outward shell iterations
                assert np.dot(yout[self.kcore:], self.dm[self.kcore-1:]) < self.mhe
                kbot = k
                break
            elif y_interior < 0:
                print('NOPE   (i={:n} ir={:n} k={:n} p={:.4f} t={:.4f} yout={:.4f}  yint={:.4f})'.format(self.iters, self.iters_rain, k, p[k], t[k], yout[k], y_interior))
                raise ValueError('got negative interior y')
            else:
                # all good; uniformly distribute all helium that has rained out from above into deeper interior
                # if verbosity > 2: print('{:>5n} {:>8.4f} {:>10.6f} {:>10.6f}'.format(k, p[k], yout[k], y_interior))
                if verbosity > 2: print('demix  (i={:n} ir={:n} k={:n} p={:.4f} t={:.4f} yout={:.4f}  yint={:.4f})'.format(self.iters, self.iters_rain, k, p[k], t[k], yout[k], y_interior))
                yout[self.kcore:k] = y_interior

        # done with gradient zone; store k1 for next time around so that we can judge extent of homogeneous envelope
        self.k1 = k1

        # if verbosity > 0: print('rainout to core %s' % rainout_to_core)
        if show_timing: print('t0 + %f seconds' % (time.time() - t0))

        if rainout_to_core:
            # gradient extends down to kbot, below which the rest of the envelope is already set Y=0.95.
            # since proposed envelope mhe < initial mhe, must grow the He-rich shell to conserve total mass.
            if verbosity > 2: print('%5s %5s %10s %10s %10s' % ('k', 'kcore', 'dm_k', 'mhe_tent', 'mhe'))
            for k in np.arange(self.kcore, self.nz):
            # for k in np.arange(kbot, self.nz):
                # yout[k] = 0.95 # in the future, obtain value from ymax_rhs(p, t)

                # gap = self.phase.miscibility_gap(p[k], 8.)
                # since ymax is nearly independent of temperature there anyway, let's use a simple
                # function for ymax(p).
                yout[k] = get_y(self.z[k], self.phase.simple_xhi(p[k]))
                # if type(gap) is str:
                #     assert gap == 'stable'
                #     if verbosity > 1: print('warmer than critical temperature at k={} p={:.4f} t-dt={:.1f}'.format(k, p[k], t[k] - phase_t_offset*1e-3))
                #     yout[k] = -1
                # elif type(gap) is tuple:
                #     xplo, xphi = gap
                #     yout[k] = get_y(self.z[k], xphi)

                tentative_total_he_mass = np.dot(yout[self.kcore:-1], self.dm[self.kcore:])
                if verbosity > 2: print('%5i %5i %10.4e %10.4e %10.4e' % (k, self.kcore, self.dm[k-1], tentative_total_he_mass, self.mhe))
                if tentative_total_he_mass >= self.mhe:
                    # if verbosity > 1: print('tentative he mass, initial total he mass', tentative_total_he_mass, self.mhe)

                    if 'adjust_shell_top_mass' in list(self.static_params):
                        if self.static_params['adjust_shell_top_mass']:
                            # experimental: move this mass coordinate to satisfy global he conservation
                            if 'adjust_shell_top_abundance' in list(self.static_params):
                                assert not self.static_params['adjust_shell_top_abundance'] # user must choose one or the other
                            delta = tentative_total_he_mass - self.mhe # > 0
                            dm_in = self.dm[k] + delta / (yout[k+1] - yout[k])
                            dm_out = (self.dm[k] + self.dm[k+1]) - dm_in
                            # print('delta {:10e}'.format(tentative_total_he_mass-self.mhe))
                            # print('{:10e} {:10e} {:10e} {:10e}'.format(self.dm[k], self.dm[k+1], self.dm[k]+self.dm[k+1], yout[k]*self.dm[k]+yout[k+1]*self.dm[k+1]))
                            # print('{:10e} {:10e} {:10e} {:10e}'.format(dm_in, dm_out, dm_in+dm_out, yout[k]*dm_in+yout[k+1]*dm_out))
                            # assert False
                            self.dm[k] = dm_in
                            self.dm[k+1] = dm_out
                            self.m[k] = self.m[k-1] + self.dm[k]
                            self.m[k+1] = self.m[k] + self.dm[k+1]

                    if 'adjust_shell_top_abundance' in list(self.static_params):
                        if self.static_params['adjust_shell_top_abundance']:
                            # experimental: conserve He exactly by setting an intermediate abundance at shell top zone
                            if 'adjust_shell_top_mass' in list(self.static_params):
                                assert not self.static_params['adjust_shell_top_mass'] # user must choose one or the other

                            ##### self.mhe = np.dot(self.y[:-1], self.dm) # initial total he mass
                            mhe_missing = self.mhe - np.dot(yout[self.kcore:-1], self.dm[self.kcore:])
                            yout[k] += mhe_missing / self.dm[k]
                            if yout[k] > yout[k-1] and k > self.kcore:
                                print('surprise in adjust shell top abundance')
                                print('k', k, 'kcore', self.kcore)
                                self.rel_mhe_error = abs(self.mhe - np.dot(yout[self.kcore:-1], self.dm[self.kcore:])) / self.mhe
                                print(yout[k-2:k+3], self.rel_mhe_error)
                                # "intermediate" zone came out with Y > Y of shell-top zone.
                                # current zone is thus the new shell top; next zone out gets intermediate abundance.
                                gap = self.phase.miscibility_gap(p[k], 8.)
                                assert type(gap) is tuple
                                xplo, xphi = gap
                                mhe_k_1 = yout[k] * self.dm[k-1]
                                yout[k] = get_y(self.z[k], xphi)
                                mhe_k_2 = yout[k] * self.dm[k-1]
                                mhe_kp1 = yout[k+1] * self.dm[k]
                                self.rel_mhe_error = abs(self.mhe - np.dot(yout[self.kcore:-1], self.dm[self.kcore:])) / self.mhe
                                print(yout[k-2:k+3], self.rel_mhe_error)
                                yout[k+1] = (mhe_kp1 + mhe_k_1 - mhe_k_2) / self.dm[k]
                                self.rel_mhe_error = abs(self.mhe - np.dot(yout[self.kcore:-1], self.dm[self.kcore:])) / self.mhe
                                print(yout[k-2:k+3], self.rel_mhe_error)

                                tentative_total_he_mass = np.dot(yout[self.kcore:-1], self.dm[self.kcore:])
                                self.rel_mhe_error = abs(self.mhe - tentative_total_he_mass) / self.mhe
                                print(self.rel_mhe_error)
                                assert False


                    tentative_total_he_mass = np.dot(yout[self.kcore:-1], self.dm[self.kcore:])
                    self.rel_mhe_error = abs(self.mhe - tentative_total_he_mass) / self.mhe
                    # if verbosity > 1: print('satisfied he mass conservation to a relative precision of %f' % self.rel_mhe_error)

                    self.k_shell_top = k
                    break

        if rainout_to_core: assert self.k_shell_top
        # self.nz_gradient = len(np.where(np.diff(yout) < 0)[0])

        if np.any(yout > 1.):
            raise UnphysicalParameterError('one or more bad y in equilibrium_y_profile')

        if verbosity > 1:
            import pickle
            with open('he{:04n}.pkl'.format(self.iters_rain), 'wb') as f:
                pickle.dump({'p':p, 't':t, 'y':self.y, 'yout':yout, 'tck_ymax':self.tck_ymax, 'k1':k1}, f)

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


    def integrate_temperature(self, adiabatic=True, brute_force_loop=True):
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

        self.grada[:self.kcore] = 0. # ignore because doesn't play a role in the temperature structure (core is isothermal).
        # grada will in general still be set inside the core from the Z eos if possible, after a static model is converged.

        # this call to get grada is slow.
        # next round of optimization should make sure we are carrying out the minimum number of eos calls because each one
        # relies so heavily on interpolation in the hhe eos.
        # try:
            # self.grada[self.kcore:] = self.hhe_eos.get_grada(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
        # except ValueError:
            # raise EOSError('failed in eos call for grada. p[-1]={:g} t[-1]={:g}'.format(self.p[-1], self.t[-1]))
        # new: get eos res and set chirho, chit as well, as long as we're doing an eos call. then don't have to
        # get those separately in evaluation of superadiabatic gradt for helium gradient region
        try:
            res = self.hhe_eos.get(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])
            self.grada[self.kcore:] = res['grada']
            self.chit[self.kcore:] = res['chit']
            self.chirho[self.kcore:] = res['chirho']
            self.chiy[self.kcore:] = res['chiy']
        except ValueError:
            raise EOSError('failed in eos call. p[-1]={:g} t[-1]={:g}'.format(self.p[-1], self.t[-1]))

        self.grada_check_nans()
        if adiabatic:
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
        if not 'transition_pressure' in list(self.static_params):
            self.ktrans = -1
            return
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
            tck = splrep(self.p[k10-2:k10+2][::-1], self.t[k10-2:k10+2][::-1], k=3) # cubic
            self.t10 = np.float64(splev(1e7, tck))
            assert self.t1 > 0., 'bad t1 %g' % self.t1
            assert len(self.p[self.p < 1e7]) >= 5, 'fewer than 5 zones outside of 10 bars; atm likely not accurate'
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
            if 'f(a) and f(b) must have different signs' in e.args[0]:
                if hasattr(self, 'teq'):
                    fname = 'atm_get_tint'
                else:
                    fname = 'atm_get_tint_teff'
                with open('atm_fail.dat', 'w') as f:
                    for p, t, y in zip(self.p, self.t, self.y):
                        f.write('{:14g} {:14g} {:14g}\n'.format(p, t, y))
                print('wrote p,t,y to atm_fail.dat')
                raise AtmError('%s failed to bracket solution for root find. g=%g, t10=%g' % (fname, self.surface_g*1e-2, self.t10))
            else:
                raise AtmError('unspecified atm error for g=%g, t10=%g: %s' % (self.surface_g*1e-2, self.t10, e.args[0]))

    def set_derivatives_etc(self):
        self.r[0] = 1. # 1 cm central radius to keep these things at least calculable at center zone
        self.g = const.cgrav * self.m / self.r ** 2
        self.g[0] = self.g[1] # hack so that we don't get infs in, e.g., pressure scale height. won't effect anything

        hhe_res_env = self.hhe_eos.get(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:])

        # this is a structure derivative, not a thermodynamic one. wherever the profile is a perfect adiabat, this is also gamma1.
        self.dlogp_dlogrho = np.diff(np.log(self.p)) / np.diff(np.log(self.rho))

        self.gamma1 = np.zeros_like(self.p)
        self.gamma3 = np.zeros_like(self.p)
        self.csound = np.zeros_like(self.p)
        self.gradt_direct = np.zeros_like(self.p)
        self.cp = np.zeros_like(self.p)
        self.cv = np.zeros_like(self.p)

        if self.kcore > 0 and self.evol_params['z_eos_option']: # compute gamma1 in core
            self.gamma1[:self.kcore] = self.z_eos.get_gamma1(np.log10(self.p[:self.kcore]), np.log10(self.t[:self.kcore]))
        self.gamma1[self.kcore:] = hhe_res_env['gamma1']
        self.gamma3[self.kcore:] = hhe_res_env['gamma3']
        self.csound = np.sqrt(self.p / self.rho * self.gamma1)
        self.lamb_s12 = 2. * self.csound ** 2 / self.r ** 2 # lamb freq. squared for l=1
        self.cp[self.kcore:] = hhe_res_env['cp']
        self.cv[self.kcore:] = hhe_res_env['cv']

        # ignoring the envelope z component here
        self.chirho[self.kcore:] = hhe_res_env['chirho']
        self.chit[self.kcore:] = hhe_res_env['chit']
        self.gradt_direct[:self.kcore] = 0.
        self.gradt_direct[self.kcore+1:] = np.diff(np.log(self.t[self.kcore:])) / np.diff(np.log(self.p[self.kcore:]))

        self.dlogrho_dlogy = np.zeros_like(self.p)
        try:
            self.dlogrho_dlogy[self.kcore:] = hhe_res_env['chiy']
        except:
            pass

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
        rho_hhe[self.kcore:] = 10 ** hhe_res_env['logrho']
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
                                * hhe_res_env['rhot'][:k_t_boundary+1-self.kcore]) # this funny slice is because res[...] only runs kcore to surface
                else: # pure H/He
                    self.dlogrho_dlogt_const_p[self.kcore:k_t_boundary+1] = hhe_res_env['rhot'][:k_t_boundary+1-self.kcore]
            except:
                print('failed in dlogrho_dlogt_const_p for hi-T part of envelope')
                raise
            try:
                if self.z1 > 0.:
                    self.dlogrho_dlogt_const_p[k_t_boundary+1:] = self.rho[k_t_boundary+1:] \
                                                    * (self.z[k_t_boundary+1:] / rho_z[k_t_boundary+1:] \
                                                    * self.z_eos_low_t.get_dlogrho_dlogt_const_p(np.log10(self.p[k_t_boundary+1:]), \
                                                                                                np.log10(self.t[k_t_boundary+1:])) \
                                                    + (1. - self.z[k_t_boundary+1:]) / rho_hhe[k_t_boundary+1:] \
                                                        * hhe_res_env['rhot'][k_t_boundary+1-self.kcore:])
                else:
                    self.dlogrho_dlogt_const_p[k_t_boundary+1:] = hhe_res_env['rhot'][k_t_boundary+1-self.kcore:]
            except:
                print('failed in dlogrho_dlogt_const_p for lo-T part of envelope')
                raise

        else: # no need to sweat low vs. high t (only an REOS-H2O limitation)
            if self.evol_params['z_eos_option']:
                if self.z1 == 0.:
                    self.dlogrho_dlogt_const_p[self.kcore:] = hhe_res_env['rhot']
                else:
                    self.dlogrho_dlogt_const_p[self.kcore:] = self.rho[self.kcore:] * \
                        (self.z[self.kcore:] / rho_z[self.kcore:] \
                        * self.z_eos.get_dlogrho_dlogt_const_p(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:])) \
                        + (1. - self.z[self.kcore:]) / rho_hhe[self.kcore:] \
                        * hhe_res_env['rhot'])
            else:
                assert np.all(self.z[self.kcore:] == 0.), 'consistency check failed: z_eos_option is None, but have non-zero z in envelope'
                self.dlogrho_dlogt_const_p[self.kcore:] = res['rhot']

        if self.z2: # two-layer envelope in terms of Z
            assert self.ktrans > 0, 'self.z2 is set but self.ktrans is <= 0. if not setting transition_pressure, then leave z2 unset.'
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
        stdout_columns = 'step', 'iters', 'iters_he', 'retr', 'limit', \
            which_t, 'teff', 'radius', 'd{}'.format(which_t), 'dt_yr', 'age_gyr', \
            'nz_grady', 'nz_shell', 'k_grady', 'k_shell', 'y1', 'mhe_rerr','et_s'
        start_time = time.time()
        header_format = '{:>6s} {:>6s} {:>8s} {:>6s} {:>6s} '
        header_format += '{:>5s} {:>5s} {:>8s} {:>5s} {:>8s} {:>8s} '
        header_format += '{:>10s} {:>10s} {:>10s} {:>10s} {:>6s} {:>8s} {:>8s} '
        stdout_format = '{:>6n} {:>6n} {:>8n} {:>6n} {:>6s} '
        stdout_format += '{:>5.1f} {:>5.1f} {:>8.1e} {:>5.1f} {:>8.1e} {:>8.4f} '
        stdout_format += '{:>10n} {:>10n} {:>10n} {:>10n} {:>6.3f} {:>8.1e} {:>8.1f}'
        print(header_format.format(*stdout_columns))

        # the evolve loop
        self.step = 0
        self.age_gyr = 0
        self.dt_yr = 0
        self.status = 'okay'
        delta_t = params['initial_delta_t'] # nominal temperature step in K, can scale as we go
        prev_y1 = -1
        done = False
        limit = ''
        self.ymax_closest = None

        params[which_t] = start_t
        other_t = {'t1':'t10', 't10':'t1'}[params['which_t']]
        params.pop(other_t, None)

        try:
            self.static(params) # pass the full evolve params; many won't be used
        except EOSError as e:
            self.status = e
        except AtmError as e:
            self.status = e

        if self.status != 'okay':
            print('failed in initial model:', self.status)
            raise self.status

        previous_t = start_t
        previous_entropy = self.entropy

        while not done:
            self.delta_t = delta_t
            retries = 0
            accept_step = False
            limit = ''
            while not accept_step:
                old_delta_t = delta_t
                params[which_t] = previous_t - delta_t
                other_t = {'t1':'t10', 't10':'t1'}[params['which_t']]
                params.pop(other_t, None) # so that static doesn't get passed both t1 and t10

                try:
                    self.static(params) # pass the full evolve params; many won't be used
                    self.delta_s = self.entropy - previous_entropy
                    self.delta_s *= const.kb / const.mp # now erg K^-1 g^-1
                    self.tds = self.t * self.delta_s
                    self.int_tdsdm = trapz(self.t * self.delta_s, dx=self.dm)
                    dt = -1. * self.int_tdsdm / self.lint
                    # now that have timestep based on total intrinsic luminosity,
                    # calculate eps_grav to see distribution of energy generation
                    self.eps_grav = -1. * self.t * self.delta_s / dt
                    self.luminosity = np.insert(cumtrapz(self.eps_grav, dx=self.dm), 0, 0.)
                    self.dt_yr = dt / const.secyear
                except EOSError as e:
                    self.status = e
                    break
                except AtmError as e:
                    if 'failed to bracket' in e.args[0]: # probably off tables
                        self.status = e
                        break # exit retry loop
                    else:
                        raise
                except Exception as e:
                    raise
                    self.status = e
                    break # exit retry loop

                # check the timestep that this static model would imply
                assert 'target_dy1' not in list(params), 'target_dy1 not implemented in evolve.'
                if self.dt_yr < 0: # bad
                    self.status = ValueError('negative timestep')
                    raise self.status
                    break
                    # formerly, we would retry with a smaller timestep
                elif self.dt_yr < params['max_timestep']: # okay on timestep
                    if prev_y1 > 0. and prev_y1 - self.y[-1] < 0:
                        self.status = ValueError('y1 increased') # usually this is caught as negative timestep above
                        raise self.status
                        break
                        # formerly, we would retry with a larger timestep
                    elif prev_y1 - self.y[-1] < params['max_dy1']: # okay on dy1
                        if hasattr(self, 'phase') and self.nz_gradient <= 0:
                            from lorenzen import get_xp
                            # are we anywhere *close* to rainout for initial y1?
                            min_abs_t_minus_tphase = 100.
                            for pval in (1, 2, 4):
                                kp = np.where(self.p*1e-12 < pval)[0][0] - 1
                                tphase = self.phase.t_phase(pval, get_xp(self.z1, self.y1))
                                t_minus_tphase = tphase + params['phase_t_offset']*1e-3 - self.t[kp]*1e-3 # effective tphase minus true t in kK
                                min_abs_t_minus_tphase = min(min_abs_t_minus_tphase, abs(t_minus_tphase))
                                # print('kp={} p={} t={} tphase={} offset={} delta_t={}'.format(kp, self.p[kp]*1e-12, self.t[kp]*1e-3, tphase, params['phase_t_offset']*1e-3, t_minus_tphase))
                            cut1 = 0.8
                            cut2 = 0.1
                            small_delta_t = 0.3
                            if min_abs_t_minus_tphase < 0.1:
                                delta_t = small_delta_t
                                msg = '{:>50}'.format('approach rainout delta={:.2f}'.format(min_abs_t_minus_tphase))
                                limit = 'vnear'
                                accept_step = True
                            elif min_abs_t_minus_tphase < cut1:
                                # linear decrease
                                alpha = (min_abs_t_minus_tphase - cut1) / (cut2 - cut1)
                                delta_t = alpha * small_delta_t + (1. - alpha) * last_normal_delta_t
                                msg = '{:>50}'.format('approach rainout delta={:.2f}'.format(min_abs_t_minus_tphase))
                                limit = 'near'
                                accept_step = True
                            else:
                                # no rainout yet and not especially close;
                                # somewhat aggressively go toward target timestep
                                f = params['target_timestep'] / self.dt_yr
                                # f = (params['target_timestep'] / self.dt_yr) ** 0.5
                                msg = '{:>50}'.format('ok')
                                # delta_t *= 0.5 * (1. + f)
                                delta_t *= f
                                accept_step = True
                                last_normal_delta_t = delta_t
                        else:
                            # have rainout and nothing else limiting timestep;
                            # gently go toward target timestep
                            f = (params['target_timestep'] / self.dt_yr) ** 0.5
                            delta_t *= 0.5 * (1. + f)
                            # delta_t = 0.5
                            msg = '{:>50}'.format('ok')
                            accept_step = True
                    else: # dy1 too large
                        if retries < 2:
                            delta_t *= (params['max_dy1'] / (prev_y1 - self.y[-1])) ** 2.
                        else:
                            delta_t *= 0.5
                        kshell = self.k_shell_top
                        msg = '{:>50}'.format('dy1 {:.2e} > limit {:.2e}'.format(prev_y1-self.y[-1], params['max_dy1']))
                        limit = 'dy1'
                        # force static to start from fresh y profile.
                        # otherwise will start from current (depleted) profile, and may find exactly same dy1 even for warmer t1.
                        self.y[self.kcore:] = self.y1
                        retries += 1
                else: # retry with smaller step
                    if retries < 2:
                        delta_t *= (params['target_timestep'] / self.dt_yr) ** 0.5
                    else:
                        delta_t *= 0.5
                    msg = '{:>50}'.format('timestep {:.2e} > limit {:.2e}'.format(self.dt_yr, params['max_timestep']))
                    limit = 'dt'
                    # force static to start from fresh y profile.
                    # otherwise will start from current (depleted) profile, and may find exactly same dy1 even for warmer t1.
                    self.y[self.kcore:] = self.y1
                    retries += 1

                if abs(delta_t) > params['max_abs_delta_t']:
                    limit = 'maxdt1'
                    msg = '{:>50}'.format('delta_t {:.2f} > limit {:.2f}'.format(delta_t, params['max_abs_delta_t']))
                    delta_t = params['max_abs_delta_t']
                elif delta_t / old_delta_t > params['max_ratio_delta_t']:
                    limit = 'Dt1'
                    msg = '{:>50}'.format('new_delta_t/old_delta_t {:.2f} > limit {:.2f}'.format(delta_t / old_delta_t, params['max_ratio_delta_t']))
                    delta_t = old_delta_t * params['max_ratio_delta_t']

                if self.step > 0 and 'debug_retries' in list(params):
                    if params['debug_retries']:
                        dy1 = prev_y1-self.y[-1] if prev_y1 > 0 else 0
                        print(msg + ' (retries {:>2n}, delta_t {:>5.2f}, dt_yr {:>10.2e}, dy1 {:>5.3f}, new_delta_t {:>5.2f})'.format(retries, old_delta_t, self.dt_yr, dy1, delta_t))

                if retries == 10:
                    self.status = ConvergenceError('reached max number of retries for evolve step')
                    raise self.status
                    break # exit retry loop

            if 'full_profiles' in list(params):
                if params['full_profiles']:
                    if not hasattr(self, 'profiles'): self.profiles = {}
                    assert self.step not in list(self.profiles), 'profiles dict already has entry for step {}'.format(self.step)
                    self.profiles[self.step] = self.get_profile()
            self.walltime = time.time() - start_time
            self.delta_y1 = prev_y1 - self.y[-1] if prev_y1 > 0 else 0
            self.append_history()

            # realtime output
            if self.status != 'okay': limit = 'fail'
            k_grady = self.k_gradient_top if self.k_gradient_top else -1
            k_shell = self.k_shell_top if self.k_shell_top else -1
            iters_rain = self.iters_rain if hasattr(self, 'iters_rain') else -1
            mhe_rerr = self.rel_mhe_error if hasattr(self, 'rel_mhe_error') else 0
            if self.step % stdout_interval == 0:
                stdout_data = self.step, self.iters, iters_rain, retries, limit, \
                    params[which_t], self.teff, self.rtot, delta_t, self.dt_yr, self.age_gyr, \
                    self.nz_gradient, self.nz_shell, k_grady, k_shell, \
                    self.y[-1], mhe_rerr, self.walltime
                print(stdout_format.format(*stdout_data))

            if self.status != 'okay': # stop gracefully and print reason
                print('stopping:', self.status, 'y1={}'.format(self.y[-1]))
                break # from top-level evolve loop

            # catch stopping condition
            if params[which_t] < end_t:
                # took a good last step
                done = True
            elif self.teff < params['min_teff']:
                done = True
            else:
                # took a normal good step
                previous_entropy = self.entropy
                previous_t = params[which_t]
                self.step += 1

                # these are normally set in static, but recalculate here in case
                # self.y has changed since that routine was called. this can happen if a candidate
                # static model saw rainout, but then evolve did a retry and subsequently found none.
                self.nz_gradient = len(np.where(np.diff(self.y) < 0)[0])
                k_shell = self.k_shell_top if self.k_shell_top else -1
                self.nz_shell = max(0, k_shell - self.kcore)

                # self.prev_y = np.copy(self.y)

                self.age_gyr += self.dt_yr * 1e-9
                # self.delta_s = delta_s # erg K^-1 g^-1
                # self.eps_grav = eps_grav
                # self.luminosity = luminosity
                prev_y1 = np.copy(self.y[-1])

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
                'mhe':self.mhe,
                'nz_gradient': self.nz_gradient,
                'nz_shell': self.nz_shell,
                'mz_env': self.mz_env,
                'mz': self.mz,
                'bulk_z': self.bulk_z,
                'status': self.status,
                'kcore':self.kcore,
                'ktrans':self.ktrans,
                'k_shell_top':self.k_shell_top,
                'k_gradient_bot':self.k_gradient_bot,
                'k_gradient_top':self.k_gradient_top,
                'walltime':self.walltime,
                'delta_t':self.delta_t,
                'delta_y1':self.delta_y1,
                'ymax_trans':self.ymax_closest
                }

            if not hasattr(self, 'history'):
                self.history = {}

            for key, qty in history_qtys.items():
                if not key in list(self.history):
                    # initialize ndarray for this column
                    if key in ('step', 'iters', 'nz_gradient', 'nz_shell', 'kcore', 'ktrans', 'k_shell_top', 'k_gradient_bot', 'k_gradient_top'):
                        self.history[key] = np.array([], dtype=int)
                    else:
                        self.history[key] = np.array([])
                self.history[key] = np.append(self.history[key], qty)

    def get_profile(self):
        profile = {}
        profile['k'] = np.arange(self.nz)
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
        profile['cp'] = np.copy(self.cp)
        profile['cv'] = np.copy(self.cv)
        profile['rf'] = np.copy(self.rf)
        profile['mf'] = np.copy(self.mf)
        profile['grady'] = np.copy(self.grady)
        profile['brunt_b'] = np.copy(self.brunt_b)
        # profile['pressure_scale_height'] = self.pressure_scale_height
        if hasattr(self, 'ymax'):
            profile['ymax'] = np.copy(self.ymax)
        if hasattr(self, 'ystart'):
            profile['dy'] = np.copy(self.y - self.ystart)
        if self.step > 0:
            profile['delta_s'] = np.copy(self.delta_s)
            profile['eps_grav'] = np.copy(self.eps_grav)
            profile['tds'] = np.copy(self.tds)
            profile['luminosity'] = np.copy(self.luminosity)
        return profile

    def dump_history(self, prefix):
        with open('{}.history'.format(prefix), 'wb') as fw:
            pickle.dump(self.history, fw, 0) # 0 means save as text
        print('wrote history data to {}.history'.format(prefix))

    def dump_profile(self, prefix):
        assert type(prefix) is str, 'output_prefix needs to be a string.'
        with open('{}{}.profile' % (prefix, self.step), 'w') as f:
            pickle.dump(self.get_profile(), f, 0) # 0 means dump as text
        print('wrote profile data to {}{}.profile'.format(prefix, self.step))

    def smooth(self, array, std):
        '''
        moving gaussian filter to smooth an array.
        wrote this to artificially smooth brunt composition term for seismology diagnostic purposes.
        '''
        width = 10 * std
        from scipy.signal import gaussian
        weights = gaussian(width, std=std)
        res = np.zeros_like(array)
        for k in np.arange(len(array)):
            window = array[k-int(width/2):k+int(width/2)]
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

class ConvergenceError(Exception):
    pass
