import numpy as np
import const
from scipy.interpolate import interp1d # , splrep, splev
from scipy.integrate import trapz, solve_ivp

class ConvergenceError(Exception):
    pass
class AtmError(Exception):
    pass
class EOSError(Exception):
    pass
class EnergyError(Exception):
    pass
class UnphysicalParameterError(Exception):
    pass
class MoistIntegrationError(Exception):
    pass
import time
from importlib import reload

class planet:
    def __init__(self, planet, nz=512, hhe_eos_option='chabrier', z_eos_option='aneos', path_to_data=None, mesh_surf_amp=1e5, mesh_surf_width=5e-2):

        self.planet = planet # important for setting atmosphere boundary condition (here) and total mass (in self.static)

        if not path_to_data:
            from pathlib import Path
            # path to data not specified; default to ./eos/data
            path_to_data = Path(__file__).absolute().parent / 'data'

        # initialize h/he and z eos
        if hhe_eos_option == 'mh13_scvh':
            from eos import mh13_scvh
            self.hhe_eos = mh13_scvh.eos(path_to_data)
        elif hhe_eos_option == 'scvh':
            from eos import scvh
            self.hhe_eos = scvh.eos(path_to_data)
        elif hhe_eos_option == 'chabrier':
            from eos import chabrier
            self.hhe_eos = chabrier.eos(path_to_data)
        else:
            raise ValueError(f'hhe_eos_option {hhe_eos_option} not recognized.')
        if z_eos_option == 'aneos':
            from eos import aneos
            self.z_eos = aneos.eos(path_to_data, material='ice')
        elif z_eos_option == 'mazevet' or 'maz':
            from eos import mazevet
            self.z_eos = mazevet.eos(path_to_data)
        else:
            raise ValueError(f'z_eos_option {z_eos_option} not recognized.')
        self.hhe_eos_option = hhe_eos_option
        self.z_eos_option = z_eos_option

        self.onset_of_condensation = True
        self.t1_when_condensation_begins = 0.0
        self.t10_when_condensation_begins = 0.0
        self.t1_when_stable_zone_forms = 0.0
        self.t10_when_stable_zone_forms = 0.0


        # initialize model atmospheres (outer boundary condition for models)
        if False:
            import f11_atm # fortney+2011 tabulated model atmospheres; not working at present for jupiter test case
            self.atm = f11_atm.atm(path_to_data=path_to_data, planet=self.planet)
        else:
            import f11_atm_fit # leconte + chabrier analytic fit to the same model atmospheres; more flexible for bad initial guesses
            self.atm = f11_atm_fit.atm(planet=self.planet)

        # initialize lagrangian mesh we solve the equations on
        self.nz = nz
        f0 = np.linspace(0, 1, self.nz)
        density_f0 = 1. / np.diff(f0)
        density_f0 = np.insert(density_f0, 0, density_f0[0])
        density_f0 += mesh_surf_amp * f0 * np.exp((f0 - 1.) / mesh_surf_width) * np.mean(density_f0) # boost mesh density near surface
        out = np.cumsum(1. / density_f0)
        out -= out[0]
        out /= out[-1]
        self.mesh = out

        # initialize structure variables
        self.y = np.zeros(self.nz) # helium mass fraction
        self.z = np.zeros(self.nz) # h2o mass fraction
        self.p = np.zeros(self.nz) # pressure (dyne cm^-2)
        self.t = np.zeros(self.nz) # temperature (K)
        self.rho = np.zeros(self.nz) # density (g cm^-3)
        self.rho_hhe = np.zeros(self.nz) # density of h/he subsystem (g cm^-3)
        self.rho_z = np.zeros(self.nz) # density of water subsystem (g cm^-3)
        self.cp_hhe = np.zeros(self.nz) # specific heat of h/he subsystem at constant pressure (erg K^-1 g^-1)
        self.grada = np.zeros(self.nz) # adiabatic temperature gradient (dlnt/dlnp)_s
        # energy equation in evolve tracks cooling of h/he via changes in its entropy
        self.entropy_hhe = np.zeros(self.nz) # entropy of h/he subsystem (erg K^-1 g^-1);
        # energy equation in evolve tracks cooling of h2o via changes in its internal energy u and specific volume 1/rho
        self.u_z = np.zeros(self.nz) # internal energy of h2o subsystem (erg g^-1)
        self.alpha = np.zeros(self.nz) # stability criterion cf. Friedson+Gonzalez 2017 eq. 3
        # self.kappa = np.zeros(self.nz) # opacities taken from Valencia et al., 2013
        # self.grav = np.zeros(self.nz) # gravity at mass coordinate
        # self.grad_rad = np.zeros(self.nz) # radiative gradient from Leconte et al., eq. (52)

    def get_eos_results(self):
        '''
        query self.hhe_eos and self.z_eos for densities and derivatives throughout the current structure model.
        most importantly this sets rho and grada everywhere (important for the static model),
        and evaluates the entropy everywhere (important for the evolutionary model).
        (in reality it's entropy for h/he, internal energy and density for z; just due to the columns available
        in the respective tables.)
        '''
        if np.any(np.isnan(np.log10(self.p))):
            raise EOSError(f'have {len(np.log10(self.p)[np.isnan(np.log10(self.p))])} nans in logp')
        elif np.any(np.isnan(np.log10(self.t))):
            raise EOSError(f'have {len(np.log10(self.t)[np.isnan(np.log10(self.t))])} nans in logt')
        elif np.any(self.y[self.kcore:] <= 0.):
            raise UnphysicalParameterError('one or more bad y')
        elif np.any(self.y >= 1.):
            raise UnphysicalParameterError('one or more bad y')
        elif np.any(self.z < 0.):
            raise UnphysicalParameterError('one or more bad z')
        elif np.any(self.z > 1.):
            raise UnphysicalParameterError('one or more bad z')

        # hhe eos results. pass not true y but y_xy, since hhe eos describes just the h/he system.
        # as a result only evaluate it outside the core, because within the core y_xy is undefined.
        hhe_res = self.hhe_eos.get(np.log10(self.p[self.kcore:]), np.log10(self.t[self.kcore:]), self.y[self.kcore:] / (1. - self.z[self.kcore:]))
        z_res = self.z_eos.get(np.log10(self.p), np.log10(self.t)) # z eos results: evaluate everywhere
        self.rho_hhe[self.kcore:] = 10 ** hhe_res['logrho']
        self.rho_z = 10 ** z_res['logrho']
        self.rho[:self.kcore] = self.rho_z[:self.kcore] # core: pure z
        self.rho[self.kcore:] = ((1. - self.z[self.kcore:]) / self.rho_hhe[self.kcore:] + \
            self.z[self.kcore:] / self.rho_z[self.kcore:]) ** -1. # envelope: additive volume rule

        if False: # aneos z eos not set up to provide delta, so skip this
            # these delta and cp expressions follow from additive volume
            # (densities add in reciprocal weighted by mass fractions; internal energies simply add weighted by their mass fractions)
            self.delta = self.z * self.rho / self.rho_z * z_res['delta'] # z part
            self.delta[self.kcore:] += (1. - self.z[self.kcore:]) * self.rho[self.kcore:] / self.rho_hhe[self.kcore:] * hhe_res['delta'] # h/he part
            self.cp_z = z_res['cp']
            self.cp_hhe[self.kcore:] = hhe_res['cp']
            self.cp = self.z * self.cp_z # z part
            self.cp[self.kcore:] += (1. - self.z[self.kcore:]) * hhe_res['cp'] # h/he part
            self.grada = self.p * self.delta / self.t / self.rho / self.cp # e.g., scheibe, nettelmann & redmer eq. (5)

            self.grada_z = self.p * z_res['delta'] / self.t / self.rho_z / self.cp_z
        else:
            # just ignore the effect of z on the dry adiabatic gradient
            self.grada[self.kcore:] = hhe_res['grada']
            self.grada[:self.kcore] = 0.

        self.entropy_hhe[self.kcore:] = 10 ** hhe_res['logs']
        self.u_z = 10 ** z_res['logu']

        # also store internal energy for h/he for updated timestep calculation (based on total energy of model)
        self.u_hhe = 10 ** hhe_res['logu']

        # internal energy of the mixture is a weighted sum of the two components
        self.u = self.z * self.u_z
        self.u[self.kcore:] += (1. - self.z[self.kcore:]) * self.u_hhe

    def integrate_continuity(self):
        ''' dm = 4 pi r^2 rho dr '''
        q = np.zeros_like(self.rho)
        q[0] = 0.
        q[1:] = 3. * self.dm / 4 / np.pi / self.rho[1:]
        self.r = np.cumsum(q) ** (1. / 3)

    def integrate_hydrostatic(self):
        ''' dp / dr = - rho g '''
        dp = const.cgrav * self.m[1:] * self.dm / 4. / np.pi / self.r[1:] ** 4
        psurf = 1e6 # changing to 1 bar 11/04/2020
        self.p[-1] = psurf
        self.p[:-1] = psurf + np.cumsum(dp[::-1])[::-1]

        if np.any(np.isnan(self.p)):
            nnan = len(self.p[np.isnan(self.p)])
            raise EOSError(f'{nnan} nans in pressure after integrate hydrostatic on iteration {self.static_iters}.')

    def integrate_temperature_dry(self, t_top):
        '''
        just do simple integration of dry adiabat from surface
        t_top :: temperature at atm top (1 or 10 bars)
        '''

        # create an interpolant for grad_dry so integrator can quickly evaluate at arbitrary pressures
        interp_grada = interp1d(self.p, self.grada)
        def dtdp(p, t):
            return t / p * interp_grada(p)

        p_eval = self.p[self.kcore:][::-1] # evaluate resulting temperature profile on pressures from base of stable region to core top
        sol = solve_ivp(dtdp, (p_eval[0], p_eval[-1]), np.array([t_top]), t_eval=p_eval) # start from top of homogeneous outer envelope
        assert sol.success, 'failed in integrate_temperature'
        self.t[self.kcore:] = sol.y[0][::-1]
        self.t[:self.kcore] = self.t[self.kcore] # isothermal core

        if np.any(np.isnan(self.t)):
            raise EOSError('%i nans in temperature after integrate gradt on static iteration %i.' % (len(self.t[np.isnan(self.t)]), self.static_iters))

    def static(self, z1, y1, mcore, t1,
        m12=None, z2=None, y2=None,
        teq=None,
        rtot_rtol=1e-5,
        max_iterations=30,
        min_iterations=10,
        debug_iterations=False):
        '''
        relaxation scheme iterates to find a hydrostatic model for the specified t10.

        z1:     envelope metallicity
        y1:     envelope helium mass fraction
        mcore:  mass of pure-z core (Earth masses)
        t1:     1-bar temperature (K)

        m12:    optional: mass coordinate (Earth masses) of composition jump from z1, y1 outside to z2, y2 inside
        z2:     optional: water mass fraction below m=m12
        y2:     optional: helium mass fraction below m=m12

        '''

        mass = {'jup':const.mjup, 'sat':const.msat}[self.planet]
        self.m = mass * self.mesh
        self.dm = np.diff(self.m)
        self.p[:] = 1e12
        self.t[:] = 1e4
        self.mcore = mcore

        self.t1 = t1

        # verify that teq is only set if using analytic fit to f11 model atmospheres
        if teq:
            import f11_atm_fit
            assert type(self.atm) is f11_atm_fit.atm
            self.teq = teq
        else:
            import f11_atm
            assert type(self.atm) is f11_atm.atm

        if mcore > 0:
            self.kcore = np.where(self.m < self.mcore * const.mearth)[0][-1]
            # print("self.kcore ", self.kcore)
            self.m[self.kcore] = self.mcore * const.mearth # move grid point to fall exactly at desired core mass
        else:
            self.kcore = 0

        # set mass fractions of helium and heavies.
        # first initialize core as z = 1 and y = 0
        self.y[:self.kcore] = 0.
        self.z[:self.kcore] = 1.
        # now, based on mass of core, set y and z outside of core
        self.y[self.kcore:] = y1
        self.z[self.kcore:] = z1
        self.z1 = z1
        # print("self.y ", self.y)
        # print("self.z ", self.z)

        if m12 or z2 or y2:
            # choosing the transition in mass coordinate has the benefit that the corresponding grid point
            # never changes, as it would for example if we chose a fixed pressure p12.
            if not z2 or not y2 or not m12: raise ValueError('must set all or none of z2, y2, m12.')
            self.m12 = m12
            self.z2 = z2
            self.y2 = y2
            self.k12 = np.where(self.m < self.m12 * const.mearth)[0][-1]
            self.m[self.k12] = self.m12 * const.mearth # move grid point to fall exactly at desired core mass
            self.z[self.kcore:self.k12] = self.z2
            self.y[self.kcore:self.k12] = self.y2

        if debug_iterations: # print header with column names for detailed iteration output
            stdout_names = 'it', 'rtot/re', 'drtot/rtot', 't1', 't10', 'et_ms', 'kstab', 'tbase', 'dtrad'
            print('{:>3} {:>10} {:>12} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}'.format(*stdout_names))

        # relax to find static model
        self.relerr_rtot_iterations = np.array([])
        t0 = time.time()

        for i in np.arange(max_iterations):
            self.static_iters = i
            self.get_eos_results() # call eos and update densities
            self.integrate_continuity() # calculates r from dm and rho
            self.integrate_hydrostatic() # calculates p from m, dm, and r

            # query model atmospheres for intrinsic temperature and effective temperature
            self.gsurf = const.cgrav * self.m[-1] / self.r[-1] ** 2 # in principle different for 1-bar vs. 10-bar surface, but negligible
            # interpolate to get 10-bar temperature, independent variable for the fortney+2011 model atmosphere tables
            self.t10 = interp1d(self.p, self.t, kind='cubic')(1e7) if i > 0 else 1e3 # on initial guess, just pass something that will not fail
            gsurf_atm = self.gsurf * 1e-2 # surface gravity tabulated in m s^-2
            try:
                # if using f11_atm.py
                # self.tint, self.teff = self.atm.get_tint_teff(gsurf_atm, self.t10)

                # if using f11_atm_fit.py
                self.tint = self.atm.get_tint(gsurf_atm, self.t10)
                self.teff = (self.tint ** 4 + self.teq ** 4) ** 0.25 if hasattr(self, 'teq') else -1
            except:
                raise
                raise AtmError(f'atmosphere lookup failed for g={gsurf_atm:.2f} m s^-1, t10={self.t10} K.')

            self.lint = np.pi * 4 * self.r[-1] ** 2 * const.sigma_sb * self.tint ** 4 # intrinsic luminosity sets the timestep

            self.integrate_temperature_dry(t1)

            et_ms = self.et_static = (time.time() - t0)*1e3 # timing

            try:
                # relerr_pc = (self.p[0] - pc_prev) / pc_prev
                # relerr_tc = (self.t[0] - tc_prev) / tc_prev
                relerr_rtot = (self.r[-1] - rtot_prev) / rtot_prev
                # relerr_y1 = (self.y[-1] - y1_prev) / y1_prev
                # rel_std_entropy = np.std(self.entropy[self.kcore:]) / np.mean(self.entropy[self.kcore:])
                self.relerr_rtot_iterations = np.append(self.relerr_rtot_iterations, relerr_rtot)
                stdout_data = i, self.r[-1] / const.rearth, relerr_rtot, self.t1, self.t10, int(et_ms)
                if debug_iterations:
                    if hasattr(self, 'k_stable') and self.k_stable > 0:
                        stdout_data = i, self.r[-1] / const.rearth, relerr_rtot, self.t1, self.t10, int(et_ms), self.k_stable, self.t_base, self.dt_rad
                        print('{:>3n} {:>10.2e} {:>12.2e} {:>6.1f} {:>6.1f} {:>6n} {:>6n} {:>6.2f} {:>6.2f}'.format(*stdout_data))
                    else:
                        print('{:>3n} {:>10.2e} {:>12.2e} {:>6.1f} {:>6.1f} {:>6n}'.format(*stdout_data))
                if abs(relerr_rtot) < rtot_rtol and i >= min_iterations: # success
                    break
            except NameError: # no previous entropy to speak of
                pass
            # pc_prev = self.p[0]
            # tc_prev = self.t[0]
            rtot_prev = self.r[-1]
            # y1_prev = self.y[-1]

            # return # for debugging, this will bail gracefully after first iteration

            if debug_iterations == 'dump':
                # dump structure at each iteration for debugging convergence woes
                debug_info = {}
                [debug_info.update({attr:getattr(self, attr)}) for attr in ('p', 't', 'r', 'y', 'z', 'k_wcz_bot')]
                import pickle
                pickle.dump(debug_info, open(f'debug_{i:03n}.pkl', 'wb'))

        else:
            raise ConvergenceError('static model reached max iterations {} for t1={}.'.format(max_iterations, t1))

        self.rtot = self.r[-1]

    def evolve(self, z1, y1, mcore,
                m12=None,z2=None,y2=None,
                teq=None,
                rtot_rtol=1e-5,
                max_iterations=30,
                debug_iterations=False,
                start_t1=460.,
                end_t1=80.,
                nsteps=151,
                verbosity=1,
                moist_atol=1e-6,
                moist_rtol=1e-8):
        '''
        computes a sequence of hydrostatic models and solves the energy equation for the timestep
        between each pair.
        takes the same arguments as self.static, but rather a single t10 value it takes bracketing
        values corresponding to t10 of the desired starting (hot) model and that of the desired
        ending (cold) model.
        '''
        self.age = 0
        t0 = time.time()

        mass = {'u':const.mura, 'n':const.mnep}[self.planet]
        self.params = {
            'mass':mass,
            'z1':z1,
            'y1':y1,
            'mcore':mcore,
            'm12':m12,
            'z2':z2,
            'y2':y2
        }

        # history dictionary will hold scalar quantities as a function of time
        self.history = {}
        [self.history.update({qty:np.array([], dtype=int)}) for qty in ('step', 'iters')]
        [self.history.update({qty:np.array([], dtype=np.float64)}) for qty in \
            ('t1', 't10', 'tint', 'teff', 'etot', 'rtot', 'dt_yr', 'age', 'y1', 'et', 'pc', 'tc', 't2', 't1', 'zsurf')]

        # profiles dictionary will hold vector quantities, one profile per timestep
        self.profiles = {}

        # write headers for real-time output pertaining to the evolution
        if verbosity > 0:
            header_names = 'step', 'iters', 't1', 't10', 'tint', 'teff', 'etot', 'rtot/re', 'dt_yr', 'age_yr', 'pc', 'tc', 'y_surf', 'z_surf', 'et'
            print('{:>5} {:>6} {:>8} {:>8} {:>8} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*header_names))

        # these were only used for old timestepping calculation
        # previous_entropy_hhe = np.zeros(self.nz)
        # previous_u_z = np.zeros(self.nz)
        # previous_rho_z = np.zeros(self.nz)
        # self.tds_hhe = np.zeros_like(self.p)
        # self.tds_z = np.zeros_like(self.p)

        # models are computed at a preordained sequence of values for the surface temperature.
        t1s = np.linspace(start_t1, end_t1, nsteps)

        etot_prev = 0.
        for step, t1 in enumerate(t1s):

            et = time.time() - t0

            # relax a hydrostatic model for this t10
            if debug_iterations: # print header for debug output in self.static
                print('{:3} {:10} {:10} {:10} {:10} {:10} {:5}'.format('i', 'relerr_pc', 'relerr_tc', 'relerr_r', 'relerr_y1', 'rel_std_s', 'et_ms'))
            self.static(z1, y1, mcore, t1, z2=z2, y2=y2, m12=m12, teq=teq,
                rtot_rtol=rtot_rtol, max_iterations=max_iterations, debug_iterations=debug_iterations,
                moist_atol=moist_atol, moist_rtol=moist_rtol)

            if self.lint == 0:
                print('terminate because intrinsic luminosity has vanished: at equilibrium.')
                return

            if False: # old timestep calculation
                # calculate change in entropy profile and corresponding timestep.
                # this amounts to a simple Euler scheme for doing the time integration.
                # we could instead set up dt_dt10 as a function and use scipy.integrate.solve_ivp.
                entropy_hhe = self.entropy_hhe
                et = time.time() - t0
                if step == 0: # first step
                    dt = 0
                else: # normal step
                    # ds = self.entropy - previous_entropy

                    if True:
                        # for the purposes of calculating the timestep, avoid 'wiggles' we get from the tables
                        # by assuming that the H+He entropy is constant at the value attained at 1 kbar.
                        k_kbar = np.where(self.p < 1e9)[0][0]
                        entropy_hhe[self.kcore:k_kbar] = entropy_hhe[k_kbar]

                    tds_hhe = self.t * (entropy_hhe - previous_entropy_hhe)
                    tds_z = self.u_z - previous_u_z + self.p * (self.rho_z ** -1 - previous_rho_z ** -1)
                    int_tdsdm = trapz(self.z * tds_z + (1. - self.z) * tds_hhe, dx=self.dm)
                    dt = -int_tdsdm / self.lint / const.seconds_per_year
                    if dt <= 0:
                        # raise EnergyError(f'got non-positive timestep at t10={t10}; probably found a bad entropy profile. lint={self.lint}')
                        if True: # experimental: just force a zero timestep
                            dt = 0
                        else: # do what we usually do
                            print(f'got negative timestep at t1={t1}; cannot proceed')
                            return
                    self.age += dt
                    if verbosity > 0:
                        stdout_data = step, self.static_iters, t1, self.t10, self.tint, self.teff, self.r[-1]/const.rearth, \
                            dt, self.age, self.p[0], self.t[0], self.y[-1], self.z[-1], et
                        print('{:>5n} {:>6n} {:>8.1f} {:>8.1f} {:>8.1f} {:>6.1f} {:>8.1f} {:>8.2e} {:>8.2e} {:>8.1e} {:>8.1e} {:>8.3f} {:>8.2e} {:>8.1f}'.format(*stdout_data))

                    # store these so they can be accessed in profiles for troubleshooting
                    self.tds_hhe[:] = tds_hhe[:]
                    self.tds_z[:] = tds_z[:]

            else: # new timestep calculation
                if step == 0: # first step
                    dt = 0
                    # calculate total energy just so it's there, even for initial step
                    eth = trapz(self.u, x=self.m) # total thermal energy of the model, erg
                    egr = -trapz(const.cgrav * self.m[1:] / self.r[1:], x=self.m[1:]) # gravitational binding energy, erg
                    etot = self.etot = eth + egr
                    etot_prev = etot
                    okay = True
                else: # normal step

                    # u[:self.kcore] = self.u_z[:self.kcore]
                    # u[self.kcore:] = self.z[self.kcore:] * self.u_z[self.kcore:] + (1. - self.z[self.kcore:]) * self.u_hhe # internal energy, erg g^-1

                    if np.any(np.isnan(self.u)):
                        raise ValueError('nans in internal energy')

                    eth = trapz(self.u, x=self.m) # total thermal energy of the model, erg
                    egr = -trapz(const.cgrav * self.m[1:] / self.r[1:], x=self.m[1:]) # gravitational binding energy, erg
                    etot = self.etot = eth + egr

                    # output for debugging timestep / energy equation
                    # print(f'eth={eth:g}, egr={egr:g}, etot={etot:g}, etot_prev={etot_prev:g}')

                    detot = etot - etot_prev

                    dt = -detot / self.lint / const.seconds_per_year # timestep, yr
                    okay = True
                    if dt <= 0:
                        # raise EnergyError(f'got non-positive timestep at t10={t10}; probably found a bad entropy profile. lint={self.lint}')
                        if True: # experimental: just force a zero timestep
                            dt = 0
                            # 06022021: important: do not update etot_prev in this case, since the run should continue until the next
                            # model is found that has a lower total energy than in the last good step!
                            okay = False # flag to skip update of etot_prev
                        else: # do what we usually do
                            print(f'got negative timestep at t1={t1}; cannot proceed')
                            return
                    self.age += dt
                    if okay:
                        etot_prev = etot

                    if verbosity > 0:
                        stdout_data = step, self.static_iters, t1, self.t10, self.tint, self.teff, self.etot * 1e-40, self.r[-1]/const.rearth, \
                            dt, self.age, self.p[0], self.t[0], self.y[-1], self.z[-1], et
                        print('{:>5n} {:>6n} {:>8.1f} {:>8.1f} {:>8.1f} {:>6.1f} {:>8.3f} {:>8.2f} {:>8.2e} {:>8.2e} {:>8.1e} {:>8.2e} {:>8.3f} {:>8.2e} {:>8.1f}'.format(*stdout_data))

            # these were used for old timestepping calculation
            # previous_entropy_hhe[:] = entropy_hhe
            # previous_u_z[:] = self.u_z
            # previous_rho_z[:] = self.rho_z

            if True:
                # update history with current values of everything
                self.history['step'] = np.append(self.history['step'], step)
                self.history['iters'] = np.append(self.history['iters'], self.static_iters)
                self.history['t1'] = np.append(self.history['t1'], t1)
                self.history['t10'] = np.append(self.history['t10'], self.t10)
                self.history['tint'] = np.append(self.history['tint'], self.tint)
                self.history['teff'] = np.append(self.history['teff'], self.teff)
                self.history['rtot'] = np.append(self.history['rtot'], self.r[-1])
                self.history['dt_yr'] = np.append(self.history['dt_yr'], dt)
                self.history['age'] = np.append(self.history['age'], self.age)
                self.history['y1'] = np.append(self.history['y1'], self.y[-1])
                self.history['et'] = np.append(self.history['et'], et)
                self.history['pc'] = np.append(self.history['pc'], self.p[0])
                self.history['tc'] = np.append(self.history['tc'], self.t[0])
                self.history['etot'] = np.append(self.history['etot'], self.etot)
                self.history['zsurf'] = np.append(self.history['zsurf'], self.z[-1])
                # if hasattr(self, 't1'):
                #     self.history['t1'] = np.append(self.history['t1'], self.t1)

                # save current profiles
                self.profiles[step] = {}
                for qty in 'p', 't', 'm', 'r', \
                        'rho', 'rho_hhe', 'rho_z', 'entropy_hhe', 'u_z', \
                        'y', 'z', 'grada', \
                        'xvap', 'xh2', 'xhe', 'alpha', 'kappa', 'grad_rad', 'grav', \
                        'u', 'k_stable':
                        # 'tds_hhe', 'tds_z':
                    self.profiles[step].update({qty:np.copy(getattr(self, qty))})

if __name__ == '__main__':

    print('test static jupiter model')

    z1 = 0.06 # outer envelope metallicity
    z2 = 0.08 # inner envelope metallicity
    mcore = 10. # mass (mearth) of pure-Z core
    m12 = 270. # mass coordinate (mearth) of inner/outer envelope transition

    y_xy = 0.27
    y1 = y_xy * (1-z1)
    y2 = y_xy * (1-z2)

    t1 = 166. # 1-bar temperature, K

    e = planet('jup')
    e.static(z1, y1, mcore, t1, z2=z2, y2=y2, m12=m12, teq=102.5, debug_iterations=True, rtot_rtol=1e-5) # set debug_iterations=False to remove verbose output during iterations

    # detailed model info is now available in attributes of e, e.g.,
    # e.p gives the array of pressures, e.t the temperature, e.z and e.y the mass fractions of Z and He, etc.
    # many scalars are defined as well.
    print(f'surface gravity = {e.gsurf:.2f}')
    print(f'teff = {e.teff:.2f}')
    print(f'tint = {e.tint:.2f}')
    print(f'rtot = {e.rtot:.5e}')
    print(f'pc   = {e.p[0]:.5e}')
    print(f'tc   = {e.t[0]:.5e}')