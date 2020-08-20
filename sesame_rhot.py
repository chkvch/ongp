from scipy.interpolate import RectBivariateSpline as rbs
import numpy as np

class eos:
    def __init__(self, path_to_data=None):
        if not path_to_data:
            import os
            path_to_data = os.environ['ongp_data_path']
        self.path = f'{path_to_data}/sesame_water7150.dat'
        self.names = 'rho', 't', 'p', 'u' # P in GPa, u in kJ g^-1
        self.data = np.genfromtxt(self.path, names=self.names, skip_header=1)
        self.data['p'] *= 1e10
        self.data['u'] *= 1e10
        
        self.rhovals = np.unique(self.data['rho'])
        self.tvals = np.unique(self.data['t'])

        self.logp = np.zeros((len(self.rhovals), len(self.tvals)))
        self.logu = np.zeros((len(self.rhovals), len(self.tvals)))
        for i, rhoval in enumerate(self.rhovals):
            logp_this_rho = np.log10(self.data['p'][self.data['rho'] == rhoval])
            logu_this_rho = np.log10(self.data['u'][self.data['rho'] == rhoval])
            t_this_rho = self.data['t'][self.data['rho'] == rhoval]
            for j, tval in enumerate(self.tvals):
                self.logp[i, j] = logp_this_rho[t_this_rho == tval]
                self.logu[i, j] = logu_this_rho[t_this_rho == tval] if not np.isnan(logu_this_rho[t_this_rho == tval]) else -99
                                
        self.logrhovals = np.log10(self.rhovals)
        self.logtvals = np.log10(self.tvals)

        self.spline_kwargs = {'kx':3, 'ky':3}

    def _get_logp(self, lgr, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logp, **self.spline_kwargs)(lgr, lgt, grid=False)
    def _get_logu(self, lgr, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logu, **self.spline_kwargs)(lgr, lgt, grid=False)
    def _get_prho(self, lgr, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logp, **self.spline_kwargs)(lgr, lgt, grid=False, dx=1)
    def _get_pt(self, lgr, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logp, **self.spline_kwargs)(lgr, lgt, grid=False, dy=1)
    def _get_urho(self, lgr, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logu, **self.spline_kwargs)(lgr, lgt, grid=False, dx=1)
    def _get_ut(self, lgr, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logu, **self.spline_kwargs)(lgr, lgt, grid=False, dy=1)

    def get(self, logrho, logt):
        if type(logrho) in (np.float64, float, int): logrho = np.array([np.float64(logrho)])
        if type(logt) in (np.float64, float, int): logt = np.array([np.float64(logt)])
        if len(logrho) != len(logt):
            if len(logrho) == 1:
                logrho = np.ones_like(logt) * logrho[0]
            elif len(logt) == 1:
                logt = np.ones_like(logrho) * logt[0]
            else:
                raise ValueError('got unequal lengths {} and {} for logrho and logt and neither is equal to 1.'.format(len(logp), len(logt)))
                
        res = {}
                
        # delta = -(dlnrho/dlnT)_P
        # cp = (du/dT)_P - P/rho^2(drho/dT)_P
        logp = res['logp'] = self._get_logp(logrho, logt)
        logu = res['logu'] = self._get_logu(logrho, logt)
        # ut = self.get_ut(logp, logt) # (dlnu/dlnT)_P
        # rhot = self.get_rhot(logp, logt) # (dlnrho/dlnT)_P
        # t = 10 ** logt
        # rho = 10 ** logrho
        # p = 10 ** logp
        # u = 10 ** logu
        # delta = -rhot
        # cp = ut * u / t - p / rho / t * rhot
        # grada = p * delta / t / rho / cp

        dlnp_dlnrho_const_t = self._get_prho(logrho, logt)
        dlnp_dlnt_const_rho = self._get_pt(logrho, logt)

        dlnrho_dlnp_const_t = dlnp_dlnrho_const_t ** -1
        dlnrho_dlnt_const_p = - dlnp_dlnt_const_rho / dlnp_dlnrho_const_t # triple product rule
        
        dlnu_dlnrho_const_t = self._get_urho(logrho, logt)
        dlnu_dlnt_const_rho = self._get_ut(logrho, logt)
        dlnu_dlnt_const_p = dlnu_dlnrho_const_t * dlnrho_dlnt_const_p + dlnu_dlnt_const_rho # chain rule
        logu = self._get_logu(logrho, logt)
        
        res['chirho'] = dlnp_dlnrho_const_t
        res['chit'] = dlnp_dlnt_const_rho
        res['delta'] = - dlnrho_dlnt_const_p
        res['cp'] = 10 ** logu / 10 ** logt * dlnu_dlnt_const_p \
            - 10 ** logp / 10 ** logrho / 10 ** logt * dlnrho_dlnt_const_p
        res['grada'] = 10 ** logp * res['delta'] / 10 ** logt / 10 ** logrho / res['cp']

        return res
