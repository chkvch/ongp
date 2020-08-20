from scipy.interpolate import RectBivariateSpline as rbs
import numpy as np

class eos:
    def __init__(self, path_to_data=None):
        if not path_to_data:
            import os
            path_to_data = os.environ['ongp_data_path']
        self.path = f'{path_to_data}/sesame_water7150.dat'
        self.names = 'p', 't', 'rho', 'u', 'chirho', 'chit' # p, rho, u all cgs
        self.data = np.genfromtxt(self.path, names=self.names)
        
        self.pvals = np.unique(self.data['p'])
        self.tvals = np.unique(self.data['t'])

        self.logrho = np.zeros((len(self.pvals), len(self.tvals)))
        self.logu = np.zeros((len(self.pvals), len(self.tvals)))
        self.chirho = np.zeros((len(self.pvals), len(self.tvals)))
        self.chit = np.zeros((len(self.pvals), len(self.tvals)))
        for i, pval in enumerate(self.pvals):
            logrho_this_p = np.log10(self.data['rho'][self.data['p'] == pval])
            logu_this_p = np.log10(self.data['u'][self.data['p'] == pval])
            chirho_this_p = self.data['chirho'][self.data['p'] == pval]
            chit_this_p = self.data['chit'][self.data['p'] == pval]
            t_this_p = self.data['t'][self.data['p'] == pval]
            for j, tval in enumerate(self.tvals):
                self.logrho[i, j] = logrho_this_p[t_this_p == tval]
                self.logu[i, j] = logu_this_p[t_this_p == tval] if not np.isnan(logu_this_p[t_this_p == tval]) else -99
                self.chirho[i, j] = chirho_this_p[t_this_p == tval]
                self.chit[i, j] = chit_this_p[t_this_p == tval]
                                
        self.logpvals = np.log10(self.pvals)
        self.logtvals = np.log10(self.tvals)

        self.spline_kwargs = {'kx':3, 'ky':3}

    def get_logrho(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logrho, **self.spline_kwargs)(lgp, lgt, grid=False)
    def get_logu(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logu, **self.spline_kwargs)(lgp, lgt, grid=False)
    def get_chirho(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.chirho, **self.spline_kwargs)(lgp, lgt, grid=False)
    def get_chit(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.chit, **self.spline_kwargs)(lgp, lgt, grid=False)
    # def get_sp_h(self, lgp, lgt):
        # return rbs(self.logpvals, self.logtvals, self.logs, **self.spline_kwargs)(lgp, lgt, dx=1, grid=False)
    # def get_st_h(self, lgp, lgt):
        # return rbs(self.logpvals, self.logtvals, self.logs, **self.spline_kwargs)(lgp, lgt, dy=1, grid=False)
    def get_ut(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logu, **self.spline_kwargs)(lgp, lgt, dy=1, grid=False)
    def get_rhot(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logrho, **self.spline_kwargs)(lgp, lgt, dy=1, grid=False)

    def get(self, logp, logt):
        if type(logp) in (np.float64, float, int): logp = np.array([np.float64(logp)])
        if type(logt) in (np.float64, float, int): logt = np.array([np.float64(logt)])
        if len(logp) != len(logt):
            if len(logp) == 1:
                logp = np.ones_like(logt) * logp[0]
            elif len(logt) == 1:
                logt = np.ones_like(logp) * logt[0]
            else:
                raise ValueError('got unequal lengths {} and {} for logp and logt and neither is equal to 1.'.format(len(logp), len(logt)))
                
                
        # delta = -(dlnrho/dlnT)_P
        # cp = (du/dT)_P - P/rho^2(drho/dT)_P
        logrho = self.get_logrho(logp, logt)
        logu = self.get_logu(logp, logt)
        ut = self.get_ut(logp, logt) # (dlnu/dlnT)_P
        rhot = self.get_rhot(logp, logt) # (dlnrho/dlnT)_P
        t = 10 ** logt
        rho = 10 ** logrho
        p = 10 ** logp
        u = 10 ** logu
        delta = -rhot
        cp = ut * u / t - p / rho / t * rhot
        grada = p * delta / t / rho / cp
                
        res = {
            'logrho':logrho,
            'logu':logu,
            'chirho':self.get_chirho(logp, logt),
            'chit':self.get_chit(logp, logt),
            'cp':cp,
            'delta':delta,
            'grada':grada
        }

        return res
