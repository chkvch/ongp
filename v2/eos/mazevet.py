import numpy as np
from scipy.interpolate import RegularGridInterpolator

class eos_part:
    def __init__(self, path_to_data, which):
        if which not in ('lo', 'hi'): raise ValueError("which must be one of 'lo' or 'hi'.")
        self.names = 'logp', 'logt', 'logrho', 'logu', 'rhot', 'ut'
        self.data = np.genfromtxt(f'{path_to_data}/maz_pt_{which}t.dat', names=self.names)

        self.logpvals = np.unique(self.data['logp'])
        self.logtvals = np.unique(self.data['logt'])

        self.npts_p = len(self.logpvals)
        self.npts_t = len(self.logtvals)
        self.logrho_on_nodes = np.zeros((self.npts_p, self.npts_t))
        self.logu_on_nodes = np.zeros((self.npts_p, self.npts_t))
        self.rhot_on_nodes = np.zeros((self.npts_p, self.npts_t)) # dlnrho_dlnt_const_p
        self.ut_on_nodes = np.zeros((self.npts_p, self.npts_t)) # dlnu_dlnt_const_p

        for i, logpval in enumerate(self.logpvals):
            data_this_logp = self.data[self.data['logp'] == logpval]
            for j, logtval in enumerate(self.logtvals):
                data_this_logp_logt = data_this_logp[data_this_logp['logt'] == logtval]
                self.logrho_on_nodes[i, j] = data_this_logp_logt['logrho']
                self.logu_on_nodes[i, j] = data_this_logp_logt['logu']
                self.rhot_on_nodes[i, j] = data_this_logp_logt['rhot']
                self.ut_on_nodes[i, j] = data_this_logp_logt['ut']

        pt_basis = (self.logpvals, self.logtvals)
        self._get_logrho = RegularGridInterpolator(pt_basis, self.logrho_on_nodes, bounds_error=False, fill_value=None)
        self._get_logu = RegularGridInterpolator(pt_basis, self.logu_on_nodes, bounds_error=False, fill_value=None)
        self._get_rhot = RegularGridInterpolator(pt_basis, self.rhot_on_nodes, bounds_error=False, fill_value=None)
        self._get_ut = RegularGridInterpolator(pt_basis, self.ut_on_nodes, bounds_error=False, fill_value=None)

        self.spline_kwargs = {'kx':3, 'ky':3}

    def get(self, logp, logt):
        res = {}
        logrho = res['logrho'] = self._get_logrho((logp, logt))
        logu = res['logu'] = self._get_logu((logp, logt))
        rhot = self._get_rhot((logp, logt))
        ut = self._get_ut((logp, logt))

        delta = -rhot
        cp = 10 ** logu / 10 ** logt * ut - 10 ** logp / 10 ** logrho / 10 ** logt * rhot
        res['delta'] = delta
        res['cp'] = cp
        res['grada'] = 10 ** logp * delta / 10 ** logt / 10 ** logrho / cp

        return res

class eos:
    def __init__(self, path_to_data):
        self.eos_lo = eos_part(path_to_data, 'lo')
        self.eos_hi = eos_part(path_to_data, 'hi')
        self.logtcut = np.log10(800.) # hi starts at ~807 K, lo ends at ~600 K

        self.names = 'logrho', 'logu', 'delta', 'cp', 'grada'

    def get(self, logp, logt):
        res = {}
        for qty in self.names:
            res[qty] = -99 * np.ones_like(logp)

        res_lot = self.eos_lo.get(logp[logt < self.logtcut], logt[logt < self.logtcut])
        for qty in self.names:
            res[qty][logt < self.logtcut] = res_lot[qty]

        res_hit = self.eos_hi.get(logp[logt > self.logtcut], logt[logt > self.logtcut])
        for qty in self.names:
            res[qty][logt > self.logtcut] = res_hit[qty]

        # blend between 500 and 1000 K
        logthi = np.log10(1e3)
        logtlo = np.log10(500)
        avg = 0.5 * (logtlo + logthi)
        wid = 0.5 * (logthi - logtlo)
        # logphi = np.interp(np.log10(logthi), logt, logp) # profile logp at t=thi
        # logplo = np.interp(np.log10(logtlo), logt, logp) # profile logp at t=tlo
        # try interpolating just in temperature
        res_hi = self.eos_hi.get(logp[abs(logt - avg) < wid], logt[abs(logt - avg) < wid])
        res_lo = self.eos_lo.get(logp[abs(logt - avg) < wid], logt[abs(logt - avg) < wid])
        alpha_t = (logt[abs(logt - avg) < wid] - logtlo) / (logthi - logtlo)
        for qty in self.names:
            res[qty][abs(logt - avg) < wid] = alpha_t * res_hi[qty] + (1. - alpha_t) * res_lo[qty]

        return res

    def get_logrho(self, logp, logt):
        return self.get(logp, logt)['logrho']
    def get_logu(self, logp, logt):
        return self.get(logp, logt)['logu']
