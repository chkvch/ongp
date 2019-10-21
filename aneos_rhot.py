from scipy.interpolate import RectBivariateSpline as rbs
try:
    from importlib import reload
except:
    pass
import numpy as np

class eos:
    def __init__(self, path_to_data, material):
        self.columns = 'logrho', 'logt', 'logp', 'logu', 'logs'
        self.data_path = '{}/raw_or_unused_eos_data/aneos/aneos_{}.dat'.format(path_to_data, material)
        self.data = np.genfromtxt(self.data_path, skip_header=0, names=self.columns)

        self.logrhovals = np.unique(self.data['logrho'])
        self.logtvals = np.unique(self.data['logt'])

        self.logp = np.zeros((len(self.logrhovals), len(self.logtvals)))
        self.logs = np.zeros((len(self.logrhovals), len(self.logtvals)))
        for i, logrhoval in enumerate(self.logrhovals):
            logp_this_rho = self.data['logp'][self.data['logrho'] == logrhoval]
            logs_this_rho = self.data['logs'][self.data['logrho'] == logrhoval]
            logt_this_rho = self.data['logt'][self.data['logrho'] == logrhoval]
            for j, logtval in enumerate(self.logtvals):
                self.logp[i, j] = logp_this_rho[logt_this_rho == logtval] + 10. # GPa to dyne cm^-2
                self.logs[i, j] = logs_this_rho[logt_this_rho == logtval] + 10. # kJ g^-1 to erg g^-1

        del(self.data)

            # class scipy.interpolate.RectBivariateSpline(  x, y, z, bbox=[None, None, None, None], kx=3, ky=3, s=0)
            #     Bivariate spline approximation over a rectangular mesh.
            #     Can be used for both smoothing and interpolating data.
            #     x,y : array_like
            #     1-D arrays of coordinates in strictly ascending order.
            #     z : array_like
            #     2-D array of data with shape (x.size,y.size).
            #     bbox : array_like, optional
            #     Sequence of length 4 specifying the boundary of the rectangular approximation domain. By default, bbox=[min(x,tx),max(x,tx), min(y,ty),max(y,ty)].
            #     kx, ky : ints, optional
            #     Degrees of the bivariate spline. Default is 3.
            #     s : float, optional
            #     Positive smoothing factor defined for estimation condition: sum((w[i]*(z[i]-s(x[i], y[i])))**2, axis=0) <= s Default is s=0, which is for interpolation.

        self.spline_kwargs = {'kx':3, 'ky':3}

    def _get_logp(self, lgrho, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logp, **self.spline_kwargs)(lgrho, lgt, grid=False)
    def _get_logs(self, lgrho, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logs, **self.spline_kwargs)(lgrho, lgt, grid=False)
    def _get_prho(self, lgrho, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logp, **self.spline_kwargs)(lgrho, lgt, grid=False, dx=1)
    def _get_pt(self, lgrho, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logp, **self.spline_kwargs)(lgrho, lgt, grid=False, dy=1)
    def _get_srho(self, lgrho, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logs, **self.spline_kwargs)(lgrho, lgt, grid=False, dx=1)
    def _get_st(self, lgrho, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logs, **self.spline_kwargs)(lgrho, lgt, grid=False, dy=1)

    def get(self, logrho, logt):
        logp = self._get_logp(logrho, logt)
        logs = self._get_logs(logrho, logt)

        dlnp_dlnrho_const_t = chirho = self._get_prho(logrho, logt)
        dlnp_dlnt_const_rho = chit = self._get_pt(logrho, logt)
        dlnrho_dlnp_const_t = dlnp_dlnrho_const_t ** -1
        dlnrho_dlnt_const_p = - dlnp_dlnt_const_rho / dlnp_dlnrho_const_t

        dlns_dlnrho_const_t = self._get_srho(logrho, logt)
        dlns_dlnt_const_rho = self._get_st(logrho, logt)
        dlns_dlnrho_const_p = dlns_dlnrho_const_t + dlns_dlnt_const_rho * (-chirho / chit)
        dlns_dlnp_const_rho = dlns_dlnt_const_rho / chit

        res = {}
        res['logp'] = logp
        res['logs'] = logs
        res['prho'] = res['chirho'] = dlnp_dlnrho_const_t
        res['pt'] = res['chit'] = dlnp_dlnt_const_rho
        res['rhop'] = dlnrho_dlnp_const_t
        res['rhot'] = dlnrho_dlnt_const_p
        res['gamma1'] = - dlns_dlnrho_const_p / dlns_dlnp_const_rho
        res['grada'] = (1. - chirho / res['gamma1']) / chit

        return res
