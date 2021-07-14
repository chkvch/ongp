from scipy.interpolate import RectBivariateSpline as rbs
try:
    from importlib import reload
except:
    pass
import numpy as np
import os

class eos:
    def __init__(self, material, path_to_data=None):
        self.material = material
        self.columns = 'logrho', 'logt', 'logp', 'logu', 'logs'
        if not path_to_data: path_to_data = os.environ['ongp_data_path']
        self.data_path = '{}/aneos_{}.dat'.format(path_to_data, self.material)
        self.data = np.genfromtxt(self.data_path, skip_header=0, names=self.columns)

        self.logrhovals = np.unique(self.data['logrho'])
        self.logtvals = np.unique(self.data['logt'])

        self.logp = np.zeros((len(self.logrhovals), len(self.logtvals)))
        self.logs = np.zeros((len(self.logrhovals), len(self.logtvals)))
        self.logu = np.zeros((len(self.logrhovals), len(self.logtvals)))
        for i, logrhoval in enumerate(self.logrhovals):
            logp_this_rho = self.data['logp'][self.data['logrho'] == logrhoval]
            logs_this_rho = self.data['logs'][self.data['logrho'] == logrhoval]
            logu_this_rho = self.data['logu'][self.data['logrho'] == logrhoval]
            logt_this_rho = self.data['logt'][self.data['logrho'] == logrhoval]
            for j, logtval in enumerate(self.logtvals):
                # previously these were all scaled by 1e10 (MJ kg^-1 for s and u, GPa to cgs for p).
                # actually not sure of any units; it seems pressure is already cgs. 
                # need to check s and u against another eos.
                self.logp[i, j] = logp_this_rho[logt_this_rho == logtval]
                self.logs[i, j] = logs_this_rho[logt_this_rho == logtval]
                self.logu[i, j] = logu_this_rho[logt_this_rho == logtval]

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
    def _get_logu(self, lgrho, lgt):
        return rbs(self.logrhovals, self.logtvals, self.logu, **self.spline_kwargs)(lgrho, lgt, grid=False)
        
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
        res['prho'] = res['chirho'] = chirho
        res['pt'] = res['chit'] = chit
        res['rhop'] = dlnrho_dlnp_const_t
        res['rhot'] = dlnrho_dlnt_const_p
        res['gamma1'] = - dlns_dlnrho_const_p / dlns_dlnp_const_rho
        res['grada'] = (1. - chirho / res['gamma1']) / chit

        return res

    def regularize_to_pt(self, outpath):
        from scipy.optimize import brentq
        import time

        print('regularizing %s tables to rectangular in P, T' % self.material)

        npts = 100 # 200 # following the old aneos_*_pt tables
        logpvals = np.linspace(6, 17, npts)
        logtvals = np.linspace(2, 6, npts)

        logrho_on_pt = np.zeros((npts, npts))
        logs_on_pt = np.zeros((npts, npts))
        logu_on_pt = np.zeros((npts, npts))

        t0 = time.time()
        for i, logpval in enumerate(logpvals):
            for j, logtval in enumerate(logtvals):
                try:
                    zero_me = lambda logrho: self._get_logp(logrho, logtval) - logpval
                    logrho_on_pt[i, j] = brentq(zero_me, min(self.logrhovals), max(self.logrhovals))
                    logs_on_pt[i, j] = self._get_logs(logrho_on_pt[i, j], logtval)
                    logu_on_pt[i, j] = self._get_logu(logrho_on_pt[i, j], logtval)
                except ValueError:
                    logrho_on_pt[i, j] = np.nan
                    logs_on_pt[i, j] = np.nan
                    logu_on_pt[i, j] = np.nan
                    
            done = i + 1
            remain = len(logpvals) - done
            et = time.time() - t0
            seconds_per_row = et / done
            eta = remain * seconds_per_row
            print('\rrow {}/{}, {:.1f} s elapsed {:.1f} s remain{:20}'.format(done, len(logpvals), et, eta, ''), end='')

        # this is what aneos.py is expecting from aneos_*_pt.dat:
        # self.names = 'logrho', 'logt', 'logp', 'logu', 'logs' # , 'chit', 'chirho', 'gamma1'

        fmt = '%21.16f\t' * 5
        outfile = '{}/aneos_{}_pt_hi-p.dat'.format(outpath, self.material)
        with open(outfile, 'w') as fw:
            for i, logpval in enumerate(logpvals):
                for j, logtval in enumerate(logtvals):
                    line = fmt % (logrho_on_pt[i, j], logtval, logpval, logu_on_pt[i, j], logs_on_pt[i, j])
                    fw.write(line + '\n')

        print('wrote {}'.format(outfile))
