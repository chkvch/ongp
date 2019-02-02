from scipy.interpolate import RectBivariateSpline as rbs
import scvh
import numpy as np
from scipy.interpolate import splrep, splev


class eos:
    def __init__(self, path_to_data):

        self.columns = 'logrho', 'logt', 'logp', 'logu', 'logs'
        self.path = '{}/reos_water_pt.dat'.format(path_to_data)
        self.data = np.genfromtxt(self.path, names=self.columns, usecols=(0, 1, 2, 3, 4))

        self.logpvals = np.unique(self.data['logp'])
        self.logtvals = np.unique(self.data['logt'])

        self.logrho = np.zeros((len(self.logpvals), len(self.logtvals)))
        self.logu = np.zeros((len(self.logpvals), len(self.logtvals)))
        self.logs = np.zeros((len(self.logpvals), len(self.logtvals)))
        for i, logpval in enumerate(self.logpvals):
            self.logrho[i] = self.data['logrho'][self.data['logp'] == logpval]
            self.logu[i] = self.data['logu'][self.data['logp'] == logpval]
            self.logs[i] = self.data['logs'][self.data['logp'] == logpval]

            if np.any(np.isnan(self.logrho[i])):
                okay = np.where(np.logical_not(np.isnan(self.logrho[i])))[0]
                bad = np.where(np.isnan(self.logrho[i]))[0]
                tck = splrep(self.logtvals[okay], self.logrho[i, okay], k=1)
                self.logrho[i, bad] = splev(self.logtvals[bad], tck, ext=0)
            if np.any(np.isnan(self.logu[i])):
                okay = np.where(np.logical_not(np.isnan(self.logu[i])))[0]
                bad = np.where(np.isnan(self.logu[i]))[0]
                tck = splrep(self.logtvals[okay], self.logu[i, okay], k=1)
                self.logu[i, bad] = splev(self.logtvals[bad], tck, ext=0)
            if np.any(np.isnan(self.logs[i])):
                okay = np.where(np.logical_not(np.isnan(self.logs[i])))[0]
                bad = np.where(np.isnan(self.logs[i]))[0]
                tck = splrep(self.logtvals[okay], self.logs[i, okay], k=1)
                self.logs[i, bad] = splev(self.logtvals[bad], tck, ext=0)

            # class scipy.interpolate.RectBivariateSpline(x, y, z, bbox=[None, None, None, None], kx=3, ky=3, s=0)
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

    # from scipy/interpolate.fitpack/parder.f:
    # c    ier=10: invalid input data (see restrictions)
    # c
    # c  restrictions:
    # c   mx >=1, my >=1, 0 <= nux < kx, 0 <= nuy < ky, kwrk>=mx+my
    # c   lwrk>=mx*(kx+1-nux)+my*(ky+1-nuy)+(nx-kx-1)*(ny-ky-1),
    # c   tx(kx+1) <= x(i-1) <= x(i) <= tx(nx-kx), i=2,...,mx
    # c   ty(ky+1) <= y(j-1) <= y(j) <= ty(ny-ky), j=2,...,my


    def get_logrho(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logrho, **self.spline_kwargs)(lgp[::-1], lgt[::-1], grid=False)[::-1]
    def get_logu(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logu, **self.spline_kwargs)(lgp[::-1], lgt[::-1], grid=False)[::-1]
    def get_logs(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logs, **self.spline_kwargs)(lgp[::-1], lgt[::-1], grid=False)[::-1]
    def get_sp(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logs, **self.spline_kwargs)(lgp[::-1], lgt[::-1], dx=1, grid=False)[::-1]
    def get_st(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logs, **self.spline_kwargs)(lgp[::-1], lgt[::-1], dy=1, grid=False)[::-1]
    def get_rhop(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logrho, **self.spline_kwargs)(lgp[::-1], lgt[::-1], dx=1, grid=False)[::-1]
    def get_rhot(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logrho, **self.spline_kwargs)(lgp[::-1], lgt[::-1], dy=1, grid=False)[::-1]

    # general method for getting eos quantities
    def get(self, logp, logt):
        logs = self.get_logs(logp, logt)
        sp = self.get_sp(logp, logt)
        st = self.get_st(logp, logt)
        grada = - sp / st

        logrho = self.get_logrho(logp, logt)
        rhop = self.get_rhop(logp, logt)
        rhot = self.get_rhot(logp, logt)

        chirho = 1. / rhop # dlnP/dlnrho|T
        chit = -1. * rhot / rhop # dlnP/dlnT|rho
        gamma1 = 1. / (sp ** 2 / st + rhop) # dlnP/dlnrho|s

        logu = self.get_logu(logp, logt)

        res =  {
            'grada':grada,
            'logrho':logrho,
            'logs':logs,
            'gamma1':gamma1,
            'chirho':chirho,
            'chit':chit,
            'rhop':rhop,
            'rhot':rhot,
            'logu':logu
            }
        return res

    # convenience methods
    def get_grada(self, logp, logt):
        return self.get(logp, logt)['grada']

    def get_gamma1(self, logp, logt):
        return self.get(logp, logt)['gamma1']
