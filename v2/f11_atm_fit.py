import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

class atm:

    """a class to implement the [F11] model atmospheres for Jupiter and Saturn,
    as fit analytically by [LC13].

    [F11] Fortney, J. J., Ikoma, M., Nettelmann, N., Guillot, T., & Marley, M. S. 2011,
    ApJ, 729, 32

    [LC13] Leconte, J., & Chabrier, G. 2013, Nature Geoscience, 6, 347

    """

    def __init__(self, planet='jup', print_table=False):
        assert planet in ['jup', 'sat'], 'planet label %s not recognized. choose from jup, sat' % planet
        self.planet = planet
        # log.debug('planet is %s', self.planet)

        if self.planet == 'jup':
            self.clo = 78.4e0
            self.klo = 0.0055e0
            self.t0lo = -122e0
            self.betalo = -0.182e0
            self.gammalo = 2.09e0

            self.chi = -283e0
            self.khi = 198e0
            self.t0hi = 143e0
            self.betahi = -0.114e0
            self.gammahi = 0.454e0

            self.tmid = 224e0
            self.dt = 25e0
        elif self.planet == 'sat':
            self.clo = 62.7e0
            self.klo = 0.00202e0
            self.t0lo = -97.1e0
            self.betalo = -0.18e0
            self.gammalo = 2.31e0

            self.chi = -1680e0
            self.khi = 831e0
            self.t0hi = 141e0
            self.betahi = -0.0721e0
            self.gammahi = 0.293e0

            self.tmid = 225e0
            self.dt = 25e0

    def get_t10(self, g, tint):

        if tint > self.tmid + self.dt:
            t10hi = self.chi + self.khi * g ** self.betahi * (tint - self.t0hi) ** self.gammahi
            return t10hi
        elif tint < self.tmid - self.dt:
            t10lo = self.clo + self.klo * g ** self.betalo * (tint - self.t0lo) ** self.gammalo
            return t10lo
        else:
            t10hi = self.chi + self.khi * g ** self.betahi * (tint - self.t0hi) ** self.gammahi
            t10lo = self.clo + self.klo * g ** self.betalo * (tint - self.t0lo) ** self.gammalo
            alpha = (tint - (self.tmid - self.dt)) / (2. * self.dt)
            return alpha * t10hi + (1. - alpha) * t10lo

    def get_tint(self, g, t10):
        """takes g (mks) and t10 (K) as arguments; uses the analytic fitting formula of Leconte+Chabrier2013 to return the
        intrinsic temperature tint (K)."""

        zero_me = lambda tint_: self.get_t10(g, tint_) - t10

        t0 = 50
        t1 = 3e3
        if not zero_me(t0) * zero_me(t1) < 0:
            print('no root between bounds t0=%g, t1=%g' % (t0, t1))
            raise ValueError

        from scipy.optimize import brentq
        tint = brentq(zero_me, t0, t1)

        return tint
