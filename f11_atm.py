import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
import gp_configs.app_config as app_cfg
import logging
import config_const as conf

log = logging.getLogger(__name__)
logging.basicConfig(filename=app_cfg.logfile, filemode='w', format=conf.FORMAT)
log.setLevel(conf.log_level)

'''
this module interpolates in the tabulated model atmospheres of
Fortney, Ikoma, Nettelmann, Guillot, and Marley 2011 for J, S, U, N.
ApJ 729:32
doi:10.1088/0004-637X/729/1/32
'''

class atm:

    def __init__(self, path_to_data, planet='jup', print_table=False):

        assert planet in ['jup', 'sat', 'u', 'n'], 'planet label %s not recognized. choose from jup, sat, u, n.' % planet
        self.planet = planet
        log.debug('planet is %s', self.planet)

        names = 'g', 'teff', 't10', 'teff_dim_sun', 't10_dim_sun', 'tint'
        log.debug('reading table from %s/f11_atm_%s.dat' % (path_to_data, self.planet))
        self.data = np.genfromtxt('%s/f11_atm_%s.dat' % (path_to_data, self.planet), delimiter='&', names=names)
        file_length = len(open('%s/f11_atm_%s.dat' % (path_to_data, self.planet)).readlines(  ))
        g_step = 12
        if planet in ['jup', 'sat']:
            g_step = 12
        if planet in ['u', 'n']:
            g_step = 9
        for i in np.arange(0, file_length, g_step):
            self.data['g'][i+1:i+g_step] = self.data['g'][i]

        self.g_grid = np.unique(self.data['g'])
        self.tint_grid = np.unique(self.data['tint'])
        npts_g = len(self.g_grid)
        npts_tint = len(self.tint_grid)

        t10 = np.zeros((npts_g, npts_tint)) # 1.0 Lsun
        teff = np.zeros((npts_g, npts_tint)) # 1.0 Lsun
        t10_dim_sun = np.zeros((npts_g, npts_tint)) # 0.7 Lsun
        teff_dim_sun = np.zeros((npts_g, npts_tint)) # 0.7 Lsun

        for m, gval in enumerate(self.g_grid):
            at_this_g = self.data['g'] == gval
            data_for_this_g = self.data[at_this_g]
            for n, tintval in enumerate(self.tint_grid):
                t10[m, n] = data_for_this_g[data_for_this_g['tint'] == tintval]['t10']
                teff[m, n] = data_for_this_g[data_for_this_g['tint'] == tintval]['teff']

                t10_dim_sun[m, n] = data_for_this_g[data_for_this_g['tint'] == tintval]['t10_dim_sun']
                teff_dim_sun[m, n] = data_for_this_g[data_for_this_g['tint'] == tintval]['teff_dim_sun']

            if t10[m, -1] == -1: # at least one blank element at high tint
                last_nan = np.where(np.diff(t10[m, :] < 0))[0][-1] + 1
                tint = self.tint_grid[last_nan]
                # extrapolate for this tint
                # print 'g=%f: last nan for 10 is at tint = %f' % (gval, tint)
                # print tint, self.tint_grid[last_nan - 1], self.tint_grid[last_nan - 2]
                alpha = (tint - self.tint_grid[last_nan -2]) / (self.tint_grid[last_nan - 1] - self.tint_grid[last_nan - 2])

                t10[m, last_nan] = alpha * t10[m, last_nan - 1] + (1. - alpha) * t10[m, last_nan - 2]
                teff[m, last_nan] = alpha * teff[m, last_nan - 1] + (1. - alpha) * teff[m, last_nan - 2]

                t10_dim_sun[m, last_nan] = alpha * t10_dim_sun[m, last_nan - 1] + (1. - alpha) * t10_dim_sun[m, last_nan - 2]
                teff_dim_sun[m, last_nan] = alpha * teff_dim_sun[m, last_nan - 1] + (1. - alpha) * teff_dim_sun[m, last_nan - 2]

            if t10[m, 0] == -1: # at least one blank element at low tint
                first_nan = np.where(np.diff(t10[m, :] < 0))[0][0] # first sign change
                tint = self.tint_grid[first_nan]
                alpha = (tint - self.tint_grid[first_nan + 2]) / (self.tint_grid[first_nan + 1] - self.tint_grid[first_nan + 2])

                t10[m, first_nan] = alpha * t10[m, first_nan + 1] + (1. - alpha) * t10[m, first_nan + 2]
                teff[m, first_nan] = alpha * teff[m, first_nan + 1] + (1. - alpha) * teff[m, first_nan + 2]

                t10_dim_sun[m, first_nan] = alpha * t10_dim_sun[m, first_nan + 1] + (1. - alpha) * t10_dim_sun[m, first_nan + 2]
                teff_dim_sun[m, first_nan] = alpha * teff_dim_sun[m, first_nan + 1] + (1. - alpha) * teff_dim_sun[m, first_nan + 2]

            self.data['t10'][self.data['g'] == gval] = t10[m, :][::-1]
            self.data['t10_dim_sun'][self.data['g'] == gval] = t10_dim_sun[m, :][::-1]
            self.data['teff'][self.data['g'] == gval] = teff[m, :][::-1]
            self.data['teff_dim_sun'][self.data['g'] == gval] = teff_dim_sun[m, :][::-1]

            if print_table:
                print '%5s %5s %6s %5s' % ('g', 'tint', 't10', 'teff')
                for n, tintval in enumerate(self.tint_grid):
                    print '%5.1f %5.1f %6.1f %6.1f' % (gval, tintval, self.data['t10'][self.data['g'] == gval][npts_tint - n - 1], self.data['teff'][self.data['g'] == gval][npts_tint - n - 1])
                    # print '%5.1f %5.1f %6.1f %5.1f' % (gval, tintval, self.data['t10'][self.data['g'] == gval][npts_tint - n - 1], self.data['teff'][self.data['g'] == gval][npts_tint - n -1])
                print

        self.get_t10 = RegularGridInterpolator((self.g_grid, self.tint_grid), t10)
        self.get_teff = RegularGridInterpolator((self.g_grid, self.tint_grid), teff)
        self.get_t10_dim_sun = RegularGridInterpolator((self.g_grid, self.tint_grid), t10_dim_sun)
        self.get_teff_dim_sun = RegularGridInterpolator((self.g_grid, self.tint_grid), teff_dim_sun)

    def get_tint(self, g, t10):
        """give g (mks) and t10 (K) as arguments, does a root find to obtain the
        intrinsic temperature tint (K)."""

        assert t10 > 0., 'get_tint got a negative t10 %f' % t10
        def zero_me(tint):
            # print tint, t10 - self.get_t10((g, tint))
            try:
                return t10 - self.get_t10((g, tint))
            except ValueError:
                print 'failed to interpolate for t10 at g = %f, tint = %f' % (g, tint)

        # for brackets on tint, consider the nodes for the two nearest values in g. take the
        # intersection of their tint values for which t10, teff are defined, and use min/max
        # of that set as our bracketing values. (avoids choosing as brackets points for which
        # t10, teff are not defined)
        if not min(self.g_grid) <= g <= max(self.g_grid):
            raise ValueError('g value %f outside bounds of f11 atmosphere table for %s' % (g, self.planet))
        g_bin = np.where(self.g_grid - g > 0)[0][0]
        minmax = lambda z: (min(z), max(z))
        g_lo, g_hi = self.g_grid[g_bin - 1], self.g_grid[g_bin]

        data_g_lo = self.data[self.data['g'] == g_lo]
        data_g_lo = data_g_lo[data_g_lo['t10'] > 0]
        min_tint_g_lo = min(data_g_lo['tint'])
        max_tint_g_lo = max(data_g_lo['tint'])

        data_g_hi = self.data[self.data['g'] == g_hi]
        data_g_hi = data_g_hi[data_g_hi['t10'] > 0]
        min_tint_g_hi = min(data_g_hi['tint'])
        max_tint_g_hi = max(data_g_hi['tint'])

        min_tint = max(min_tint_g_lo, min_tint_g_hi)
        max_tint = min(max_tint_g_lo, max_tint_g_hi)

        return brentq(zero_me, min_tint, max_tint)
