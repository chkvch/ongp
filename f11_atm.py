import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
# import gp_configs.app_config as app_cfg
# import logging
# import config_const as conf

# log = logging.getLogger(__name__)
# logging.basicConfig(filename=app_cfg.logfile, filemode='w', format=conf.FORMAT)
# log.setLevel(conf.log_level)

class atm:

    '''
    this class interpolates in the tabulated model atmospheres of
    Fortney, Ikoma, Nettelmann, Guillot, and Marley 2011 for J, S, U, N.
    ApJ 729:32
    doi:10.1088/0004-637X/729/1/32
    '''

    def __init__(self, path_to_data, planet='jup', print_table=False, flux_level=None, jupiter_modified_teq=None):

        self.planet = planet
        # log.debug('planet is %s', self.planet)

        # setting usecols based on the use case and then passing that to genfromtxt
        # lets us only load the relevant columns, then subsequent code can be uniform across cases
        if self.planet in ('jup', 'sat'):
            self.table_path = '%s/f11_atm_%s.dat' % (path_to_data, self.planet)
            names = 'g', 'teff', 't10', 'tint'
            g_step = 12
            usecols = {
                '1.0j':(0, 1, 2, 5),
                '0.7j':(0, 3, 4, 5)
            }
            usecols['1.0s'] = usecols['1.0j']
            usecols['0.7s'] = usecols['0.7j']

            if not flux_level: # pick a default based on planet
                flux_level = {'jup':'1.0j', 'sat':'1.0s'}[self.planet]
        elif self.planet in ('u', 'n'):
            self.table_path = '%s/f11_atm_un.dat' % path_to_data
            names = 'g', 'teff', 't10', 't1', 'tint'
            g_step = 9
            usecols = {
                '0.12n':(0, 1, 2, 3, 13),
                '1.0n':(0, 4, 5, 6, 13),
                '1.0u':(0, 7, 8, 9, 13),
                '1.8u':(0, 10, 11, 12, 13)
            }
            if not flux_level: # default to 1.0u or 1.0n
                flux_level = '1.0' + self.planet
        else:
            raise ValueError('planet label %s not recognized. choose from jup, sat, u, n.' % self.planet)

        # print 'planet %s, flux level %s' % (planet, flux_level)

        # log.debug('reading table from %s' % self.table_path)
        self.data = np.genfromtxt(self.table_path, delimiter='&', names=names, usecols=usecols[flux_level])
        file_length = len(open(self.table_path).readlines())

        if jupiter_modified_teq:
            # shift T_int column from the published one consistent with Teq=109.9 K to a different one
            assert self.planet == 'jup', 'jupiter_modified_teq has no meaning unless planet is jup.'
            # this option messes up the uniformity of the grid in T_int
#            current_teq4 = self.data['teff'] ** 4 - self.data['tint'] ** 4
#            self.data['tint'] = (self.data['tint'] ** 4 + current_teq4 - jupiter_modified_teq ** 4) ** 0.25
            # this option works easily
            self.data['tint'] = (self.data['tint'] ** 4 + 109.9 ** 4 - jupiter_modified_teq ** 4) ** 0.25
        self.jupiter_modified_teq = jupiter_modified_teq

        for i in np.arange(0, file_length, g_step):
            self.data['g'][i+1:i+g_step] = self.data['g'][i]

        self.g_grid = np.unique(self.data['g'])
        self.tint_grid = np.unique(self.data['tint'])
        npts_g = len(self.g_grid)
        npts_tint = len(self.tint_grid)

        t = {}
        t_columns = names[1:-1] # ('teff', 't10') in the j/s case; (teff, t1, t10) in the u/n case
        for t_column in t_columns:
            t[t_column] = np.zeros((npts_g, npts_tint))

        for m, gval in enumerate(self.g_grid):
            data_for_this_g = self.data[self.data['g'] == gval]
            for n, tintval in enumerate(self.tint_grid):
                for t_column in t_columns:
                    t[t_column][m, n] = data_for_this_g[data_for_this_g['tint'] == tintval][t_column]

            if t['t10'][m, -1] == -1: # this g block has at least one blank element at high tint. should only happen for j/s
                assert not 't1' in t.keys(), 'found a blank (-1) element in u/n atm tables.'
                last_nan = np.where(np.diff(t['t10'][m, :] < 0))[0][-1] + 1
                tint = self.tint_grid[last_nan]
                alpha = (tint - self.tint_grid[last_nan -2]) / (self.tint_grid[last_nan - 1] - self.tint_grid[last_nan - 2])

                for t_column in t_columns:
                    t[t_column][m, last_nan] = alpha * t[t_column][m, last_nan - 1] + (1. - alpha) * t[t_column][m, last_nan - 2]

            if t['t10'][m, 0] == -1: # this g block at least one blank element at low tint. should only happen for j/s
                assert not 't1' in t.keys(), 'found a blank (-1) element in u/n atm tables.'
                first_nan = np.where(np.diff(t['t10'][m, :] < 0))[0][0] # first sign change
                tint = self.tint_grid[first_nan]
                alpha = (tint - self.tint_grid[first_nan + 2]) / (self.tint_grid[first_nan + 1] - self.tint_grid[first_nan + 2])

                for t_column in t_columns:
                    t[t_column][m, first_nan] = alpha * t[t_column][m, first_nan + 1] + (1. - alpha) * t[t_column][m, first_nan + 2]


            for t_column in t_columns:
                self.data[t_column][self.data['g'] == gval] = t[t_column][m, :][::-1]

            if print_table:
                print('%8s ' * (2 + len(t_columns))) % (('g', 'tint') + t_columns)
                for n, tintval in enumerate(self.tint_grid):
                    try:
                        print('%8.2f ' * (2 + len(t_columns))) % ((gval, tintval) + \
                            ( \
                                self.data['teff'][self.data['g'] == gval][npts_tint - n - 1],
                                self.data['t10'][self.data['g'] == gval][npts_tint - n - 1],
                                self.data['t1'][self.data['g'] == gval][npts_tint - n - 1],
                            ))
                    except ValueError:
                        # no t1
                        print('%8.2f ' * (2 + len(t_columns))) % ((gval, tintval) + \
                            ( \
                                self.data['teff'][self.data['g'] == gval][npts_tint - n - 1],
                                self.data['t10'][self.data['g'] == gval][npts_tint - n - 1],
                            ))

        self.get = {}
        for t_column in t_columns:
            self.get[t_column] = RegularGridInterpolator((self.g_grid, self.tint_grid), t[t_column])

    def get_tint(self, g, t10):
        """
        give g (mks) and t10 (K) as arguments, does a root find to obtain the
        intrinsic temperature tint (K).
        """

        assert t10 > 0., 'get_tint got a negative t10 %f' % t
        def zero_me(tint):
            try:
                return t10 - self.get['t10']((g, tint))
            except ValueError:
                print('failed to interpolate for t10 at g = %f, tint = %f' % (g, tint))

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


    def get_tint_teff(self, g, t10):
        """
        give g (mks) and t10 (K) as arguments, does a root find to obtain the
        intrinsic temperature tint (K) and effective temperature teff (K).
        """

        assert t10 > 0., 'get_tint got a negative t10 %f' % t
        def zero_me(tint):
            try:
                return t10 - self.get['t10']((g, tint))
            except ValueError:
                print('failed to interpolate for t10 at g = %f, tint = %f' % (g, tint))

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

        tint = brentq(zero_me, min_tint, max_tint)
        teff = float(self.get['teff']((g, tint)))
        return tint, teff
