import numpy as np
from scipy.interpolate import RegularGridInterpolator, splrep, splev
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

    def __init__(self, path_to_data=None, planet='jup', print_table=False, force_teq=None):

        if not path_to_data:
            import os
            path_to_data = os.environ['ongp_data_path']
        self.planet = planet
        # log.debug('planet is %s', self.planet)

        # setting usecols based on the use case and then passing that to genfromtxt
        # lets us only load the relevant columns, then subsequent code can be uniform across cases
        if self.planet in ('jup', 'sat'):
            self.table_path = '%s/f11_atm_%s.dat' % (path_to_data, self.planet)
            names = 'g', 'teff_10', 't10_10', 'teff_07', 't10_07', 'tint'
            g_step = 12
            # usecols = {
                # '1.0j':(0, 1, 2, 5),
                # '0.7j':(0, 3, 4, 5)
            # }
            # usecols['1.0s'] = usecols['1.0j'] # column indices have same meaning for jup and sat tables
            # usecols['0.7s'] = usecols['0.7j'] # column indices have same meaning for jup and sat tables
            usecols = (0, 1, 2, 3, 4, 5)

            # if not flux_level: # pick a default based on planet
                # flux_level = {'jup':'1.0j', 'sat':'1.0s'}[self.planet]
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
            # if not flux_level: # default to 1.0u or 1.0n
                # flux_level = '1.0' + self.planet
            usecols = usecols['1.0' + self.planet]
        else:
            raise ValueError('planet label %s not recognized. choose from jup, sat, u, n.' % self.planet)

        # print 'planet %s, flux level %s' % (planet, flux_level)

        # log.debug('reading table from %s' % self.table_path)
        self.data = np.genfromtxt(self.table_path, delimiter='&', names=names, usecols=usecols)
        file_length = len(open(self.table_path).readlines())

        if force_teq:
            # shift T_int column from the published one to a different one.
            # option 1 messes up the uniformity of the grid in T_int:
#            current_teq4 = self.data['teff'] ** 4 - self.data['tint'] ** 4
#            self.data['tint'] = (self.data['tint'] ** 4 + current_teq4 - jupiter_modified_teq ** 4) ** 0.25
            # option 2 works simply:
            if self.planet == 'jup':
                existing_teq = 109.9
            elif self.planet == 'sat':
                existing_teq = 81.3
            else:
                raise ValueError
            self.data['tint'] = (self.data['tint'] ** 4 + existing_teq ** 4 - force_teq ** 4) ** 0.25
        self.force_teq = force_teq

        for i in np.arange(0, file_length, g_step):
            self.data['g'][i+1:i+g_step] = self.data['g'][i]

        self.g_grid = np.unique(self.data['g'])
        self.tint_grid = np.unique(self.data['tint'])
        npts_g = len(self.g_grid)
        npts_tint = len(self.tint_grid)

        t = {}
        t_columns = names[1:-1] # ('teff_10', 't10_10', 'teff_07', 't10_07') in the j/s case; (teff, t1, t10) in the u/n case
        for t_column in t_columns:
            t[t_column] = np.zeros((npts_g, npts_tint))

        for m, gval in enumerate(self.g_grid):
            data_for_this_g = self.data[self.data['g'] == gval]
            for n, tintval in enumerate(self.tint_grid):
                for t_column in t_columns:
                    t[t_column][m, n] = data_for_this_g[data_for_this_g['tint'] == tintval][t_column]

            if print_table:
                print('BEFORE')
                # print('%8s ' * (2 + len(t_columns))) % (('g', 'tint') + t_columns)
                header_fmt = '{:>8s} ' * (2 + len(t_columns))
                print(header_fmt.format('g', 'tint', *t_columns))
                for n, tintval in enumerate(self.tint_grid):
                    fmt = '{:>8.1f} ' * (2 + len(t_columns))
                    res_this_row = {}
                    for name in t_columns:
                        res_this_row[name] = self.data[name][self.data['g'] == gval][npts_tint - n - 1]
                    print(fmt.format(gval, tintval, *[res_this_row[key] for key in list(res_this_row)]))

            if t['t10_10'][m, -1] == -1: # this g block has at least one blank element at high tint. should only happen for j/s
                assert not 't1' in t.keys(), 'found a blank (-1) element in u/n atm tables.'
                last_nan = np.where(np.diff(t['t10_10'][m, :] < 0))[0][-1] + 1
                tint = self.tint_grid[last_nan]
                alpha = (tint - self.tint_grid[last_nan -2]) / (self.tint_grid[last_nan - 1] - self.tint_grid[last_nan - 2])

                for t_column in t_columns:
                    t[t_column][m, last_nan] = alpha * t[t_column][m, last_nan - 1] + (1. - alpha) * t[t_column][m, last_nan - 2]
            # assert not t['t10_07'][m, -1] == -1
            if t['t10_07'][m, -1] == -1: # this g block has at least one blank element at high tint. should only happen for j/s
                assert not 't1' in t.keys(), 'found a blank (-1) element in u/n atm tables.'
                last_nan = np.where(np.diff(t['t10_07'][m, :] < 0))[0][-1] + 1
                tint = self.tint_grid[last_nan]
                alpha = (tint - self.tint_grid[last_nan -2]) / (self.tint_grid[last_nan - 1] - self.tint_grid[last_nan - 2])

                for t_column in t_columns:
                    t[t_column][m, last_nan] = alpha * t[t_column][m, last_nan - 1] + (1. - alpha) * t[t_column][m, last_nan - 2]

            if t['t10_10'][m, 0] == -1: # this g block has at least one blank element at low tint. should only happen for j/s
                assert not 't1' in t.keys(), 'found a blank (-1) element in u/n atm tables.'
                first_nan = np.where(np.diff(t['t10_10'][m, :] < 0))[0][0] # first sign change
                tint = self.tint_grid[first_nan]
                alpha = (tint - self.tint_grid[first_nan + 2]) / (self.tint_grid[first_nan + 1] - self.tint_grid[first_nan + 2])

                for t_column in t_columns:
                    t[t_column][m, first_nan] = alpha * t[t_column][m, first_nan + 1] + (1. - alpha) * t[t_column][m, first_nan + 2]
            # assert not t['t10_07'][m, 0] == -1
            if t['t10_07'][m, 0] == -1: # this g block has at least one blank element at low tint. should only happen for j/s
                assert not 't1' in t.keys(), 'found a blank (-1) element in u/n atm tables.'
                first_nan = np.where(np.diff(t['t10_07'][m, :] < 0))[0][0] # first sign change
                tint = self.tint_grid[first_nan]
                alpha = (tint - self.tint_grid[first_nan + 2]) / (self.tint_grid[first_nan + 1] - self.tint_grid[first_nan + 2])

                for t_column in t_columns:
                    t[t_column][m, first_nan] = alpha * t[t_column][m, first_nan + 1] + (1. - alpha) * t[t_column][m, first_nan + 2]


            for t_column in t_columns:
                self.data[t_column][self.data['g'] == gval] = t[t_column][m, :][::-1]

            if print_table:
                print('AFTER')
                # print('%8s ' * (2 + len(t_columns))) % (('g', 'tint') + t_columns)
                header_fmt = '{:>8s} ' * (2 + len(t_columns))
                print(header_fmt.format('g', 'tint', *t_columns))
                for n, tintval in enumerate(self.tint_grid):
                    fmt = '{:>8.1f} ' * (2 + len(t_columns))
                    res_this_row = {}
                    for name in t_columns:
                        res_this_row[name] = self.data[name][self.data['g'] == gval][npts_tint - n - 1]
                    print(fmt.format(gval, tintval, *[res_this_row[key] for key in list(res_this_row)]))
                print()

        self.get = {}
        for t_column in t_columns:
            self.get[t_column] = RegularGridInterpolator((self.g_grid, self.tint_grid), t[t_column])

    # def get_tint(self, g, t10):
    #     """
    #     given g (mks) and t10 (K) as arguments, does a root find to obtain the
    #     intrinsic temperature tint (K).
    #     """
    #
    #     assert t10 > 0., 'get_tint got a negative t10 %f' % t
    #     def zero_me(tint):
    #         try:
    #             return t10 - self.get['t10']((g, tint))
    #         except ValueError:
    #             print('failed to interpolate for t10 at g = %f, tint = %f' % (g, tint))
    #
    #     # for brackets on tint, consider the nodes for the two nearest values in g. take the
    #     # intersection of their tint values for which t10, teff are defined, and use min/max
    #     # of that set as our bracketing values. (avoids choosing as brackets points for which
    #     # t10, teff are not defined)
    #     if not min(self.g_grid) <= g <= max(self.g_grid):
    #         raise ValueError('g value %f outside bounds of f11 atmosphere table for %s' % (g, self.planet))
    #     g_bin = np.where(self.g_grid - g > 0)[0][0]
    #     minmax = lambda z: (min(z), max(z))
    #     g_lo, g_hi = self.g_grid[g_bin - 1], self.g_grid[g_bin]
    #
    #     data_g_lo = self.data[self.data['g'] == g_lo]
    #     data_g_lo = data_g_lo[data_g_lo['t10'] > 0]
    #     min_tint_g_lo = min(data_g_lo['tint'])
    #     max_tint_g_lo = max(data_g_lo['tint'])
    #
    #     data_g_hi = self.data[self.data['g'] == g_hi]
    #     data_g_hi = data_g_hi[data_g_hi['t10'] > 0]
    #     min_tint_g_hi = min(data_g_hi['tint'])
    #     max_tint_g_hi = max(data_g_hi['tint'])
    # 
    #     min_tint = max(min_tint_g_lo, min_tint_g_hi)
    #     max_tint = min(max_tint_g_lo, max_tint_g_hi)
    #
    #     return brentq(zero_me, min_tint, max_tint)


    def get_tint_teff(self, g, t10, flux_level='10'):
        """
        given g (mks) and t10 (K) as arguments, does a root find to obtain the
        intrinsic temperature tint (K) and effective temperature teff (K).
        """

        assert t10 > 0., 'get_tint got a negative t10 %f' % t
        def zero_me(tint):
            try:
                return t10 - self.get['t10_{:s}'.format(flux_level)]((g, tint))
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
        data_g_lo = data_g_lo[data_g_lo['t10_{:s}'.format(flux_level)] > 0]
        min_tint_g_lo = min(data_g_lo['tint'])
        max_tint_g_lo = max(data_g_lo['tint'])

        data_g_hi = self.data[self.data['g'] == g_hi]
        data_g_hi = data_g_hi[data_g_hi['t10_{:s}'.format(flux_level)] > 0]
        min_tint_g_hi = min(data_g_hi['tint'])
        max_tint_g_hi = max(data_g_hi['tint'])

        min_tint = max(min_tint_g_lo, min_tint_g_hi)
        max_tint = min(max_tint_g_lo, max_tint_g_hi)

        tint = brentq(zero_me, min_tint, max_tint)
        teff = float(self.get['teff_{:s}'.format(flux_level)]((g, tint)))
        if np.isnan(teff):
            _ = np.linspace(min_tint, max_tint)
            while np.isnan(teff):
                _ = np.delete(_, -1)
                teff = splev(tint, splrep(_, self.get['teff_{:s}'.format(flux_level)]((g, _))))
        return tint, teff
