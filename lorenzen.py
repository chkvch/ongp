from scipy import interpolate
from scipy.optimize import brentq, minimize
from scipy.interpolate import splev
# from scipy.interpolate import BSpline
import numpy as np
import time
import sys

class hhe_phase_diagram:
    """interpolates in the Lorenzen et al. 2011 phase diagram to return maximum soluble helium fraction,
    either by number (xmax) or by mass (ymax), as a function of P, T.

    the way this is done is to look look at each P-node from the tables and construct a cubic spline
    representing x-T for this P. when seeking xmax for a general P-T point, the two splines corresponding
    to the two P-nodes bracketing the desired P value are evaluated at T, returning two bracketing
    values of xmax. these two values are finally linearly interpolated (in logP) to the desired P value.

    this approach is motivated by the tables being regularly spaced in logP - 1, 2, 4, 10, 24 Mbar.
    x and T are irregular.

    for this dataset, applying this approach to a saturn-like adiabat seems to generally give Y inversions
    between 1 and 2 Mbar.  so something else might be in order -- perhaps first make x-P splines and then
    interpolate in logT instead. the workaround implemented for now is to ignore the 1 Mbar data and
    extrapolate down from higher P instead.

    """

    def __init__(self, path_to_data=None, order=3, extrapolate_to_low_pressure=False):

        if extrapolate_to_low_pressure:
            raise NotImplementedError('extrapolate_to_low_pressure not implemented in lorenzen')

        if not path_to_data:
            path_to_data = '/Users/chris/ongp/data'

        self.columns = 'x', 'p', 't' # x refers to the helium number fraction
        data = np.genfromtxt('{}/demixHHe_Lorenzen.dat'.format(path_to_data), names=self.columns)

        # for 1 Mbar curve, max T is reached for two consecutive points:
        # [ 0.24755  0.29851]
        # [ 1.  1.]
        # [ 7141.  7141.]
        # insert a tiny made-up bump so that critical temperature can be solved for.
        k1, k2 = np.where(data['t'] == 7141.)[0]
        self.data = {}
        # self.data['p'] = np.insert(data['p'], k2, 1.)
        # self.data['x'] = np.insert(data['x'], k2, 0.5 * (0.29851 + 0.24755))
        # self.data['t'] = np.insert(data['t'], k2, 8141.)
        self.data['p'] = np.delete(data['p'], k2)
        self.data['x'] = np.delete(data['x'], k2)
        self.data['t'] = np.delete(data['t'], k2)
        del(data)

        self.pvals = np.unique(self.data['p'])

        # self.tck = {} # dict used to look up knots, coefficients, order of bspline for any p in self.pvals
        self.tck_x = {} # dict used to look up knots, coefficients, order of bspline for any p in self.pvals
        self.tck_t = {} # dict used to look up knots, coefficients, order of bspline for any p in self.pvals
        self.tcrit = {} # maximum (critical) temperature of each spline
        self.zcrit = {} # z value ("t" in tck, called z to avoid confusion with temperature) of critical temperature
        # self.splinex = {} # dict used to look up x component of bspline curve for any p in self.pvals
        # self.splinet = {} # dict used to look up t component of bspline curve for any p in self.pvals

        self.initialize_splines()

    def splinex(self, pval, z):
        return splev(z, self.tck_x[pval])
        # t, c, k = self.tck_x[pval]
        # return BSpline(t, c, k)(z)

    def splinet(self, pval, z):
        return splev(z, self.tck_t[pval])
        # t, c, k = self.tck_t[pval]
        # return BSpline(t, c, k)(z)

    def initialize_splines(self):

        # throw away points colder than this at high x
        # tmin_hi = {1:6, 2:7, 3:7.9, 4:7.9, 10:7.9, 24:7}

        self.x_clean = {}
        self.t_clean = {}
        self.minmax_z = {}

        for pval in self.pvals:

            x = self.data['x'][self.data['p'] == pval]
            t = self.data['t'][self.data['p'] == pval] * 1e-3

            if True:
                # for all curves, discard points corresponding to Y < 0.01 or so
                t = t[x > 3e-3]
                x = x[x > 3e-3]

                # if pval == 24: # discard points corresponding to Y<0.2 or so
                #     t = t[x > 1e-1]
                #     x = x[x > 1e-1]

                # z is "t" in b-spline language
                z = np.linspace(0, 1, len(x) - 2)
                z = np.append(np.zeros(3), z)
                z = np.append(z, np.ones(3))
                # self.tck[pval] = [z, [x, t], 3]
                self.tck_x[pval] = [z, x, 3]
                self.tck_t[pval] = [z, t, 3]

                # get z and x corresponding to t maximum (critical point) for this curve.
                # this should be trivial to get from spline coefficients analytically,
                # but get it numerically for convenience
                def one_div_t_this_spline(z):
                    return self.splinet(pval, z) * -1
                sol = minimize(one_div_t_this_spline, 0.5, tol=1e-2)
                assert sol['success']
                self.tcrit[pval] = sol['fun'] * -1
                self.zcrit[pval] = sol['x'][0]

                izcrit = int(self.zcrit[pval] * len(z))
                izskip = {1:20, 2:5, 4:5, 10:5, 24:5}[pval]

                # now that we know zcrit, discard more extraneous points and recalculate splines.
                x_ = np.array([])
                t_ = np.array([])
                for iz, (xval, tval, zval) in enumerate(zip(x, t, z)):
                    if zval > self.zcrit[pval] and tval < 4:
                        continue
                    elif abs(iz - izcrit) > 2 and iz % izskip != 0:
                        continue
                    else:
                        x_ = np.append(x_, xval)
                        t_ = np.append(t_, tval)

                # print numbers to give an idea of downsampling
                # print('{}: {} -> {}'.format(pval, len(x), len(x_)))

                del(x, t, z)
                x = x_
                t = t_
                if pval == 1:
                    x = x[2:]
                    t = t[2:]
                elif pval == 2:
                    x = x[4:]
                    t = t[4:]

            z = np.linspace(0, 1, len(x) - 2)
            z = np.append(np.zeros(3), z)
            z = np.append(z, np.ones(3))
            # self.tck[pval] = [z, [x, t], 3]
            self.tck_x[pval] = [z, x, 3]
            self.tck_t[pval] = [z, t, 3]

            # get z and x corresponding to t maximum (critical point) for this curve.
            # this should be trivial to get from spline coefficients analytically,
            # but get it numerically for convenience
            def one_div_t_this_spline(z):
                return self.splinet(pval, z) * -1
            sol = minimize(one_div_t_this_spline, 0.5, tol=1e-3)
            assert sol['success']
            self.tcrit[pval] = sol['fun'] * -1
            self.zcrit[pval] = sol['x'][0]

            # # provide min and max values of z for sake of root find bounds in miscibility_gap
            self.minmax_z[pval] = z[0], z[-1]
            # # provide scrubbed versions for diagnostic purposes
            self.x_clean[pval] = x
            self.t_clean[pval] = t


    def miscibility_gap(self, p, t):
        '''for a given pressure and temperature, return the helium number fraction of the
        helium-poor and helium-rich phases.

        args
        p: pressure in Mbar = 1e12 dyne cm^-2
        t: temperature in kK = 1e3 K
        )
        returns
        (ylo, yhi): helium mass fractions of helium-poor and helium-rich phases
        '''
        if p < min(self.pvals):
            return 'failed'
            raise ValueError('p value {} outside bounds for phase diagram'.format(p))
        elif p > max(self.pvals):
            return 'failed'
            raise ValueError('p value {} outside bounds for phase diagram'.format(p))

        plo = self.pvals[self.pvals < p][-1]
        phi = self.pvals[self.pvals > p][0]
        assert p > plo
        assert p < phi

        if t > self.tcrit[plo] or t > self.tcrit[phi]:
            return 'stable'

        zmin_plo, zmax_plo = self.minmax_z[plo]
        zmin_phi, zmax_phi = self.minmax_z[phi]
        # for abundance in helium-poor phase, search between zmin and z(tcrit)
        # for abundance in helium-rich phase, search between z(tcrit) and zmax
        try:
            zlo_plo = brentq(lambda z: self.splinet(plo, z) - t, zmin_plo, self.zcrit[plo])
            zhi_plo = brentq(lambda z: self.splinet(plo, z) - t, self.zcrit[plo], zmax_plo)
            zlo_phi = brentq(lambda z: self.splinet(phi, z) - t, zmin_phi, self.zcrit[phi])
            zhi_phi = brentq(lambda z: self.splinet(phi, z) - t, self.zcrit[phi], zmax_phi) # this one fails
        except ValueError as e:
            return 'failed'

        xlo_plo = self.splinex(plo, zlo_plo)
        xhi_plo = self.splinex(plo, zhi_plo)
        xlo_phi = self.splinex(phi, zlo_phi)
        xhi_phi = self.splinex(phi, zhi_phi)

        # P interpolation is linear in logP
        alpha = (np.log10(p) - np.log10(plo)) / (np.log10(phi) - np.log10(plo))
        # alpha = (p - plo) / (phi - plo) # linear in P doesn't work as well
        xlo = alpha * xlo_phi + (1. - alpha) * xlo_plo
        xhi = alpha * xhi_phi + (1. - alpha) * xhi_plo

        return xlo, xhi
