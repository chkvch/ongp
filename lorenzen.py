from scipy import interpolate
from scipy.optimize import brentq, minimize
from scipy.interpolate import splev
import numpy as np
import time
import sys

def get_y(xhe):
    return 1.0 / (1. + (1. - xhe) / (4. * xhe))

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

        self.pvals = np.unique(self.data['p'])

        self.tck = {} # dict used to look up knots, coefficients, order of bspline for any p in self.pvals
        self.tcrit = {} # maximum (critical) temperature of each spline
        self.zcrit = {} # z value ("t" in tck, called z to avoid confusion with temperature) of critical temperature
        # self.splinex = {} # dict used to look up x component of bspline curve for any p in self.pvals
        # self.splinet = {} # dict used to look up t component of bspline curve for any p in self.pvals

        self.initialize_splines()

    def initialize_tck(self, pval):
        data = self.data
        x = data['x'][data['p'] == pval]
        t = data['t'][data['p'] == pval] * 1e-3

        # z is "t" in b-spline language
        z = np.linspace(0, 1, len(x) - 2)
        z = np.append(np.zeros(3), z)
        z = np.append(z, np.ones(3))
        self.tck[pval] = [z, [x, t], 3]

    def splinex(self, pval, z):
        return splev(z, self.tck[pval])[0]

    def splinet(self, pval, z):
        return splev(z, self.tck[pval])[1]

    def initialize_splines(self):

        for pval in self.pvals:
            self.initialize_tck(pval)

            # get z and x corresponding to t maximum (critical point) for this curve.
            # this should be trivial to get from spline coefficients analytically,
            # but get it numerically for convenience
            sol = minimize(lambda z: self.splinet(pval, z) * -1, 0.5, tol=1e-3)
            assert sol['success']
            self.tcrit[pval] = sol['fun'] * -1
            self.zcrit[pval] = sol['x'][0]

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
            raise ValueError('p value {} outside bounds for phase diagram'.format(p))
        elif p > max(self.pvals):
            raise ValueError('p value {} outside bounds for phase diagram'.format(p))

        plo = self.pvals[self.pvals < p][-1]
        phi = self.pvals[self.pvals > p][0]
        assert p > plo
        assert p < phi

        if t > self.tcrit[plo] or t > self.tcrit[phi]:
            return 'stable', 'stable'

        dz = 1e-3
        # for abundance in helium-poor phase, search between dz and z(tcrit)
        # for abundance in helium-rich phase, search between z(tcrit) and 1-dz
        try:
            zlo_plo = brentq(lambda z: self.splinet(plo, z) - t, dz, self.zcrit[plo])
            zhi_plo = brentq(lambda z: self.splinet(plo, z) - t, self.zcrit[plo], 1 - dz)
            zlo_phi = brentq(lambda z: self.splinet(phi, z) - t, dz, self.zcrit[phi])
            zhi_phi = brentq(lambda z: self.splinet(phi, z) - t, self.zcrit[phi], 1 - dz)
        except ValueError:
            return 'failed', 'failed'

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
