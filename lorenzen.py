from scipy import interpolate
from scipy.optimize import brentq
import numpy as np
import utils
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
    interpolate in logT instead."""
    
    def __init__(self, order=3, smooth=0., t_offset=0., extrapolate_to_low_pressure=True):
        
        
        self.path_to_lhs_data = 'data/lorenzen_lhs.dat'
        self.path_to_rhs_data = 'data/lorenzen_rhs.dat'
        self.columns = 'x', 'p', 't' # x refers to the helium number fraction
        self.lhs_data = np.genfromtxt(self.path_to_lhs_data, names=self.columns)
        self.rhs_data = np.genfromtxt(self.path_to_rhs_data, names=self.columns)

        # April 12 2017: implement t_offset inside Evolver.staticModel instead, so we don't
        # need to create a new Evolver instance every time we want to change dtphase
        #
        # self.t_offset = t_offset
        # self.lhs_data['t'] += self.t_offset
        # self.rhs_data['t'] += self.t_offset
                
        self.p_grid = np.unique(self.lhs_data['p'])

        self.pmin = self.p_grid[0] # if doing extrapolate_to_low_p, could also decrease this to, e.g., 0.8 Mbar
        self.pmax = self.p_grid[-1]   
                            
        '''option to remove the 1 Mbar curve and extrapolate down in P for those pressures instead.
        this is motivated by the fact that the x-T curve for 1 Mbar is concave up toward low helium
        concentrations, which tends to give non-monotone helium profiles for cool giant planets. 
        this might not be real.'''
        self.extrapolate_to_low_pressure = extrapolate_to_low_pressure
        if self.extrapolate_to_low_pressure:
            self.p_grid = np.delete(self.p_grid, 0)
            
        self.xt_splines_lhs = {}
        for i, pval in enumerate(self.p_grid):
            x = self.lhs_data['x'][self.lhs_data['p'] == pval]
            t = self.lhs_data['t'][self.lhs_data['p'] == pval]
            self.xt_splines_lhs[pval] = interpolate.splrep(t, x, s=smooth, k=order)
        
    def xmax_lhs(self, pval, tval):
        assert self.pmin <= pval < self.pmax, 'p value %f Mbar out of bounds' % pval
        
        if pval < 2. and self.extrapolate_to_low_pressure:
            p_lo, p_hi = self.p_grid[0], self.p_grid[1]
            # alpha_p will be negative.
        else: # use the full tables
            p_bin = np.where(self.p_grid - pval > 0)[0][0]
            p_lo, p_hi = self.p_grid[p_bin - 1], self.p_grid[p_bin]
        
        alpha_p = (np.log10(pval) - np.log10(p_lo)) / (np.log10(p_hi) - np.log10(p_lo)) # linear in logp
        # alpha_p = (pval - p_lo) / (p_hi - p_lo) # could also do linear in p if preferred
    
        # from the scipy.interpolate.splev documentation:
        #
        # der : int, optional; The order of derivative of the spline to compute (must be less than or equal to k).
        # ext : int, optional; Controls the value returned for elements of x not in the interval defined by the knot sequence.
        # if ext=0, return the extrapolated value.
        # if ext=1, return 0
        # if ext=2, raise a ValueError
        # if ext=3, return the boundary value.

        ext=0
        res_lo_p = interpolate.splev(tval, self.xt_splines_lhs[p_lo], der=0, ext=ext)
        res_hi_p = interpolate.splev(tval, self.xt_splines_lhs[p_hi], der=0, ext=ext)
        
        res = alpha_p * res_hi_p + (1. - alpha_p) * res_lo_p
        
        return res
        
    def ymax_lhs(self, pval, tval):
        x = self.xmax_lhs(pval, tval)
        return get_y(x)

    def tphase_lhs(self, xval, pval):
        '''old-style interpolation for tphase from x, P. only using for testing at present.'''
        from scipy.interpolate import griddata
        return griddata(zip(self.lhs_data['x'], np.log10(self.lhs_data['p'])), self.lhs_data['t'], (xval, np.log10(pval)), method='linear')
        
    def show_phase_curves(self, **kwargs):
        import matplotlib.pyplot as plt
        plt.xlabel(r'$Y$')
        plt.ylabel(r'$T$')
        for pval in self.p_grid:
            x = self.lhs_data['x'][self.lhs_data['p'] == pval]
            t = self.lhs_data['t'][self.lhs_data['p'] == pval]
            plt.plot(get_y(x), t, '-', label='%2f Mbar' % pval, lw=1, **kwargs)
            
        plt.xlim(0, 0.3)
        plt.ylim(2e3, 8e3)
        