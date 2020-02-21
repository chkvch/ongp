import numpy as np
from scipy.interpolate import RegularGridInterpolator
import const
# from scipy.optimize import brentq

class atm:

    '''
    interpolate in Jonathan Fortney's model atmospheres
    solar composition
    four levels of flux corresponding to 0.02, 0.05, 0.1, and 1 AU around a 1 Lsun star
    five surface gravities: logg = 0, 1.6, 2.6, 3.6, 4.6
    for a given flux and surface gravity, provides T_int as a function of T_990
    '''

    def __init__(self, path_to_data=None):

        if not path_to_data:
            import os
            path_to_data = os.environ['ongp_data_path']

        self.data = np.load('atmosphereGrid.npz')
        
        self._get_logTint = RegularGridInterpolator(
            (
            self.data['flux'], # erg cm^-2 s^-1
            self.data['logGravity'], # cm s^-2
            self.data['logT990'] # K
            ),
            self.data['logTint'], bounds_error=False, fill_value=None) 
        # might use fill_value=extrapolate instead if need to evaluate outside of 1 AU
        
        # the call signature for this looks like:
        # intrinsicTemp = 10.**self._get_tint((flux, logGravity, log10(t990)))

        
                # self.getIntrinsicTemp = RegularGridInterpolator(
                #     (atmos['flux'],
                #      atmos['logGravity'],
                #      atmos['logT990']),
                #     atmos['logTint'],bounds_error=False,fill_value=None)
        
    def get_tint(self, g, teq, t990):
        # will be called from ongp like this:
        # self.tint = self.atm.get_tint(self.surface_g, self.teq, self.t990)
        flux = const.sigma_sb * teq ** 4
        return 10. ** self._get_logTint((flux, np.log10(g), np.log10(t990)))
        