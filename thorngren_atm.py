import const
import numpy as np

class atm:
    def __init__(self):
        pass

    def get_tint(self, teq):
        f = 4. * const.sigma_sb * teq ** 4
        # Thorngren, Gau & Fortney (2019) equation 3 and Thorngren & Fortney (2018) equation 34
        # expect F in units of 10^9 erg s^-1 cm^-2, and use the natural log:
        # return 1.24 * teq * np.exp(-(np.log(f) - np.log(1e9) - 0.14) ** 2 / 2.96)
        return 0.39 * teq * np.exp(-(np.log10(f) - 9 - 0.14) ** 2 / 1.0952)
