import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

class eos:

    def __init__(self, path_to_data):
        # original table is rectangular in logrho, logt
        path = '%s/eosH2OREOS_13a_wS.dat' % path_to_data

        self.names = 'rho', 't', 'p', 'u', 's'
        self.data = np.genfromtxt(path, skip_header=1, names=self.names)

        import reos_rhos
        self.rhos_eos = reos_rhos.eos() # for computing certain derivatives at constant entropy

        self.data['p'] *= 1e10 # GPa to dyne cm^-2
        self.data['u'] *= 1e10 # kJ g^-1 to erg g^-1
        self.data['s'] *= 1e10 # kJ g^-1 K^-1 to erg g^-1 K^-1

        basis = np.array([np.log10(self.data['p']), np.log10(self.data['t'])]).T
        self.get_logs = LinearNDInterpolator(basis, np.log10(self.data['s']))
        self.get_logu = LinearNDInterpolator(basis, np.log10(self.data['u']))
        self.get_logrho = LinearNDInterpolator(basis, np.log10(self.data['rho']))

    def get_dlogrho_dlogp_const_t(self, logp, logt, f=1e-1):
        logp_lo = logp - f
        logp_hi = logp + f
        logrho_lo = self.get_logrho(logp_lo, logt)
        logrho_hi = self.get_logrho(logp_hi, logt)
        return (logrho_hi - logrho_lo) / (logp_hi - logp_lo)

    def get_dlogrho_dlogt_const_p(self, logp, logt, f=1e-1):
        logt_lo = logt - f
        logt_hi = logt + f
        logrho_lo = self.get_logrho(logp, logt_lo)
        logrho_hi = self.get_logrho(logp, logt_hi)
        return (logrho_hi - logrho_lo) / (logt_hi - logt_lo)
