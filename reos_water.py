import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

class eos:

    def __init__(self, path_to_data):

        path = '%s/reos_water_pt.dat' % path_to_data

        # Nadine 22 Sep 2015: Fifth column is entropy in kJ/g/K+offset

        self.names = 'logrho', 'logt', 'logp', 'logu', 'logs' #, 'chit', 'chirho', 'gamma1'
        self.data = np.genfromtxt(path, names=self.names, usecols=(0, 1, 2, 3, 4))

        self.logpvals = np.unique(self.data['logp'])
        self.logtvals = np.unique(self.data['logt'])

        self.logpmin = min(self.logpvals)
        self.logpmax = max(self.logpvals)
        self.logtmin = min(self.logtvals)
        self.logtmax = max(self.logtvals)

        self.nptsp = len(self.logpvals)
        self.nptst = len(self.logtvals)

        self.logrho_on_pt = np.zeros((self.nptsp, self.nptst))
        self.logu_on_pt = np.zeros((self.nptsp, self.nptst))
        self.logs_on_pt = np.zeros((self.nptsp, self.nptst))
        # self.chit_on_pt = np.zeros((self.nptsp, self.nptst))
        # self.chirho_on_pt = np.zeros((self.nptsp, self.nptst))
        # self.gamma1_on_pt = np.zeros((self.nptsp, self.nptst))

        for i, logpval in enumerate(self.logpvals):
            data_this_logp = self.data[self.data['logp'] == logpval]
            for j, logtval in enumerate(self.logtvals):
                data_this_logp_logt = data_this_logp[data_this_logp['logt'] == logtval]
                self.logrho_on_pt[i, j] = data_this_logp_logt['logrho']
                self.logu_on_pt[i, j] = data_this_logp_logt['logu']
                self.logs_on_pt[i, j] = data_this_logp_logt['logs']
                # self.chit_on_pt[i, j] = data_this_logp_logt['chit']
                # self.chirho_on_pt[i, j] = data_this_logp_logt['chirho']
                # self.gamma1_on_pt[i, j] = data_this_logp_logt['gamma1']

        pt_basis = (self.logpvals, self.logtvals)
        self._get_logrho = RegularGridInterpolator(pt_basis, self.logrho_on_pt, bounds_error=False)
        self._get_logu = RegularGridInterpolator(pt_basis, self.logu_on_pt)
        self._get_logs = RegularGridInterpolator(pt_basis, self.logs_on_pt)
        # self._get_chit = RegularGridInterpolator(pt_basis, self.chit_on_pt)
        # self._get_chirho = RegularGridInterpolator(pt_basis, self.chirho_on_pt)
        # self._get_gamma1 = RegularGridInterpolator(pt_basis, self.gamma1_on_pt)

    def get_logrho(self, logp, logt):
        return self._get_logrho((logp, logt))

    def get_logu(self, logp, logt):
        return self._get_logu((logp, logt))

    def get_logs(self, logp, logt):
        return self._get_logs((logp, logt)) # + 10. # kJ/g/K to erg/g/K
