import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from importlib import reload
import aneos_rhot; reload(aneos_rhot)

class eos:

    def __init__(self, path_to_data=None):

        if not path_to_data:
            import os
            path_to_data = os.environ['ongp_data_path']
        self.path = '{}/mazevet_pt.dat'.format(path_to_data)
        self.names = 'p', 't', 'rho', 'u', 'chirho', 'chit' # p, rho, u all cgs
        self.data = np.genfromtxt(self.path, names=self.names)

        # this version loads tables already regularized to rectangular in P, T.
        # thus use PT as a basis so we can use RegularGridInterpolator (fast.)
        self.pvals = np.unique(self.data['p'])
        self.tvals = np.unique(self.data['t'])

        assert len(self.pvals) == len(self.tvals), 'mazevet was implemented assuming square grid in p-t'
        self.npts = len(self.pvals)
        self.logrho_on_nodes = np.zeros((self.npts, self.npts))
        self.logu_on_nodes = np.zeros((self.npts, self.npts))
        # self.logs_on_nodes = np.zeros((self.npts, self.npts))
        self.chit_on_nodes = np.zeros((self.npts, self.npts))
        self.chirho_on_nodes = np.zeros((self.npts, self.npts))
        # self.gamma1_on_nodes = np.zeros((self.npts, self.npts))

        for i, pval in enumerate(self.pvals):
            data_this_p = self.data[self.data['p'] == pval]
            for j, tval in enumerate(self.tvals):
                data_this_p_t = data_this_p[data_this_p['t'] == tval]
                self.logrho_on_nodes[i, j] = np.log10(data_this_p_t['rho'])
                self.logu_on_nodes[i, j] = np.log10(data_this_p_t['u'])
                # self.logs_on_nodes[i, j] = data_this_logp_logt['logs']
                self.chit_on_nodes[i, j] = data_this_p_t['chit']
                self.chirho_on_nodes[i, j] = data_this_p_t['chirho']
                # self.gamma1_on_nodes[i, j] = data_this_logp_logt['gamma1']

        pt_basis = (self.pvals, self.tvals)
        self._get_logrho = RegularGridInterpolator(pt_basis, self.logrho_on_nodes)
        self._get_logu = RegularGridInterpolator(pt_basis, self.logu_on_nodes)
        # self._get_logs = RegularGridInterpolator(pt_basis, self.logs_on_nodes)
        self._get_chit = RegularGridInterpolator(pt_basis, self.chit_on_nodes)
        self._get_chirho = RegularGridInterpolator(pt_basis, self.chirho_on_nodes)
        # self._get_gamma1 = RegularGridInterpolator(pt_basis, self.gamma1_on_nodes)

        # self.rhot_eos = aneos_rhot.eos(material)

    def get_logrho(self, logp, logt):
        return self._get_logrho((logp, logt))

    def get(self, logp, logt):
        res = {}
        res['logrho'] = self._get_logrho((10**logp, 10**logt))
        res['logu'] = self._get_logu((10**logp, 10**logt))
        # res['logs'] = self._get_logs((logp, logt))
        res['chirho'] = self._get_chirho((10**logp, 10**logt)) # (dlnP/dlnrho)_T
        res['chit'] = self._get_chit((10**logp, 10**logt)) # (dlnP/dlnT)_rho
        
        # delta = -(dlnrho/dlnT)_P
        # cp = (du/dT)_P - P/rho^2(drho/dT)_P
        
        return res
