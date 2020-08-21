import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from importlib import reload
import aneos_rhot; reload(aneos_rhot)

class eos:
    ''' does a rock/ice mix from aneos ice and serpentine tables '''
    def __init__(self, path_to_data=None, f_ice=0.5, extended=False):
        self.f_ice = f_ice
        if not path_to_data:
            import os
            path_to_data = os.environ['ongp_data_path']
        if extended:
            path_ice = f'{path_to_data}/aneos_ice_pt_hi-p.dat'
            path_ser = f'{path_to_data}/aneos_serpentine_pt_hi-p.dat'
        else:
            path_ice = f'{path_to_data}/aneos_ice_pt.dat'
            path_ser = f'{path_to_data}/aneos_serpentine_pt.dat'
        self.names = 'logrho', 'logt', 'logp', 'logu', 'logs' # , 'chit', 'chirho', 'gamma1'
        self.data_ice = np.genfromtxt(path_ice, names=self.names, usecols=(0, 1, 2, 3, 4)) # will fail if haven't saved version of aneos_*_pt.dat with eight columns
        self.data_ser = np.genfromtxt(path_ser, names=self.names, usecols=(0, 1, 2, 3, 4)) # will fail if haven't saved version of aneos_*_pt.dat with eight columns

        # this version of aneos.py loads tables already regularized to rectangular in P, T.
        # thus use PT as a basis so we can use RegularGridInterpolator (fast.)
        self.logpvals = np.unique(self.data_ice['logp'])
        self.logtvals = np.unique(self.data_ice['logt'])
        assert np.all(np.unique(self.data_ser['logp']) == self.logpvals), 'inconsistent ice and serpentine tables?'
        assert np.all(np.unique(self.data_ser['logt']) == self.logtvals), 'inconsistent ice and serpentine tables?'

        assert len(self.logpvals) == len(self.logtvals), 'aneos was implemented assuming square grid in p-t'
        self.npts = len(self.logpvals)
        self.logrho_on_nodes_ice = np.zeros((self.npts, self.npts))
        self.logu_on_nodes_ice = np.zeros((self.npts, self.npts))
        self.logs_on_nodes_ice = np.zeros((self.npts, self.npts))
        self.logrho_on_nodes_ser = np.zeros((self.npts, self.npts))
        self.logu_on_nodes_ser = np.zeros((self.npts, self.npts))
        self.logs_on_nodes_ser = np.zeros((self.npts, self.npts))
        # self.chit_on_nodes = np.zeros((self.npts, self.npts))
        # self.chirho_on_nodes = np.zeros((self.npts, self.npts))
        # self.gamma1_on_nodes = np.zeros((self.npts, self.npts))

        for i, logpval in enumerate(self.logpvals):
            ice_data_this_logp = self.data_ice[self.data_ice['logp'] == logpval]
            ser_data_this_logp = self.data_ser[self.data_ser['logp'] == logpval]
            for j, logtval in enumerate(self.logtvals):
                ice_data_this_logp_logt = ice_data_this_logp[ice_data_this_logp['logt'] == logtval]
                ser_data_this_logp_logt = ser_data_this_logp[ser_data_this_logp['logt'] == logtval]
                self.logrho_on_nodes_ice[i, j] = ice_data_this_logp_logt['logrho']
                self.logu_on_nodes_ice[i, j] = ice_data_this_logp_logt['logu']
                self.logs_on_nodes_ice[i, j] = ice_data_this_logp_logt['logs']
                self.logrho_on_nodes_ser[i, j] = ser_data_this_logp_logt['logrho']
                self.logu_on_nodes_ser[i, j] = ser_data_this_logp_logt['logu']
                self.logs_on_nodes_ser[i, j] = ser_data_this_logp_logt['logs']
                # self.chit_on_nodes[i, j] = data_this_logp_logt['chit']
                # self.chirho_on_nodes[i, j] = data_this_logp_logt['chirho']
                # self.gamma1_on_nodes[i, j] = data_this_logp_logt['gamma1']

        pt_basis = (self.logpvals, self.logtvals)
        self._get_logrho_ice = RegularGridInterpolator(pt_basis, self.logrho_on_nodes_ice)
        self._get_logu_ice = RegularGridInterpolator(pt_basis, self.logu_on_nodes_ice)
        self._get_logs_ice = RegularGridInterpolator(pt_basis, self.logs_on_nodes_ice)
        self._get_logrho_ser = RegularGridInterpolator(pt_basis, self.logrho_on_nodes_ser)
        self._get_logu_ser = RegularGridInterpolator(pt_basis, self.logu_on_nodes_ser)
        self._get_logs_ser = RegularGridInterpolator(pt_basis, self.logs_on_nodes_ser)
        # self._get_chit = RegularGridInterpolator(pt_basis, self.chit_on_nodes)
        # self._get_chirho = RegularGridInterpolator(pt_basis, self.chirho_on_nodes)
        # self._get_gamma1 = RegularGridInterpolator(pt_basis, self.gamma1_on_nodes)

        self.rhot_eos_ice = aneos_rhot.eos('ice', path_to_data)
        self.rhot_eos_ser = aneos_rhot.eos('serpentine', path_to_data)

    def get_logrho(self, logp, logt):
        X = self.f_ice
        return -np.log10(X / 10 ** self._get_logrho_ice((logp, logt)) + (1. - X) / 10 ** self._get_logrho_ser((logp, logt)))

    def get(self, logp, logt):
        ''' of the heavy elements, X is the mass fraction of ice, 1-X the mass fraction of rock '''
        X = self.f_ice
        res = {}
        logrho_ice = self._get_logrho_ice((logp, logt))
        logu_ice = self._get_logu_ice((logp, logt))
        logs_ice = self._get_logs_ice((logp, logt))
        logrho_ser = self._get_logrho_ser((logp, logt))
        logu_ser = self._get_logu_ser((logp, logt))
        logs_ser = self._get_logs_ser((logp, logt))
        
        rhoinv = X / 10 ** logrho_ice + (1. - X) / 10 ** logrho_ser
        logrho = - np.log10(rhoinv)
        u = X * 10 ** logu_ice + (1. - X) * 10 ** logu_ser
        s = X * 10 ** logs_ice + (1. - X) * 10 ** logs_ser
        
        res['logrho'] = logrho
        res['logu'] = np.log10(u)
        res['logs'] = np.log10(s)
        
        # some derivs are easier to evaluate from the original rho, t basis
        rhot_res_ice = self.rhot_eos_ice.get(logrho_ice, logt)
        rhot_res_ser = self.rhot_eos_ser.get(logrho_ser, logt)
        delta = X * 10 ** (logrho - logrho_ice) * rhot_res_ice['rhot']
        delta += (1. - X) * 10 ** (logrho - logrho_ser) * rhot_res_ser['rhot']
        res['rhot'] = -delta
        
        res['rhop'] = -X * 10 ** (logrho - logrho_ice) * rhot_res_ice['rhop']
        res['rhop'] -= (1. - X) * 10 ** (logrho - logrho_ser) * rhot_res_ser['rhop']
        
        res['chirho'] = res['rhop'] ** -1
        res['chit'] = -res['rhot'] / res['rhop']
        res['grada'] = np.nan * np.ones_like(logrho)
        res['gamma1'] = np.nan * np.ones_like(logrho)

        # res['chirho'] = rhot_res['chirho']
        # res['chit'] = rhot_res['chit']
        # res['rhop'] = rhot_res['rhop']
        # res['rhot'] = rhot_res['rhot'] # "-delta"
        # res['grada'] = rhot_res['grada']
        # res['gamma1'] = rhot_res['gamma1']


        # chirho = 1. / rhop # dlnP/dlnrho|T
        # chit = -1. * rhot / rhop # dlnP/dlnT|rho
        # gamma3 = 1. + dpdu_const_rho # cox and giuli 9.93a
        # gamma1 = (gamma3 - 1.) / res['grada']

        return res

    # wrapper functions so we can pass logp, logt as args instead of the (logp, logt) tuple
    # def get_logrho(self, logp, logt):
    #     assert not np.any(np.isinf(logt)), 'have inf in logt; cannot look up density.'
    #     try:
    #         return self._get_logrho((logp, logt))
    #     except ValueError:
    #         print('out of bounds in aneos get_logrho.')
    #         raise
    # def get_logu(self, logp, logt):
    #     return self._get_logu((logp, logt))
    # def get_logs(self, logp, logt):
    #     return self._get_logs((logp, logt))
    # def get_chit(self, logp, logt):
    #     return self._get_chit((logp, logt))
    # def get_chirho(self, logp, logt):
    #     return self._get_chirho((logp, logt))
    # def get_gamma1(self, logp, logt):
    #     return self._get_gamma1((logp, logt))

    # def get_dlogrho_dlogp_const_t(self, logp, logt, f=0.8):
    #     logp_lo = logp - np.log10(1. - f)
    #     logp_hi = logp + np.log10(1. + f)
    #     logrho_lo = self.get_logrho(logp_lo, logt)
    #     logrho_hi = self.get_logrho(logp_hi, logt)
    #     return (logrho_hi - logrho_lo) / (logp_hi - logp_lo)
    #
    # def get_dlogrho_dlogt_const_p(self, logp, logt, f=0.8):
    #     logt_lo = logt - np.log10(1. - f)
    #     logt_hi = logt + np.log10(1. + f)
    #     logrho_lo = self.get_logrho(logp, logt_lo)
    #     logrho_hi = self.get_logrho(logp, logt_hi)
    #     return (logrho_hi - logrho_lo) / (logt_hi - logt_lo)

    # def get_dlogrho_dlogt_const_p(self, logp, logt):
    #     return self.get_chit(logp, logt) / self.get_chirho(logp, logt)
    #
    # def get_dlogrho_dlogp_const_t(self, logp, logt):
    #     return 1. / self.get_chirho(logp, logt)

    def regularize_to_ps(self):
        from scipy.optimize import brentq
        import time

        print('regularizing %s tables to rectangular in P, s' % self.material)

        logpvals = np.linspace(6, 15, self.npts)
        logsvals = np.linspace(min(self.data['logs']), max(self.data['logs']), self.npts)

        logt_on_ps = np.zeros((self.npts, self.npts))
        logrho_on_ps = np.zeros((self.npts, self.npts))
        logu_on_ps = np.zeros((self.npts, self.npts))

        t0 = time.time()
        for i, logpval in enumerate(logpvals):
            for j, logsval in enumerate(logsvals):
                try:
                    zero_me = lambda logt: self._get_logs((logpval, logt)) - logsval
                    logt_on_ps[i, j] = brentq(zero_me, min(self.data['logt']), max(self.data['logt']))
                    logrho_on_ps[i, j] = self._get_logrho((logpval, logt_on_ps[i, j]))
                    logu_on_ps[i, j] = self._get_logu((logpval, logt_on_ps[i, j]))
                except ValueError:
                    logt_on_ps[i, j] = np.nan
                    logrho_on_ps[i, j] = np.nan
                    logu_on_ps[i, j] = np.nan
            print('row %i/%i, %f s' % (i+1, self.npts, time.time() - t0))

        fmt = '%21.16f\t' * 5
        with open('../aneos/aneos_%s_ps.dat' % self.material, 'w') as fw:
            for i, logpval in enumerate(logpvals):
                for j, logsval in enumerate(logsvals):
                    line = fmt % (logrho_on_ps[i, j], logt_on_ps[i, j], logpval, logu_on_ps[i, j], logsval)
                    fw.write(line + '\n')

        print('wrote aneos/aneos_%s_ps.dat' % self.material)


    def plot_rhot_coverage(self, ax=None):
        import matplotlib.pyplot as plt
        if ax == None: ax = plt.gca()
        ax.plot(self.data['logrho'], self.data['logt'], 'k,')
        ax.set_xlabel(r'$\log\rho$')
        ax.set_ylabel(r'$\log T$')

    def plot_pt_coverage(self, ax=None):
        import matplotlib.pyplot as plt
        if ax == None: ax = plt.gca()
        ax.plot(self.data['logp'], self.data['logt'], 'k,')
        ax.set_xlabel(r'$\log P$')
        ax.set_ylabel(r'$\log T$')
