import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

class eos:
    
    def __init__(self, material='ice'):
        
        self.material = material
        available_materials = 'dunite', 'ice', 'iron', 'serpentine', 'water'
        assert self.material in available_materials, 'material must be one of %s, %s, %s, %s, %s' % available_materials
        self.path = '/Users/chris/Dropbox/planet_models/aneos/aneos_%s_pt.dat' % self.material
        self.names = 'logrho', 'logt', 'logp', 'logu', 'logs', 'chit', 'chirho', 'gamma1'
        self.data = np.genfromtxt(self.path, names=self.names)
      
        # this version of aneos.py loads tables already regularized to rectangular in P, T.
        # thus use PT as a basis so we can use RegularGridInterpolator (fast.)
        self.logpvals = np.unique(self.data['logp'])
        self.logtvals = np.unique(self.data['logt'])
        
        assert len(self.logpvals) == len(self.logtvals), 'aneos was implemented assuming square grid in p-t'
        self.npts = len(self.logpvals)
        self.logrho_on_nodes = np.zeros((self.npts, self.npts))
        self.logu_on_nodes = np.zeros((self.npts, self.npts))
        self.logs_on_nodes = np.zeros((self.npts, self.npts))
        self.chit_on_nodes = np.zeros((self.npts, self.npts))
        self.chirho_on_nodes = np.zeros((self.npts, self.npts))
        self.gamma1_on_nodes = np.zeros((self.npts, self.npts))
        
        for i, logpval in enumerate(self.logpvals):
            data_this_logp = self.data[self.data['logp'] == logpval]
            for j, logtval in enumerate(self.logtvals):
                data_this_logp_logt = data_this_logp[data_this_logp['logt'] == logtval]
                self.logrho_on_nodes[i, j] = data_this_logp_logt['logrho']
                self.logu_on_nodes[i, j] = data_this_logp_logt['logu']
                self.logs_on_nodes[i, j] = data_this_logp_logt['logs']
                self.chit_on_nodes[i, j] = data_this_logp_logt['chit']
                self.chirho_on_nodes[i, j] = data_this_logp_logt['chirho']
                self.gamma1_on_nodes[i, j] = data_this_logp_logt['gamma1']
                        
        pt_basis = (self.logpvals, self.logtvals)
        self._get_logrho = RegularGridInterpolator(pt_basis, self.logrho_on_nodes)
        self._get_logu = RegularGridInterpolator(pt_basis, self.logu_on_nodes)
        self._get_logs = RegularGridInterpolator(pt_basis, self.logs_on_nodes)
        self._get_chit = RegularGridInterpolator(pt_basis, self.chit_on_nodes)
        self._get_chirho = RegularGridInterpolator(pt_basis, self.chirho_on_nodes)
        self._get_gamma1 = RegularGridInterpolator(pt_basis, self.gamma1_on_nodes)
        
    # wrapper functions so we can pass logp, logt as args instead of the (logp, logt) tuple
    def get_logrho(self, logp, logt):
        assert not np.any(np.isinf(logt)), 'have inf in logt; cannot look up density.'
        try:
            return self._get_logrho((logp, logt))
        except ValueError:
            print 'out of bounds with logp, logt = ', logp, logt
            raise
    def get_logu(self, logp, logt):
        return self._get_logu((logp, logt))
    def get_logs(self, logp, logt):
        return self._get_logs((logp, logt))
    def get_chit(self, logp, logt):
        return self._get_chit((logp, logt))
    def get_chirho(self, logp, logt):
        return self._get_chirho((logp, logt))
    def get_gamma1(self, logp, logt):
        return self._get_gamma1((logp, logt))
                        
    def regularize_to_ps(self):
        from scipy.optimize import brentq
        import time
        
        print 'regularizing %s tables to rectangular in P, s' % self.material
        
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
            print 'row %i/%i, %f s' % (i+1, self.npts, time.time() - t0)
            
        fmt = '%21.16f\t' * 5
        with open('../aneos/aneos_%s_ps.dat' % self.material, 'w') as fw:
            for i, logpval in enumerate(logpvals):
                for j, logsval in enumerate(logsvals):
                    line = fmt % (logrho_on_ps[i, j], logt_on_ps[i, j], logpval, logu_on_ps[i, j], logsval)
                    fw.write(line + '\n')
                    
        print 'wrote aneos/aneos_%s_ps.dat' % self.material        
        
            
    def plot_rhot_coverage(self, ax=None):
        if ax == None: ax = plt.gca()
        ax.plot(self.data['logrho'], self.data['logt'], 'k,')
        ax.set_xlabel(r'$\log\rho$')
        ax.set_ylabel(r'$\log T$')

    def plot_pt_coverage(self, ax=None):
        if ax == None: ax = plt.gca()
        ax.plot(self.data['logp'], self.data['logt'], 'k,')
        ax.set_xlabel(r'$\log P$')
        ax.set_ylabel(r'$\log T$')