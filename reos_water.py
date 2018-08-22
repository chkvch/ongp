import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import gp_configs.app_config as app_cfg
import gp_configs.model_config as model_cfg
import logging
import config_const as conf

log = logging.getLogger(__name__)
logging.basicConfig(filename=app_cfg.logfile, filemode='w', format=conf.FORMAT)
log.setLevel(conf.log_level)

class eos:

    def __init__(self, path_to_data):

        path = '%s/reos_water_pt.dat' % path_to_data

        # Nadine 22 Sep 2015: Fifth column is entropy in kJ/g/K+offset

        log.debug('message')
        self.names = 'logrho', 'logt', 'logp', 'logu', 'logs', 'chit', 'chirho', 'gamma1'
        self.data = np.genfromtxt(path, names=self.names)

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
        self.chit_on_pt = np.zeros((self.nptsp, self.nptst))
        self.chirho_on_pt = np.zeros((self.nptsp, self.nptst))
        self.gamma1_on_pt = np.zeros((self.nptsp, self.nptst))

        for i, logpval in enumerate(self.logpvals):
            data_this_logp = self.data[self.data['logp'] == logpval]
            for j, logtval in enumerate(self.logtvals):
                data_this_logp_logt = data_this_logp[data_this_logp['logt'] == logtval]
                self.logrho_on_pt[i, j] = data_this_logp_logt['logrho']
                self.logu_on_pt[i, j] = data_this_logp_logt['logu']
                self.logs_on_pt[i, j] = data_this_logp_logt['logs']
                self.chit_on_pt[i, j] = data_this_logp_logt['chit']
                self.chirho_on_pt[i, j] = data_this_logp_logt['chirho']
                self.gamma1_on_pt[i, j] = data_this_logp_logt['gamma1']

        pt_basis = (self.logpvals, self.logtvals)
        self._get_logrho = RegularGridInterpolator(pt_basis, self.logrho_on_pt, bounds_error=False)
        self._get_logu = RegularGridInterpolator(pt_basis, self.logu_on_pt)
        self._get_logs = RegularGridInterpolator(pt_basis, self.logs_on_pt)
        self._get_chit = RegularGridInterpolator(pt_basis, self.chit_on_pt)
        self._get_chirho = RegularGridInterpolator(pt_basis, self.chirho_on_pt)
        self._get_gamma1 = RegularGridInterpolator(pt_basis, self.gamma1_on_pt)

    def get_logrho(self, logp, logt):
        return self._get_logrho((logp, logt))

    def get_logu(self, logp, logt):
        return self._get_logu((logp, logt))

    def get_logs(self, logp, logt):
        return self._get_logs((logp, logt)) # + 10. # kJ/g/K to erg/g/K

    def get_chit(self, logp, logt):
        return self._get_chit((logp, logt))

    def get_chirho(self, logp, logt):
        return self._get_chirho((logp, logt))

    def get_gamma1(self, logp, logt):
        return self._get_gamma1((logp, logt))

    def get_dlogrho_dlogp_const_t(self, logp, logt, f=1e-1):
        logp_lo = logp - np.log10(1. - f)
        logp_hi = logp + np.log10(1. + f)
        logrho_lo = self.get_logrho(logp_lo, logt)
        logrho_hi = self.get_logrho(logp_hi, logt)
        return (logrho_hi - logrho_lo) / (logp_hi - logp_lo)

    def get_dlogrho_dlogt_const_p(self, logp, logt, f=1e-1):
        logt_lo = logt + np.log10(1. - f)
        logt_hi = logt + np.log10(1. + f)
        logrho_lo = self.get_logrho(logp, logt_lo)
        logrho_hi = self.get_logrho(logp, logt_hi)
        return (logrho_hi - logrho_lo) / (logt_hi - logt_lo)

    def get_dlogs_dlogp_const_t(self, logp, logt, f=1e-1):
        # logp_lo = logp - np.log10(1. - f)
        # logp_hi = logp + np.log10(1. + f)
        logp_lo = logp - f
        logp_hi = logp + f
        logs_lo = self.get_logs(logp_lo, logt)
        logs_hi = self.get_logs(logp_hi, logt)
        return (logs_hi - logs_lo) / (logp_hi - logp_lo)

    def get_dlogs_dlogt_const_p(self, logp, logt, f=1e-1):
        # logt_lo = logt - np.log10(1. - f)
        # logt_hi = logt + np.log10(1. + f)
        logt_lo = logt - f
        logt_hi = logt + f
        logs_lo = self.get_logs(logp, logt_lo)
        logs_hi = self.get_logs(logp, logt_hi)
        return (logs_hi - logs_lo) / (logt_hi - logt_lo)

    def get_cp(self, logp, logt):
        rhop = self.get_dlogrho_dlogp_const_t(logp, logt)
        rhot = self.get_dlogrho_dlogt_const_p(logp, logt)
        rho = 10 ** self.get_logrho(logp, logt)
        sp = self.get_dlogs_dlogp_const_t(logp, logt)
        st = self.get_dlogs_dlogt_const_p(logp, logt)
        s = 10 ** self.get_logs(logp, logt)
        dpdt_const_rho = - 10 ** logp / 10 ** logt * rhot / rhop
        dudt_const_rho = s * (st - sp * rhot / rhop)
        dpdu_const_rho = dpdt_const_rho / rho / dudt_const_rho
        gamma3 = 1. + dpdu_const_rho # cox and giuli 9.93a
        gamma1 = self.get_gamma1(logp, logt) # (gamma3 - 1.) / res['grada']
        chirho = rhop ** -1
        chit = dpdt_const_rho * 10 ** logt / 10 ** logp
        cv = chit * 10 ** logp / (rho * 10 ** logt * (gamma3 - 1.))
        cp = cv + 10 ** logp * chit ** 2 / (rho * 10 ** logt * chirho)
        return cp

    def get_grada(self, logp, logt):
        rhop = self.get_dlogrho_dlogp_const_t(logp, logt)
        rhot = self.get_dlogrho_dlogt_const_p(logp, logt)
        rho = 10 ** self.get_logrho(logp, logt)
        sp = self.get_dlogs_dlogp_const_t(logp, logt)
        st = self.get_dlogs_dlogt_const_p(logp, logt)
        s = 10 ** self.get_logs(logp, logt)
        dpdt_const_rho = - 10 ** logp / 10 ** logt * rhot / rhop
        dudt_const_rho = s * (st - sp * rhot / rhop)
        dpdu_const_rho = dpdt_const_rho / rho / dudt_const_rho
        gamma3 = 1. + dpdu_const_rho # cox and giuli 9.93a
        gamma1 = self.get_gamma1(logp, logt) # (gamma3 - 1.) / res['grada']
        # gamma3 = 1 + dpdt / (din * dedt) ! C&G 9.93
        # grad_ad = dtdp_cs_hhe/(tin * inv_pres)
        # gamma1 = (gamma3 - 1) / grad_ad ! C&G 9.88 & 9.89
        grada = dpdu_const_rho / gamma1
        return grada
