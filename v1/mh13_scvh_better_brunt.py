from scipy.interpolate import RectBivariateSpline as rbs
try:
    from importlib import reload
except:
    pass
import scvh; reload(scvh)
import numpy as np

class eos:
    def __init__(self, path_to_data=None):
        if not path_to_data:
            import os
            path_to_data = os.environ['ongp_data_path']
        self.columns = 'logp', 'logt', 'logrho', 'logs'
        self.h_path = '{}/MH13+SCvH-H-2018.dat'.format(path_to_data)
        self.h_data = np.genfromtxt(self.h_path, skip_header=16, names=self.columns)

        self.logpvals = np.unique(self.h_data['logp'][self.h_data['logp'] <= 16.])
        self.logpvals = self.logpvals[self.logpvals > 5.8]
        self.logtvals = np.unique(self.h_data['logt'][self.h_data['logp'] <= 16.])
        # self.logtvals = self.logtvals[self.logtvals < 5]

        self.logrho = np.zeros((len(self.logpvals), len(self.logtvals)))
        self.logs = np.zeros((len(self.logpvals), len(self.logtvals)))
        for i, logpval in enumerate(self.logpvals):
            logrho_this_p = self.h_data['logrho'][self.h_data['logp'] == logpval]
            logs_this_p = self.h_data['logs'][self.h_data['logp'] == logpval]
            logt_this_p = self.h_data['logt'][self.h_data['logp'] == logpval]
            for j, logtval in enumerate(self.logtvals):
                self.logrho[i, j] = logrho_this_p[logt_this_p == logtval]
                self.logs[i, j] = logs_this_p[logt_this_p == logtval]

        del(self.h_data)

        self.logtlo_h = 2.25

        self.spline_kwargs = {'kx':3, 'ky':3}
        self.he_eos = scvh.eos(path_to_data)

    # methods for getting pure hydrogen quantities by interpolating in mh13
    def get_logrho_h(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logrho, **self.spline_kwargs)(lgp, lgt, grid=False)
    def get_logs_h(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logs, **self.spline_kwargs)(lgp, lgt, grid=False)
    def get_sp_h(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logs, **self.spline_kwargs)(lgp, lgt, dx=1, grid=False)
    def get_st_h(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logs, **self.spline_kwargs)(lgp, lgt, dy=1, grid=False)

    # if you want a more reasonable "direct" Brunt, get rhop and rhot this way: # models for Janosz this way
    def get_rhop_h(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logrho, **self.spline_kwargs)(lgp, lgt, dx=1, grid=False)
    def get_rhot_h(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.logrho, **self.spline_kwargs)(lgp, lgt, dy=1, grid=False)
    # rho_t and rho_p from MH13 tables are presenting some difficulties, e.g., rhot_h changes sign in the neighborhood of
    # 1 Mbar in a Jupiter adiabat. instead get rhot_h and rhop_h from the scvh tables below. only really enters
    # the calculation of brunt_B.

    # methods for getting pure helium quantities by interpolating in scvh
    def get_logrho_he(self, lgp, lgt):
        return self.he_eos.get_he['logrho']((lgp, lgt))
    def get_logs_he(self, lgp, lgt):
        return self.he_eos.get_he['logs']((lgp, lgt))
    def get_sp_he(self, lgp, lgt):
        return self.he_eos.get_he['sp']((lgp, lgt))
    def get_st_he(self, lgp, lgt):
        return self.he_eos.get_he['st']((lgp, lgt))
    def get_rhop_he(self, lgp, lgt):
        return self.he_eos.get_he['rhop']((lgp, lgt))
    def get_rhot_he(self, lgp, lgt):
        return self.he_eos.get_he['rhot']((lgp, lgt))
    # if you want a more reasonable Gamma_1 (but direct form of Brunt might look bad), get rhop and rhot this way: # this is our default
    # def get_rhop_h(self, lgp, lgt):
        # return self.he_eos.get_h['rhop']((lgp, lgt))
    # def get_rhot_h(self, lgp, lgt):
        # return self.he_eos.get_h['rhot']((lgp, lgt))

    # general method for getting quantities for hydrogen-helium mixture
    def get(self, logp, logt, y):
        if type(logp) is float: logp = np.array([logp])
        if type(logt) is float: logt = np.array([logt])
        if len(logp) != len(logt):
            if len(logp) == 1:
                logp = np.ones_like(logt) * logp[0]
            elif len(logt) == 1:
                logt = np.ones_like(logp) * logt[0]
            else:
                raise ValueError('got unequal lengths {} and {} for logp and logt and neither is equal to 1.'.format(len(logp), len(logt)))

        s_h = 10 ** self.get_logs_h(logp, logt)
        sp_h = self.get_sp_h(logp, logt)
        st_h = self.get_st_h(logp, logt)

        # fix up values for low t by asking scvh h directly; mh13+scvh h table only covers logT >= 2.25, about 178 K.
        tlo = self.logtlo_h
        s_h[logt < tlo] = 10 ** self.he_eos.get_h['logs']((logp[logt < tlo], logt[logt < tlo]))
        sp_h[logt < tlo] = self.he_eos.get_h['sp']((logp[logt < tlo], logt[logt < tlo]))
        st_h[logt < tlo] = self.he_eos.get_h['st']((logp[logt < tlo], logt[logt < tlo]))

        s_he = 10 ** self.get_logs_he(logp, logt)
        sp_he = self.get_sp_he(logp, logt)
        st_he = self.get_st_he(logp, logt)

        # smix = 10 ** self.he_eos.get_logsmix(logp, logt, y)
        s = (1. - y) * s_h + y * s_he # + smix
        st = (1. - y) * s_h / s * st_h + y * s_he / s * st_he # + smix/s*dlogsmix/dlogt
        sp = (1. - y) * s_h / s * sp_h + y * s_he / s * sp_he # + smix/s*dlogsmix/dlogp
        grada = - sp / st

        rho_h = 10 ** self.get_logrho_h(logp, logt)
        rho_h[logt < tlo] = 10 ** self.he_eos.get_h['logrho']((logp[logt < tlo], logt[logt < tlo]))
        rho_he = 10 ** self.get_logrho_he(logp, logt)
        rhoinv = y / rho_he + (1. - y) / rho_h
        rho = rhoinv ** -1.
        logrho = np.log10(rho)
        rhop_h = self.get_rhop_h(logp, logt)
        rhot_h = self.get_rhot_h(logp, logt)
        rhop_h[logt < tlo] = self.he_eos.get_h['rhop']((logp[logt < tlo], logt[logt < tlo]))
        rhot_h[logt < tlo] = self.he_eos.get_h['rhot']((logp[logt < tlo], logt[logt < tlo]))
        rhop_he = self.get_rhop_he(logp, logt)
        rhot_he = self.get_rhot_he(logp, logt)

        rhot = (1. - y) * rho / rho_h * rhot_h + y * rho / rho_he * rhot_he
        rhop = (1. - y) * rho / rho_h * rhop_h + y * rho / rho_he * rhop_he

        chirho = 1. / rhop # dlnP/dlnrho|T
        chit = -1. * rhot / rhop # dlnP/dlnT|rho
        # gamma1 = 1. / (sp ** 2 / st + rhop) # dlnP/dlnrho|s
        chiy = -1. * rho * y * (1. / rho_he - 1. / rho_h) # dlnrho/dlnY|P,T

        # from scvh.py
        # dpdt_const_rho = - 10 ** logp / 10 ** logt * res['rhot'] / res['rhop']
        # dudt_const_rho = s * (res['st'] - res['sp'] * res['rhot'] / res['rhop'])
        # dpdu_const_rho = dpdt_const_rho / 10 ** res['logrho'] / dudt_const_rho
        # gamma3 = 1. + dpdu_const_rho # cox and giuli 9.93a
        # gamma1 = (gamma3 - 1.) / res['grada']
        # res['gamma3'] = gamma3
        # res['gamma1'] = gamma1
        # res['chirho'] = res['rhop'] ** -1 # rhop = dlogrho/dlogp|t
        # res['chit'] = dpdt_const_rho * 10 ** logt / 10 ** logp
        # res['chiy'] = -1. * 10 ** res['logrho'] * y * (1. / 10 ** res_he['logrho'] - 1. / 10 ** res_h['logrho']) # dlnrho/dlnY|P,T
        # # from mesa's scvh in mesa/eos/eosPT_builder/src/scvh_eval.f
        # # 1005:      Cv = chiT * P / (rho * T * (gamma3 - 1)) ! C&G 9.93
        # # 1006:      Cp = Cv + P * chiT**2 / (Rho * T * chiRho) ! C&G 9.86
        # res['cv'] = res['chit'] * 10 ** logp / (10 ** res['logrho'] * 10 ** logt * (gamma3 - 1.)) # erg g^-1 K^-1
        # res['cp'] = res['cv'] + 10 ** logp * res['chit'] ** 2 / (10 ** res['logrho'] * 10 ** logt * res['chirho']) # erg g^-1 K^-1

        gamma1 = chirho / (1. - chit * grada)
        gamma3 = 1. + gamma1 * grada
        cp = s * st
        cv = cp * chirho / gamma1 # Unno 13.87
        csound = np.sqrt(10 ** logp / rho * gamma1)

        res =  {
            'grada':grada,
            'logrho':logrho,
            'logs':np.log10(s),
            'chirho':chirho,
            'chit':chit,
            'gamma1':gamma1,
            'gamma3':gamma3,
            'chiy':chiy,
            'rho_h':rho_h,
            'rho_he':rho_he,
            'rhop':rhop,
            'rhot':rhot,
            'cp':cp,
            'cv':cv,
            'csound':csound
            }
        # debugging
        res['sp'] = sp
        res['st'] = st
        res['s_h'] = s_h
        return res

    def get_grada(self, logp, logt, y):
        # return self.get(logp, logt, y)['grada'] # this is fine, but more than necessary
        s_h = 10 ** self.get_logs_h(logp, logt)
        sp_h = self.get_sp_h(logp, logt)
        st_h = self.get_st_h(logp, logt)

        # fix up values for low t by asking scvh h directly; mh13+scvh h table only covers logT >= 2.25, about 178 K.
        tlo = self.logtlo_h
        s_h[logt < tlo] = 10 ** self.he_eos.get_h['logs']((logp[logt < tlo], logt[logt < tlo]))
        sp_h[logt < tlo] = self.he_eos.get_h['sp']((logp[logt < tlo], logt[logt < tlo]))
        st_h[logt < tlo] = self.he_eos.get_h['st']((logp[logt < tlo], logt[logt < tlo]))

        s_he = 10 ** self.get_logs_he(logp, logt)
        sp_he = self.get_sp_he(logp, logt)
        st_he = self.get_st_he(logp, logt)

        # smix = 10 ** self.he_eos.get_logsmix(logp, logt, y)
        s = (1. - y) * s_h + y * s_he # + smix
        st = (1. - y) * s_h / s * st_h + y * s_he / s * st_he # + smix/s*dlogsmix/dlogt
        sp = (1. - y) * s_h / s * sp_h + y * s_he / s * sp_he # + smix/s*dlogsmix/dlogp
        grada = - sp / st
        return grada

    def get_logrho(self, logp, logt, y):
        return self.get(logp, logt, y)['logrho']

    def get_logs(self, logp, logt, y):
        return self.get(logp, logt, y)['logs']

    def get_gamma1(self, logp, logt, y):
        return self.get(logp, logt, y)['gamma1']
