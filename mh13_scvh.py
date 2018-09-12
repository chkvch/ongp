import numpy as np
from scipy.interpolate import bisplrep, bisplev
import scvh

class eos:
    def __init__(self, path_to_data):

        self.columns = 'logp', 'logt', 'logrho', 'logs'
        self.h_path = '{}/MH13+SCvH-H-2018.dat'.format(path_to_data)
        self.h_data = np.genfromtxt(self.h_path, skip_header=16, names=self.columns)

        kwargs = {'kx':3, 'ky':3}

        self.tck_logrho_h = bisplrep(self.h_data['logp'], self.h_data['logt'], self.h_data['logrho'], **kwargs)
        self.tck_logs_h = bisplrep(self.h_data['logp'], self.h_data['logt'], self.h_data['logs'], **kwargs)

        # get t profile at constant entropy by using p, s as a basis?
        self.tck_logt_h_ps = bisplrep(self.h_data['logp'], self.h_data['logs'], self.h_data['logt'], **kwargs)
        self.tck_logp_h_ts = bisplrep(self.h_data['logt'], self.h_data['logs'], self.h_data['logp'], **kwargs)

        self.scvh_eos = scvh.eos(path_to_data)

    def get_logrho_he(self, logp, logt, dx=0, dy=0):
        if dx == 0 and dy == 0:
            return self.scvh_eos.get_he_logrho((logp, logt))
        elif dx == 1 and dy == 0:
            return self.scvh_eos.get_he_rhop((logp, logt))
        elif dx == 0 and dy == 1:
            return self.scvh_eos.get_he_rhot((logp, logt))
        else:
            raise ValueError('invalid combination of partials (%i, %i) for logrho_he from scvh' % (dx, dy))

    def get_logs_he(self, logp, logt, dx=0, dy=0):
        if dx == 0 and dy == 0:
            return self.scvh_eos.get_he_logs((logp, logt))
        elif dx == 1 and dy == 0:
            return self.scvh_eos.get_he_sp((logp, logt))
        elif dx == 0 and dy == 1:
            return self.scvh_eos.get_he_st((logp, logt))
        else:
            raise ValueError('invalid combination of partials (%i, %i) for logs_he from scvh' % (dx, dy))

    def get_logs_h(self, logp, logt, dx=0, dy=0):
        return bisplev(logp[::-1], logt[::-1], self.tck_logs_h, dx=dx, dy=dy).diagonal()[::-1]

    def get_logrho_h(self, logp, logt, dx=0, dy=0):
        return bisplev(logp[::-1], logt[::-1], self.tck_logrho_h, dx=dx, dy=dy).diagonal()[::-1]

    def get_logrho(self, logp, logt, y):
        rhoinv = y / 10 ** self.get_logrho_he(logp, logt)
        rhoinv += (1. - y) / 10 ** self.get_logrho_h(logp, logt)
        return np.log10(rhoinv ** -1.)

    def get_logs(self, logp, logt, y):
        s = y * 10 ** self.get_logs_he(logp, logt)
        s += (1. - y) * 10 ** self.get_logs_h(logp, logt)
        return np.log10(s)

    def _get_s_(self, logp, logt, y):
        s_h = 10 ** self.get_logs_h(logp, logt)
        s_he = 10 ** self.get_logs_he(logp, logt)
        s = y * s_he + (1. - y) * s_h
        return s_h, s_he, s

    def _get_rho_(self, logp, logt, y):
        rho_h = 10 ** self.get_logrho_h(logp, logt)
        rho_he = 10 ** self.get_logrho_he(logp, logt)
        rhoinv = (1. - y) / rho_h
        rhoinv += y / rho_he
        rho = rhoinv ** -1.
        return rho_h, rho_he, rho

    def get_grada(self, logp, logt, y):
        s_h, s_he, s = self._get_s_(logp, logt, y)

        sp_h = self.get_logs_h(logp, logt, dx=1, dy=0)
        st_h = self.get_logs_h(logp, logt, dx=0, dy=1)
        sp_he = self.get_logs_he(logp, logt, dx=1, dy=0)
        st_he = self.get_logs_he(logp, logt, dx=0, dy=1)

        st = (1. - y) * s_h / s * st_h
        st += y * s_he / s * st_he

        sp = (1. - y) * s_h / s * sp_h
        sp += y * s_he / s * sp_he

        return - sp / st

    def get_grada_diff(self, logp, logt, y):
        return


    # dpdt_const_rho = - 10 ** logp / 10 ** logt * res['rhot'] / res['rhop']
    # dudt_const_rho = s * (res['st'] - res['sp'] * res['rhot'] / res['rhop'])
    # dpdu_const_rho = dpdt_const_rho / 10 ** res['logrho'] / dudt_const_rho
    # gamma3 = 1. + dpdu_const_rho # cox and giuli 9.93a
    # gamma1 = (gamma3 - 1.) / res['grada']
    # res['gamma3'] = gamma3
    # res['gamma1'] = gamma1
    # res['chirho'] = res['rhop'] ** -1 # rhop = dlogrho/dlogp|t
    # res['chit'] = dpdt_const_rho * 10 ** logt / 10 ** logp

    def get_chirho(self, logp, logt, y):
        rhop_h = self.get_logrho_h(logp, logt, dx=1, dy=0)
        rhot_h = self.get_logrho_h(logp, logt, dx=0, dy=1)
        rhop_he = self.get_logrho_he(logp, logt, dx=1, dy=0)
        rhot_he = self.get_logrho_he(logp, logt, dx=0, dy=1)

        rho_h, rho_he, rho = self._get_rho_(logp, logt, y)
        rhot = (1. - y) * rho / rho_h * rhot_h
        rhot += y * rho / rho_he * rhot_he
        rhop = (1. - y) * rho / rho_h * rhop_h
        rhop += y * rho / rho_he * rhop_he

        chirho = rhop ** -1.
        chit = - rhot / rhop

        return chirho

    def get_chit(self, logp, logt, y):
        rhop_h = self.get_logrho_h(logp, logt, dx=1, dy=0)
        rhot_h = self.get_logrho_h(logp, logt, dx=0, dy=1)
        rhop_he = self.get_logrho_he(logp, logt, dx=1, dy=0)
        rhot_he = self.get_logrho_he(logp, logt, dx=0, dy=1)

        rho_h, rho_he, rho = self._get_rho_(logp, logt, y)
        rhot = (1. - y) * rho / rho_h * rhot_h
        rhot += y * rho / rho_he * rhot_he
        rhop = (1. - y) * rho / rho_h * rhop_h
        rhop += y * rho / rho_he * rhop_he

        chit = - rhot / rhop

        return chit
