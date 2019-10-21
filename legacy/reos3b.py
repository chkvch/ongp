import numpy as np
from scipy.interpolate import bisplrep, bisplev

class eos:
    def __init__(self, path_to_data):

        self.columns = 'logp', 'logt', 'logrho', 'logs'

        self.h_path = '%s/REOS3b-H-2018.dat' % path_to_data
        self.h_data = np.genfromtxt(self.h_path, skip_header=16, names=self.columns)

        self.he_path = '%s/REOS3b-He-2018.dat' % path_to_data
        self.he_data = np.genfromtxt(self.he_path, skip_header=16, names=self.columns)

        # take out large pressures where entropies seem harder to fit, store results in dictionary
        self.h = {}
        self.he = {}
        for key in ('logt', 'logrho', 'logs', 'logp'):
            self.h[key] = np.delete(self.h_data[key], np.where(self.h_data['logp'] > 14.))
            self.he[key] = np.delete(self.he_data[key], np.where(self.he_data['logp'] > 14.))

        kwargs = {'kx':3, 'ky':3}

        self.tck_logrho_h = bisplrep(self.h['logp'], self.h['logt'], self.h['logrho'], **kwargs)
        self.tck_logrho_he = bisplrep(self.he['logp'], self.he['logt'], self.he['logrho'], **kwargs)

        # # logs column wants larger expected number of knots?
        # kwargs['nxest'] = 50
        # kwargs['nyest'] = 50
        self.tck_logs_h = bisplrep(self.h['logp'], self.h['logt'], self.h['logs'], **kwargs)
        self.tck_logs_he = bisplrep(self.he['logp'], self.he['logt'], self.he['logs'], **kwargs)

        self.tck_logt_ps_h = bisplrep(self.h['logp'], self.h['logs'], self.h['logt'], **kwargs)
        self.tck_logt_ps_he = bisplrep(self.he['logp'], self.he['logs'], self.he['logt'], **kwargs)

    def get_logt_ps_h(self, logp, logs):
        return bisplev(logp, logs, self.tck_logt_ps_h).diagonal()
    def get_logt_ps_he(self, logp, logs):
        return bisplev(logp, logs, self.tck_logt_ps_he).diagonal()

    def get_logrho_h(self, logp, logt, dx=0, dy=0):
        return bisplev(logp[::-1], logt[::-1], self.tck_logrho_h, dx=dx, dy=dy).diagonal()[::-1]

    def get_logrho_he(self, logp, logt, dx=0, dy=0):
        return bisplev(logp[::-1], logt[::-1], self.tck_logrho_he, dx=dx, dy=dy).diagonal()[::-1]

    def get_logs_h(self, logp, logt, dx=0, dy=0):
        return bisplev(logp[::-1], logt[::-1], self.tck_logs_h, dx=dx, dy=dy).diagonal()[::-1]

    def get_logs_he(self, logp, logt, dx=0, dy=0):
        return bisplev(logp[::-1], logt[::-1], self.tck_logs_he, dx=dx, dy=dy).diagonal()[::-1]

    def get_logrho(self, logp, logs, y):
        assert y >= 0, 'got bad y %f' % y
        assert y <= 1, 'got bad y %f' % y
        rhoinv = (1. - y) / 10 ** self.get_logrho_h(logp, logs)
        rhoinv += y / 10 ** self.get_logrho_he(logp, logs)
        return np.log10(rhoinv ** -1)

    def get_logs(self, logp, logt, y):
        assert y >= 0, 'got bad y %f' % y
        assert y <= 1, 'got bad y %f' % y
        s = (1. - y) * 10 ** self.get_logs_h(logp, logt)
        s += y * 10 ** self.get_logs_he(logp, logt)
        # s += smix
        return np.log10(s)

    def get_grada(self, logp, logt, y):
        # see Saumon, Chabrier, van Horn (1995) Equations (45-46)
        # correct both equations; should read (1-Y) * S^H/S; (1-Y) * S^He/S

        s_h = 10 ** self.get_logs_h(logp, logt)
        s_he = 10 ** self.get_logs_he(logp, logt)
        s = 10 ** self.get_logs(logp, logt, y)

        st_h = self.get_logs_h(logp, logt, dy=1)
        sp_h = self.get_logs_h(logp, logt, dx=1)
        st_he = self.get_logs_he(logp, logt, dy=1)
        sp_he = self.get_logs_he(logp, logt, dx=1)


        st = (1. - y) * s_h / s * st_h + y * s_he / s * st_he # + smix term
        sp = (1. - y) * s_h / s * sp_h + y * s_he / s * sp_he # + smix term

        return - sp / st


    # def get_dlogs_dlogt_const_p(self, logp, logt, y, f=1e-4, n=4):
    #     logts = np.linspace(logt-f, logt+f, n)
    #     s = CubicSpline(logts, self.get_logs(logp, logts, y), False)
    #     return s.derivative()(logt)
    #
    # def get_dlogs_logp_const_t(self, logp, logt, y, f=1e-4, n=4):
    #     logps = np.linspace(logp-f, logp+f, n)
    #     s = CubicSpline(logps, self.get_logs(logps, logt, y), False)
    #     return s.derivative()(logp)
    #
    # def get_grada(self, logp, logt, y, f=1e-4):
    #     return self.get_dlogs_logp_const_t(logp, logt, y, f) / \
    #         self.get_dlogs_dlogt_const_p(logp, logt, y, f)
