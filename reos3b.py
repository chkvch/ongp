import numpy as np
from scipy.interpolate import bisplrep, bisplev

class eos:
    def __init__(self, path_to_data):

        self.columns = 'logp', 'logt', 'logrho', 'logs'

        self.h_path = '%s/REOS3b-H-2018.dat' % path_to_data
        self.h_data = np.genfromtxt(self.h_path, skip_header=16, names=self.columns)

        self.he_path = '%s/REOS3b-He-2018.dat' % path_to_data
        self.he_data = np.genfromtxt(self.he_path, skip_header=16, names=self.columns)

        # Linear 2d interpolation really sucks unless you want noisy derivatives (including grada)
        #
        # h_basis = np.array([self.h_data['logp'], self.h_data['logt']]).T
        # self.get_logrho_h = LinearNDInterpolator(h_basis, self.h_data['logrho'])
        # self.get_logs_h = LinearNDInterpolator(h_basis, self.h_data['logs'])

        # he_basis = np.array([self.he_data['logp'], self.he_data['logt']]).T
        # self.get_logrho_he = LinearNDInterpolator(he_basis, self.he_data['logrho'])
        # self.get_logs_he = LinearNDInterpolator(he_basis, self.he_data['logs'])

        kwargs = {'kx':3, 'ky':3}
        self.tck_logrho_h = bisplrep(self.h_data['logp'], self.h_data['logt'], self.h_data['logrho'], **kwargs)
        self.tck_logrho_he = bisplrep(self.he_data['logp'], self.he_data['logt'], self.he_data['logrho'], **kwargs)

        # logs column wants larger expected number of knots
        kwargs['nxest'] = 50
        kwargs['nyest'] = 50
        self.tck_logs_h = bisplrep(self.h_data['logp'], self.h_data['logt'], self.h_data['logs'], **kwargs)

        # try using knots from logs_h for logs_he
        # kwargs['s'] = 1500
        # self.tck_logs_he = bisplrep(self.he_data['logp'], self.he_data['logt'], self.he_data['logs'], **kwargs)

    def get_logrho_h(self, logp, logt):
        return bisplev(logp[::-1], logt[::-1], self.tck_logrho_h).diagonal()[::-1]
    def get_dlogrho_dlogp_const_t_h(self, logp, logt):
        return bisplev(logp[::-1], logt[::-1], self.tck_logrho_h, dx=1).diagonal()[::-1]
    def get_dlogrho_dlogt_const_p_h(self, logp, logt):
        return bisplev(logp[::-1], logt[::-1], self.tck_logrho_h, dy=1).diagonal()[::-1]

    def get_logrho_he(self, logp, logt):
        return bisplev(logp[::-1], logt[::-1], self.tck_logrho_he).diagonal()[::-1]
    def get_dlogrho_dlogp_const_t_he(self, logp, logt):
        return bisplev(logp[::-1], logt[::-1], self.tck_logrho_he, dx=1).diagonal()[::-1]
    def get_dlogrho_dlogt_const_p_he(self, logp, logt):
        return bisplev(logp[::-1], logt[::-1], self.tck_logrho_he, dy=1).diagonal()[::-1]

    def get_logs_h(self, logp, logt):
        return bisplev(logp[::-1], logt[::-1], self.tck_logs_h).diagonal()[::-1]
    def get_dlogs_dlogp_const_t_h(self, logp, logt):
        return bisplev(logp[::-1], logt[::-1], self.tck_logs_h, dx=1).diagonal()[::-1]
    def get_dlogs_dlogt_const_p_h(self, logp, logt):
        return bisplev(logp[::-1], logt[::-1], self.tck_logs_h, dy=1).diagonal()[::-1]

    def get_logs_he(self, logp, logt):
        return bisplev(logp[::-1], logt[::-1], self.tck_logs_he).diagonal()[::-1]
    def get_dlogs_dlogp_const_t_he(self, logp, logt):
        return bisplev(logp[::-1], logt[::-1], self.tck_logs_he, dx=1).diagonal()[::-1]
    def get_dlogs_dlogt_const_p_he(self, logp, logt):
        return bisplev(logp[::-1], logt[::-1], self.tck_logs_he, dy=1).diagonal()[::-1]

    def get_logrho(self, logp, logt, y):
        assert y >= 0, 'got bad y %f' % y
        assert y <= 1, 'got bad y %f' % y
        rhoinv = (1. - y) / 10 ** self.get_logrho_h(logp, logt)
        rhoinv += y / 10 ** self.get_logrho_he(logp, logt)
        return np.log10(rhoinv ** -1)

    def get_logs(self, logp, logt, y):
        assert y >= 0, 'got bad y %f' % y
        assert y <= 1, 'got bad y %f' % y
        s = (1. - y) * 10 ** self.get_logs_h(logp, logt)
        s += y * 10 ** self.get_logs_he(logp, logt)
        # s += smix
        return np.log10(s)

    # def get_dlogs_dlogt_const_p(self, logp, logt, y, f=1e-4, n=4):
    #     logts = np.linspace(logt-f, logt+f, n)
    #     s = CubicSpline(logts, self.get_logs(logp, logts, y), False)
    #     return s.derivative()(logt)
    #
    #
    # def get_dlogs_logp_const_t(self, logp, logt, y, f=1e-4, n=4):
    #     logps = np.linspace(logp-f, logp+f, n)
    #     s = CubicSpline(logps, self.get_logs(logps, logt, y), False)
    #     return s.derivative()(logp)
    #
    # def get_grada(self, logp, logt, y, f=1e-4):
    #     return self.get_dlogs_logp_const_t(logp, logt, y, f) / \
    #         self.get_dlogs_dlogt_const_p(logp, logt, y, f)
