from scipy.interpolate import RectBivariateSpline as rbs
from itertools import islice
import numpy as np

class eos:
    def __init__(self, path_to_data=None, interpolation_order=3):

        if not path_to_data:
            import os
            path_to_data = os.environ['ongp_data_path']

        npts_t = 121
        npts_t -= 60 # skip logT > 5; there are 60 such points
        npts_p = 441
        # - The (log T, log P) tables include NT=121 isotherms
        # between logT=2.0 and logT=8.0 with a step dlogT=0.05.
        # Each isotherm includes NP=441 pressures from
        # logP=-9.0 to log P=+13.0 with a step dlogP=0.05.

#log T [K]        log P [GPa]   log rho [g/cc]  log U [MJ/kg] log S [MJ/kg/K]  dlrho/dlT_P,  dlrho/dlP_T,   dlS/dlT_P,     dlS/dlP_T         grad_ad
        columns = 'logt', 'logp', 'logrho', 'logu', 'logs', 'rhot', 'rhop', 'st', 'sp', 'grada'
        path = f'{path_to_data}/DirEOS2019/TABLEEOS_HHE_TP_Y0.275_v1'
        self.logtvals = np.array([])

        self.data = {}
        for name in columns:
            self.data[name] = np.zeros((npts_p, npts_t))
        it = 0
        with open(path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if line[0] == '#':
                    if '=' in line:
                        if float(line.split()[-1]) > 5:
                            continue
                        with open(path, 'rb') as g:
                            chunk = np.genfromtxt(islice(g, i+1, i+npts_p+1), names=columns)
                        for name in columns:
                            self.data[name][:, it] = chunk[name]
                        self.logtvals = np.append(self.logtvals, chunk['logt'][0])
                        it += 1
        self.data['logp'] += 10 # 1 GPa = 1e10 cgs
        self.data['logu'] += 10 # 1 MJ/kg = 1e13 erg/kg = 1e10 erg/g
        self.data['logs'] += 10 # 1 MJ/kg = 1e13 erg/kg = 1e10 erg/g

        self.logpvals = chunk['logp'] + 10 # logps are always the same, just grab from last chunk read
        self.spline_kwargs = {'kx':interpolation_order, 'ky':interpolation_order}

    def get_logrho(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.data['logrho'], **self.spline_kwargs)(lgp, lgt, grid=False)
    def get_logs(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.data['logs'], **self.spline_kwargs)(lgp, lgt, grid=False)
    # def get_rhop(self, lgp, lgt):
    #     return rbs(self.logpvals, self.logtvals, self.data['rhop'], **self.spline_kwargs)(lgp, lgt, grid=False)
    # def get_rhot(self, lgp, lgt):
    #     return rbs(self.logpvals, self.logtvals, self.data['rhot'], **self.spline_kwargs)(lgp, lgt, grid=False)
    # def get_sp(self, lgp, lgt):
    #     return rbs(self.logpvals, self.logtvals, self.data['sp'], **self.spline_kwargs)(lgp, lgt, grid=False)
    # def get_st(self, lgp, lgt):
    #     return rbs(self.logpvals, self.logtvals, self.data['st'], **self.spline_kwargs)(lgp, lgt, grid=False)

    # actually compute derivatives from splines rather than read derivatives from tables; they may not be reliable
    # (see comments in chabrier.py)
    def get_rhop(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.data['logrho'], **self.spline_kwargs)(lgp, lgt, grid=False, dx=1)
    def get_rhot(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.data['logrho'], **self.spline_kwargs)(lgp, lgt, grid=False, dy=1)
    def get_sp(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.data['logs'], **self.spline_kwargs)(lgp, lgt, grid=False, dx=1)
    def get_st(self, lgp, lgt):
        return rbs(self.logpvals, self.logtvals, self.data['logs'], **self.spline_kwargs)(lgp, lgt, grid=False, dy=1)

    # def get_grada(self, lgp, lgt):
    #     return rbs(self.logpvals, self.logtvals, self.data['grada'], **self.spline_kwargs)(lgp, lgt, grid=False)

    # general method for getting quantities for hydrogen-helium mixture
    def get(self, logp, logt):
        logs = self.get_logs(logp, logt)
        # sp = self.get_sp(logp, logt)
        # st = self.get_st(logp, logt)
        sp = self.get_sp(logp, logt)
        st = self.get_st(logp, logt)
        grada = - sp / st

        logrho = self.get_logrho(logp, logt)
        rhop = self.get_rhop(logp, logt)
        rhot = self.get_rhot(logp, logt)

        chirho = 1. / rhop # dlnP/dlnrho|T
        chit = -1. * rhot / rhop # dlnP/dlnT|rho
        # gamma1 = 1. / (sp ** 2 / st + rhop) # dlnP/dlnrho|s

        gamma1 = chirho / (1. - chit * grada)
        gamma3 = 1. + gamma1 * grada
        cp = 10 ** logs * st
        cv = cp * chirho / gamma1 # Unno 13.87
        csound = np.sqrt(10 ** logp / 10 ** logrho * gamma1)

        res =  {
            'grada':grada,
            'logrho':logrho,
            'logs':logs,
            'gamma1':gamma1,
            'chirho':chirho,
            'chit':chit,
            'gamma1':gamma1,
            'gamma3':gamma3,
            # 'chiy':chiy,
            # 'rho_h':rho_h,
            # 'rho_he':rho_he,
            'rhop':rhop,
            'rhot':rhot,
            'cp':cp,
            'cv':cv,
            'csound':csound
            }
        return res
