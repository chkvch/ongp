import numpy as np
import const; reload(const)
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

class eos:
    
    def __init__(self, path_to_data, use_raw_tables=False):
        '''load the Saumon, Chabrier, van Horn 1995 EOS tables for H and He.
        the eos tables were pulled from mesa-r8845/eos/eosDT_builder/eos_input_data/scvh/.
        to see the dependent variables available, check the attributes eos.h_names and eos.he_names.'''
        
        self.path_to_data = path_to_data
        
        # not using these at present, just making them available for reference
        self.logtmin, self.logtmax = 2.10, 7.06
        
        if use_raw_tables:
            path_to_h_data = '%s/scvh_h_original.dat' % self.path_to_data
            path_to_he_data = '%s/scvh_he_original.dat' % self.path_to_data
        else:
            # use versions with two extra nodes in logp, logt, calculated by extrapolating on isotherms.
            # I tended to run just too low in logT near logP=11.4, 11.6 for Saturn models.
            path_to_h_data = '%s/scvh_h.dat' % self.path_to_data
            path_to_he_data = '%s/scvh_he.dat' % self.path_to_data
        
        self.h_names = 'logp', 'xh2', 'xh', 'logrho', 'logs', 'logu', 'rhot', 'rhop', 'st', 'sp', 'grada'
        self.h_data = {}
        logtvals_h = np.array([])
        with open(path_to_h_data) as fr:
            for i, line in enumerate(fr.readlines()):
                if len(line.split()) is 2:
                    logt, nrows = line.split()
                    logt = float(logt)
                    nrows = int(nrows)
                    # print 'reading %i rows for logT = %f' % (nrows, logt)
                    data = np.genfromtxt(path_to_h_data, skip_header=i+1, max_rows=nrows, 
                        names=self.h_names)
                    self.h_data[logt] = data
                    logtvals_h = np.append(logtvals_h, logt)
                    
        self.he_names = 'logp', 'xhe', 'xhep', 'logrho', 'logs', 'logu', 'rhot', 'rhop', 'st', 'sp', 'grada'
        self.he_data = {}
        logtvals_he = np.array([])
        with open(path_to_he_data) as fr:
            for i, line in enumerate(fr.readlines()):
                if len(line.split()) is 2:
                    logt, nrows = line.split()
                    logt = float(logt)
                    nrows = int(nrows)
                    # print 'reading %i rows for logT = %f' % (nrows, logt)
                    data = np.genfromtxt(path_to_he_data, skip_header=i+1, max_rows=nrows, 
                        names=self.he_names)
                    self.he_data[logt] = data
                    logtvals_he = np.append(logtvals_he, logt)
                    
        assert np.all(logtvals_h == logtvals_he) # verify H an He are on the same temperature grid
        self.logtvals = logtvals_h
        
        # set up reasonable rectangular grid in logP for the purposes of modelling Jupiter and Saturn-mass planets. 
        # points not in the original tables will just return nans.
        self.logpvals = np.union1d(self.h_data[2.1]['logp'], self.h_data[5.06]['logp'])
        self.logpmin, self.logpmax = 4.0, 17. # 14.0 # january 17 2017: extend to high p for daniel
        self.logpvals = self.logpvals[self.logpvals >= self.logpmin]
        self.logpvals = self.logpvals[self.logpvals <= self.logpmax]
        
        npts_t = len(self.logtvals)
        npts_p = len(self.logpvals)
        basis_shape = (npts_p, npts_t)
        
        self.h_xh2 = np.zeros(basis_shape)
        self.h_xh = np.zeros(basis_shape)
        self.h_logrho = np.zeros(basis_shape)
        self.h_logs = np.zeros(basis_shape)
        self.h_logu = np.zeros(basis_shape)
        self.h_rhot = np.zeros(basis_shape)
        self.h_rhop = np.zeros(basis_shape)
        self.h_st = np.zeros(basis_shape)
        self.h_sp = np.zeros(basis_shape)
        self.h_grada = np.zeros(basis_shape)
        
        self.he_xhe = np.zeros(basis_shape)
        self.he_xhep = np.zeros(basis_shape)
        self.he_logrho = np.zeros(basis_shape)
        self.he_logs = np.zeros(basis_shape)
        self.he_logu = np.zeros(basis_shape)
        self.he_rhot = np.zeros(basis_shape)
        self.he_rhop = np.zeros(basis_shape)
        self.he_st = np.zeros(basis_shape)
        self.he_sp = np.zeros(basis_shape)
        self.he_grada = np.zeros(basis_shape)
        
        for ip, logp in enumerate(self.logpvals):
            for it, logt in enumerate(self.logtvals):
                self.h_xh2[ip, it] = self.get_h_on_node('xh2', logp, logt)
                self.h_xh[ip, it] = self.get_h_on_node('xh', logp, logt)
                self.h_logrho[ip, it] = self.get_h_on_node('logrho', logp, logt)
                self.h_logs[ip, it] = self.get_h_on_node('logs', logp, logt)
                self.h_logu[ip, it] = self.get_h_on_node('logu', logp, logt)
                self.h_rhot[ip, it] = self.get_h_on_node('rhot', logp, logt)
                self.h_rhop[ip, it] = self.get_h_on_node('rhop', logp, logt)
                self.h_st[ip, it] = self.get_h_on_node('st', logp, logt)
                self.h_sp[ip, it] = self.get_h_on_node('sp', logp, logt)
                self.h_grada[ip, it] = self.get_h_on_node('grada', logp, logt)
                
                self.he_xhe[ip, it] = self.get_he_on_node('xhe', logp, logt)
                self.he_xhep[ip, it] = self.get_he_on_node('xhep', logp, logt)
                self.he_logrho[ip, it] = self.get_he_on_node('logrho', logp, logt)
                self.he_logs[ip, it] = self.get_he_on_node('logs', logp, logt)
                self.he_logu[ip, it] = self.get_he_on_node('logu', logp, logt)
                self.he_rhot[ip, it] = self.get_he_on_node('rhot', logp, logt)
                self.he_rhop[ip, it] = self.get_he_on_node('rhop', logp, logt)
                self.he_st[ip, it] = self.get_he_on_node('st', logp, logt)
                self.he_sp[ip, it] = self.get_he_on_node('sp', logp, logt)
                self.he_grada[ip, it] = self.get_he_on_node('grada', logp, logt)
                
        self.get_h_xh2 = RegularGridInterpolator((self.logpvals, self.logtvals), self.h_xh2)
        self.get_h_xh = RegularGridInterpolator((self.logpvals, self.logtvals), self.h_xh)
        self.get_h_logrho = RegularGridInterpolator((self.logpvals, self.logtvals), self.h_logrho)
        self.get_h_logs = RegularGridInterpolator((self.logpvals, self.logtvals), self.h_logs)
        self.get_h_logu = RegularGridInterpolator((self.logpvals, self.logtvals), self.h_logu)
        self.get_h_rhot = RegularGridInterpolator((self.logpvals, self.logtvals), self.h_rhot)
        self.get_h_rhop = RegularGridInterpolator((self.logpvals, self.logtvals), self.h_rhop)
        self.get_h_st = RegularGridInterpolator((self.logpvals, self.logtvals), self.h_st)
        self.get_h_sp = RegularGridInterpolator((self.logpvals, self.logtvals), self.h_sp)
        self.get_h_grada = RegularGridInterpolator((self.logpvals, self.logtvals), self.h_grada)
        
        self.get_he_xhe = RegularGridInterpolator((self.logpvals, self.logtvals), self.he_xhe)
        self.get_he_xhep = RegularGridInterpolator((self.logpvals, self.logtvals), self.he_xhep)
        self.get_he_logrho = RegularGridInterpolator((self.logpvals, self.logtvals), self.he_logrho)
        self.get_he_logs = RegularGridInterpolator((self.logpvals, self.logtvals), self.he_logs)
        self.get_he_logu = RegularGridInterpolator((self.logpvals, self.logtvals), self.he_logu)
        self.get_he_rhot = RegularGridInterpolator((self.logpvals, self.logtvals), self.he_rhot)
        self.get_he_rhop = RegularGridInterpolator((self.logpvals, self.logtvals), self.he_rhop)
        self.get_he_st = RegularGridInterpolator((self.logpvals, self.logtvals), self.he_st)
        self.get_he_sp = RegularGridInterpolator((self.logpvals, self.logtvals), self.he_sp)
        self.get_he_grada = RegularGridInterpolator((self.logpvals, self.logtvals), self.he_grada)
        
    # these wrapper functions are the ones meant to be called externally.
    
    def get(self, logp, logt, y):
        '''return all eos results for a (logp, logt) pair at any H-He mixture.'''
        if type(logp) is np.float64 or type(logp) is float: logp = np.array([logp])
        if type(logt) is np.float64 or type(logt) is float: logt = np.array([logt])
        if type(y) is np.float64 or type(y) is float: y = np.array([y])
        pair = (logp, logt)
        res = {}
        if type(y) is np.ndarray:
            assert np.all(0. <= y) and np.all(y <= 1.), 'invalid helium mass fraction'
        elif type(y) is np.float64:
            assert 0. <= y <= 1., 'invalid helium mass fraction %f' % y
        try:
            res = self.get_hhe(pair, y)
        except ValueError:
            print 'probably out of bounds in logP, logT, or Y -- did you accidentally pass P, T? (or loglogP, loglogT?)'
            print logp
            print logt
            raise
            
        res['logp'] = logp
        res['logt'] = logt
            
        return res
                
    # convenience routines for essential quantities
    
    def get_logrho(self, logp, logt, y):
        return self.get(logp, logt, y)['logrho']
        
    def get_logs(self, logp, logt, y):
        return self.get(logp, logt, y)['logs']
    
    def get_logsmix(self, logp, logt, y):
        return self.get(logp, logt, y)['logsmix']

    def get_logu(self, logp, logt, y):
        return self.get(logp, logt, y)['logu']

    def get_grada(self, logp, logt, y):
        return self.get(logp, logt, y)['grada']

    def get_gamma1(self, logp, logt, y):
        return self.get(logp, logt, y)['gamma1']
                
    def get_chirho(self, logp, logt, y):
        return self.get(logp, logt, y)['chirho']

    def get_chit(self, logp, logt, y):
        return self.get(logp, logt, y)['chit']        
        
    def get_rhot(self, logp, logt, y):
        return self.get(logp, logt, y)['rhot']
        
    def get_cv(self, logp, logt, y):
        return self.get(logp, logt, y)['cv']

    def get_cp(self, logp, logt, y):
        return self.get(logp, logt, y)['cp']

    def rhot_get(self, logrho, logt, y, logp_guess=None):
        # if want to use (rho, t, y) as basis.
        # comes at the expense of doing root finds; looks to take an order of magnitude
        # more cpu time than a call to self.get (30 ms versus 2.5 ms).
        # this is prohibitively slow -- 30 ms per zone times 10^3 zones is 30 seconds.
        # then a 100-step evolutionary model is taking close to an hour.
        
        if type(logrho) is float or type(logrho) is np.float64: logrho = np.array([logrho])
        if type(logt) is float or type(logt) is np.float64: logt = np.array([logt])
        if type(y) is float or type(y) is np.float64: y = np.array([y])
        
        assert len(logrho) == 1 and len(logt) == 1 and len(y) == 1, 'rhot_get only works for length-1 arrays at present.'
        def zero_me(logpval):
            return self.get(np.array([logpval]), logt, y)['logrho'] - logrho
        
        if logp_guess:
            # a good starting guess only helps marginally with time for root find (order unity)
            logpmin = logp_guess * (1. - 5e-4)
            logpmax = logp_guess * (1. + 5e-4)
        else:
            logt_lower = self.logtvals[np.where(np.sign(self.logtvals - logt) > 0)[0][0] - 1]
            logt_upper = self.logtvals[np.where(np.sign(self.logtvals - logt) > 0)[0][0]]
            logpmax_lot = max(self.h_data[logt_lower]['logp'])
            logpmax_hit = max(self.h_data[logt_upper]['logp'])
            logpmax = min(logpmax_lot, logpmax_hit)
            logpmax = min(self.logpmax, logpmax)
            logpmin = self.logpmin
            
        
        # print 'logp brackets for root find: ', self.logpmin, logpmax
        logp, solve_details = brentq(zero_me, logpmin, logpmax, full_output=True)
        # print 'brentq found root (logp = %f) in %i iterations' % (logp, solve_details.iterations)
        res = self.get(logp, logt, y)

        res['logp'] = logp
        res['logt'] = logt

        return res

    def get_chiy(self, logp, logt, y, f=5e-3):
        """dlogrho/dlogY at const p, t"""
        y_lo = y * (1. - f)
        y_hi = y * (1. + f)
        if np.any(y_lo < 0.) or np.any(y_hi > 1.):
            print 'warning: chiy not calculable for y this close to 0 or 1. should change size of step for finite differences.'
            return None

        # logrho = self.get_logrho(logp, logt, y)
        # logp_lo = self.rhot_get(logrho, logt, y_lo)['logp']
        # logp_hi = self.rhot_get(logrho, logt, y_hi)['logp']

        # return (logp_hi - logp_lo) / 2. / f

        logrho_lo = self.get_logrho(logp, logt, y_lo)
        logrho_hi = self.get_logrho(logp, logt, y_hi)
        return (logrho_hi  - logrho_lo) / 2. / f
        
        
        
    # def get_dlogrho_dlogy_numerical(self, logp, logt, y, f=5e-3):
    #     y_lo = y * (1. - f)
    #     y_hi = y * (1. + f)
    #     if np.any(y_lo <= 0.) or np.any(y_hi >= 1.):
    #         print 'warning: dlogrho_dlogy not calculable for y this close to 0 or 1. should change size of step for finite differences.'
    #         return None
    #
    #     logrho_y_lo = self.get_logrho(logp, logt, y_lo)
    #     logrho_y_hi = self.get_logrho(logp, logt, y_hi)
    #
    #     return (logrho_y_hi - logrho_y_lo) / (np.log10(y_hi) - np.log10(y_lo))

    # actually, no need to do this numerically -- since it's just additive volume, it's simple analytically
    def get_dlogrho_dlogy(self, logp, logt, y):
        rho = 10 ** self.get_logrho(logp, logt, y)
        rho_h = 10 ** self.get_h_logrho((logp, logt))
        rho_he = 10 ** self.get_he_logrho((logp, logt))
        return -1. * rho * y * (1. / rho_he - 1. / rho_h)
    
    
    # rest of these are only meant to be called internally    
    
    def get_h_on_node(self, qty, logp, logt):
        '''return quantity `qty' at a single (logp, logt) pair *on* the hydrogen grid.
        qty is a string corresponding to any one of the dependent variables, e.g., 'logs'.'''
        data_for_this_logt = self.h_data[logt]
        try:
            return data_for_this_logt[data_for_this_logt['logp'] == logp][qty][0]
        except IndexError: # off tables
            return np.nan
        
    def get_he_on_node(self, qty, logp, logt):
        '''return quantity `qty' at a single (logp, logt) pair *on* the helium grid.
        qty is a string corresponding to any one of the dependent variables, e.g., 'logs'.'''
        data_for_this_logt = self.he_data[logt]
        try:
            return data_for_this_logt[data_for_this_logt['logp'] == logp][qty][0]
        except IndexError: # off tables
            return np.nan

    def get_hhe(self, pair, y):
        '''combines the results of the hydrogen and helium equations of state for an arbitrary 
        mixture of the two. takes the helium mass fraction Y as input. makes use of the equations
        in SCvH 1995, namely equations 39, 41, 45-47, and 53-56, with typos corrected as per 
        Baraffe et al. 2008 (footnote 4).
        pair is just the tuple (logp, logt).'''
        
        # can't be bothered to sort out handling of pure H or He for now
        if np.any(y == 0.):
            raise ValueError('get_hhe cannot handle pure H for the time being. try a mixture.')
        if np.any(y == 1.):
            raise ValueError('get_hhe cannot handle pure He for the time being. try a mixture.')
        
        def get_beta(y):
            '''eq. 54 of SCvH95.
            y is the helium mass fraction.'''
            try:
                return const.mh / const.mhe * y / (1. - y)
            except ZeroDivisionError:
                print 'tried divide by zero in beta'
                return np.nan
        
        def get_gamma(xh, xh2, xhe, xhep):
            '''eq. 55 of SCvH95.
            xh is the number fraction of atomic H (relative to all particles, inc. electrons)
            xh2 is that of molecular H
            xhe is that of neutral helium
            xhep is that of singly ionized helium
            '''
            return 3. / 2 * (1. + xh + 3 * xh2) / (1. + 2 * xhe + xhep)
            
        def get_delta(y, xh, xh2, xhe, xhep):
            '''eq. 56 of SCvH95.
            input parameters are as for the functions beta and gamma.
            prefactor is corrected as pointed out in footnote 4 of Baraffe et al. 2008;
            flipped relative to how it appears in SCvH95 eq. 56.'''
                        
            species_num = (2. - 2. * xhe - xhep) # proportional to the abundance of free electrons assuming pure He
            species_den = (1. - xh2 - xh) # proportional to the abundance of free electrons assuming pure H
            
            # print 'type xh, xh2, xhe, xhep:', type(xh), type(xh2), type(xhe), type(xhep)
            # print 'xh, xh2, xhe, xhep:', xh, xh2, xhe, xhep
            # print type(species_num)
            # print type(species_den)

            if type(xh) is np.ndarray:
                assert type(species_num) is np.ndarray, 'species are ndarray, but delta numerator is not.'
                assert type(species_den) is np.ndarray, 'species are ndarray, but delta denominator is not.'
                # number density of free e- for one of the pure species is sometimes a tiny negative number.
                # in cases where there are no free electrons, delta does not matter (prefactor vanishes), it's just crucial that it's > 0 and not a nan.
                species_num[species_num <= 0.] = 1.
                species_den[species_den <= 0.] = 1.
            elif type(xh) is np.float64:
                if species_num <= 0.: species_num = 1.
                if species_den <= 0.: species_den = 1.
            else:
                raise TypeError('input type %s not recognized in get_delta' % str(type(xh)))
                            
            return 2. / 3 * species_num / species_den * get_beta(y) * get_gamma(xh, xh2, xhe, xhep)
                        
        def get_smix(y, xh, xh2, xhe, xhep):
            '''ideal entropy of mixing for H/He, same units as H and He tables -- erg K^-1 g^-1.
            eq. 53 of SCvH95.
            input parameters as above.'''
            
            beta = get_beta(y)
            gamma = get_gamma(xh, xh2, xhe, xhep)
            delta = get_delta(y, xh, xh2, xhe, xhep)
            
            # assert not np.any(np.isnan(beta)), 'beta nan'
            # assert not np.any(np.isnan(gamma)), 'gamma nan'
            # assert not np.any(np.isnan(delta)), 'delta nan'
            
            # print d
            xeh = 1. / 2 * (1. - xh2 - xh) # number fraction of e-, for pure H -- eq. 34.
            xehe = 1. / 3 * (2. - 2. * xhe - xhep) # number fraction of e- for pure He -- eq. 35.
            return const.kb * (1. - y) / const.mh * 2. / (1. + xh + 3. * xh2) * \
                (np.log(1. + beta * gamma) \
                - xeh * np.log(1. + delta) \
                + beta * gamma * (np.log(1. + 1. / beta / gamma) \
                - xehe * np.log(1. + 1. / delta)))
                
        def species_partials(pair, y, f=5e-3): 
            # f is a dimensionless factor by which we perturb logp, logt to compute centered differences for numerical partials.
            # returns pair (d*_dlogp, d*_dlogt) of quadruples (dxh2_dlog*, dxh_dlog*, dhe_dlog*, dhep_dlog*)
            
            logp, logt = pair
            
            if np.any(logp * (1. + f) > self.logpmax): return ((0., 0., 0., 0.,), (0., 0., 0., 0.))
            if np.any(logp * (1. - f) < self.logpmax): return ((0., 0., 0., 0.,), (0., 0., 0., 0.))
            if np.any(logt * (1. + f) > self.logtmax): return ((0., 0., 0., 0.,), (0., 0., 0., 0.))
            if np.any(logt * (1. - f) < self.logtmax): return ((0., 0., 0., 0.,), (0., 0., 0., 0.))
                        
            pair_p_plus = logp * (1. + f), logt
            pair_p_minus = logp * (1. - f), logt
            xh2_p_plus = self.get_h_xh2(pair_p_plus)
            xh2_p_minus = self.get_h_xh2(pair_p_minus)
            xh_p_plus = self.get_h_xh(pair_p_plus)
            xh_p_minus = self.get_h_xh(pair_p_minus)
            xhe_p_plus = self.get_he_xhe(pair_p_plus)
            xhe_p_minus = self.get_he_xhe(pair_p_minus)
            xhep_p_plus = self.get_he_xhep(pair_p_plus)
            xhep_p_minus = self.get_he_xhep(pair_p_minus)
                        
            pair_t_plus = logp, logt * (1. + f)
            pair_t_minus = logp, logt * (1. - f)
            xh2_t_plus = self.get_h_xh2(pair_t_plus)
            xh2_t_minus = self.get_h_xh2(pair_t_minus)
            xh_t_plus = self.get_h_xh(pair_t_plus)
            xh_t_minus = self.get_h_xh(pair_t_minus)
            xhe_t_plus = self.get_he_xhe(pair_t_plus)
            xhe_t_minus = self.get_he_xhe(pair_t_minus)
            xhep_t_plus = self.get_he_xhep(pair_t_plus)
            xhep_t_minus = self.get_he_xhep(pair_t_minus)
            
            dxh2_dlogp = (xh2_p_plus - xh2_p_minus) / (2. * f)
            dxh_dlogp = (xh_p_plus - xh_p_minus) / (2. * f)
            dxhe_dlogp = (xhe_p_plus - xhe_p_minus) / (2. * f)
            dxhep_dlogp = (xhep_p_plus - xhep_p_minus) / (2. * f)
            
            dxh2_dlogt = (xh2_t_plus - xh2_t_minus) / (2. * f)
            dxh_dlogt = (xh_t_plus - xh_t_minus) / (2. * f)
            dxhe_dlogt = (xhe_t_plus - xhe_t_minus) / (2. * f)
            dxhep_dlogt = (xhep_t_plus - xhep_t_minus) / (2. * f)
            
            d_dlogp = (dxh2_dlogp, dxh_dlogp, dxhe_dlogp, dxhep_dlogp)
            d_dlogt = (dxh2_dlogt, dxh_dlogt, dxhe_dlogt, dxhep_dlogt)
            
            return d_dlogp, d_dlogt
            
        def y_partials(pair, y, f=5e-3):

            logp, logt = pair

            pair_p_plus = logp * (1. + f), logt
            pair_p_minus = logp * (1. - f), logt
            pair_t_plus = logp, logt * (1. + f)
            pair_t_minus = logp, logt * (1. - f)
            y_plus = y * (1. + f)
            y_minus = y * (1. - f)
            
            if np.any(logp * (1. + f) > self.logpmax): return None
            if np.any(logp * (1. - f) < self.logpmax): return None
            if np.any(logt * (1. + f) > self.logtmax): return None
            if np.any(logt * (1. - f) < self.logtmax): return None
            if np.any(y_plus > 1.): return None
            if np.any(y_minus < 1.): return None
            
            
                                            
        res_h = {}
        res_h['xh2'] = self.get_h_xh2(pair)
        res_h['xh'] = self.get_h_xh(pair)
        res_h['xhe'] = np.zeros_like(pair[0])
        res_h['xhep'] = np.zeros_like(pair[0])
        res_h['logrho'] = self.get_h_logrho(pair)
        res_h['logs'] = self.get_h_logs(pair)
        res_h['logu'] = self.get_h_logu(pair)
        res_h['rhot'] = self.get_h_rhot(pair)
        res_h['rhop'] = self.get_h_rhop(pair)
        res_h['st'] = self.get_h_st(pair)
        res_h['sp'] = self.get_h_sp(pair)
        res_h['grada'] = self.get_h_grada(pair)

        res_he = {}
        res_he['xh2'] = np.zeros_like(pair[0])
        res_he['xh'] = np.zeros_like(pair[0])
        res_he['xhe'] = self.get_he_xhe(pair)
        res_he['xhep'] = self.get_he_xhep(pair)
        res_he['logrho'] = self.get_he_logrho(pair)
        res_he['logs'] = self.get_he_logs(pair)
        res_he['logu'] = self.get_he_logu(pair)
        res_he['rhot'] = self.get_he_rhot(pair)
        res_he['rhop'] = self.get_he_rhop(pair)
        res_he['st'] = self.get_he_st(pair)
        res_he['sp'] = self.get_he_sp(pair)
        res_he['grada'] = self.get_he_grada(pair)
        
        # assert not np.any(np.isnan(res_h['xh2'])), 'got nan in h eos call within overall P-T limits. probably off the original tables.'
        # assert not np.any(np.isnan(res_he['xhe'])), 'got nan in he eos call within overall P-T limits. probably off the original tables.'
        
        res = {}
        rho_h = 10 ** res_h['logrho']
        rho_he = 10 ** res_he['logrho']
        rhoinv = (1. - y) / rho_h + y / rho_he # additive volume rule -- eq. 39.
        rho = rhoinv ** -1.
        res['logrho'] = np.log10(rho)
        
        u = (1. - y) * 10 ** res_h['logu'] + y * 10 ** res_he['logu'] # also additive volume -- eq. 40.
        res['logu'] = np.log10(u)
        
        res['rhot'] = (1. - y) * rho / rho_h * res_h['rhot'] + y * rho / rho_he * res_he['rhot']
        res['rhop'] = (1. - y) * rho / rho_h * res_h['rhop'] + y * rho / rho_he * res_he['rhop']
        
        # note-- additive volume approximation for internal energy and density means if you do the energy 
        # equation with du, drho, you're leaving out the contribution from d(entropy of mixing).
        # this is included if you're differencing the entropy, which includes s_mix.
        
        s_h = 10 ** res_h['logs']
        s_he = 10 ** res_he['logs']
        xh = res_h['xh']
        xh2 = res_h['xh2']
        xhe = res_he['xhe']
        xhep = res_he['xhep']
        smix = get_smix(y, xh, xh2, xhe, xhep)
        s = (1. - y) * s_h + y * s_he + smix # entropy for an ideal (noninteracting) mixture -- eq. 41.
        
        res['logs'] = np.log10(s)
        res['logsmix'] = np.log10(smix)
        res['xh'] = xh
        res['xh2'] = xh2
        res['xhe'] = xhe
        res['xhep'] = xhep
                
        # the bits to compute derivatives of entropy, and thus grad_ad, make use of analytic derivatives of SCvH 1995 eq. 53 with respect to abundances
        # of the four independent species (see CM notes 11/29/2016). the derivatives of each abundance with respect to logp and logt are computed 
        # numerically with centered finite differences. equations with alphanumeric labels (A*) are in the handwritten notes.

        dxi_dlogp, dxi_dlogt = species_partials(pair, y, f=1e-10)
        dxh2_dlogp, dxh_dlogp, dxhe_dlogp, dxhep_dlogp = dxi_dlogp
        dxh2_dlogt, dxh_dlogt, dxhe_dlogt, dxhep_dlogt = dxi_dlogt
        
        # prefactor defined such that smix = smix_prefactor * s_tilde, where s_tilde is the dimensionless entropy I work with in the handwritten notes. (in code below i'll refer to s_tilde as ss)
        smix_prefactor = 2. * const.kb * (1. - y) / const.mh
                                                             
        beta = get_beta(y)
        gamma = get_gamma(xh, xh2, xhe, xhep)
        delta = get_delta(y, xh, xh2, xhe, xhep)
                        
        # eqs. (A5-A8)
        dgamma_dxh2 = 9. / 2 * (1. + 2 * xhe + xhep) ** -1
        dgamma_dxh = dgamma_dxh2 / 3.
        dgamma_dxhe = -3. * (1. + xh + 3 * xh2) / (1. + 2 * xhe + xhep) ** 2
        dgamma_dxhep = dgamma_dxhe / 2.
        
        # eqs. (A9-A12)
        num = (2. - 2 * xhe - xhep)
        num[num < 0.] = 0
        den = (1. - xh2 - xh)
        
        # special handling is required for cases where hydrogen (and thus helium) is totally neutral, or else dividing by zero
        hydrogen_is_neutral = den == 0.
                                                  
        if type(xh) is np.ndarray:
            den[hydrogen_is_neutral] = 1. # kludge to guarantee that delta derivs are calculable. we'll zero them in the neutral case afterward.
        elif type(xh) is np.float64:
            if hydrogen_is_neutral: den = 1.
        else:
            raise TypeError('type %s not recognized in get_hhe' % str(type(xh)))
            
        ddelta_dxh2 = 2. / 3 * num / den ** 2 * beta * gamma + delta / gamma * dgamma_dxh2
        ddelta_dxh = 2. / 3 * num / den ** 2 * beta * gamma + delta / gamma * dgamma_dxh
        ddelta_dxhe = - 4. / 3 * den ** -1 * beta * gamma + delta / gamma * dgamma_dxhe
        ddelta_dxhep = -2. / 3 * den ** -1 * beta * gamma + delta / gamma * dgamma_dxhep
        
        ddelta_dxh2[hydrogen_is_neutral] = 0.
        ddelta_dxh[hydrogen_is_neutral] = 0.
        ddelta_dxhe[hydrogen_is_neutral] = 0.
        ddelta_dxhep[hydrogen_is_neutral] = 0.
                
        in_square_brackets = np.log(1. + 1. / beta / gamma) - 1. / 3 * (2. - 2 * xhe - xhep) * np.log(1. + 1. / delta)
        in_curly_brackets = np.log(1. + beta * gamma) - 1. / 2 * (1. - xh2 - xh) * np.log(1. + delta) + \
                            beta * gamma * in_square_brackets
                            
        dss_dxh2 = -1. * (1. + xh + 3 * xh2) ** -2 * 3 * in_curly_brackets + \
                    (1. + xh + 3 * xh2) ** -1 * ((1. + beta * gamma) ** -1 * beta * dgamma_dxh2 + \
                    1. / 2 * np.log(1. + delta) - 1. / 2 * (1. - xh2 - xh) * (1. + delta) ** -1 * ddelta_dxh2 + \
                    beta * dgamma_dxh2 * in_square_brackets + \
                    beta * gamma * ((1. + 1. / beta / gamma) ** -1 * (-1.) / beta / gamma ** 2 * dgamma_dxh2)) # eq. (A1)
        dss_dxh = -1. * (1. + xh * 3 * xh2) ** -2 * in_curly_brackets + \
                    (1. + xh + 3 * xh2) ** -1 * ((1. + beta * gamma) ** -1 * beta * dgamma_dxh + \
                    1. / 2 * np.log(1. + delta) - 1. / 2 * (1. - xh2 - xh) * (1. + delta) ** -1 * ddelta_dxh + \
                    beta * dgamma_dxh * in_square_brackets + \
                    beta * gamma * ((1. + 1. / beta / gamma) ** -1 * (-1.) / beta / gamma ** 2 * dgamma_dxh)) # eq. (A2)
        dss_dxhe = (1. + xh + 3 * xh2) ** -1 * ( \
                    (1. + beta * gamma) ** -1 * beta * dgamma_dxhe - 1. / 2 * (1. - xh2 - xh) * (1. + delta) ** -1 * ddelta_dxhe + \
                    beta * dgamma_dxhe * in_square_brackets + beta * gamma * ( \
                    (1. + 1. / beta / gamma) ** -1 * (-1.) / beta / gamma ** 2 * dgamma_dxhe + 2. / 3 * np.log(1. + 1. / delta) - \
                    1. / 3 * (2. - 2 * xhe - xhep) * (1. + 1. / delta) ** -1 * (-1.) * delta ** -2 * ddelta_dxhe)) # eq. (A3)
        dss_dxhep = (1. + xh + 3 * xh2) ** -1 * ( \
                    (1. + beta * gamma) ** -1 * beta * dgamma_dxhep - 1. / 2 * (1. - xh2 - xh) * (1. + delta) ** -1 * ddelta_dxhep + \
                    beta * dgamma_dxhep * in_square_brackets + \
                    beta * gamma * ((1. + 1. / beta / gamma) ** -1 * (-1.) / beta / gamma ** 2 * dgamma_dxhep + \
                    1. / 3 * np.log(1. + 1. / delta) - 1. / 3 * (2. - 2 * xhe - xhep) * (1. + 1. / delta) ** -1 * (-1.) / delta ** 2 * ddelta_dxhep))
                    
        dsmix_dxh2 = dss_dxh2 * smix_prefactor
        dsmix_dxh = dss_dxh * smix_prefactor
        dsmix_dxhe = dss_dxhe * smix_prefactor
        dsmix_dxhep = dss_dxhep * smix_prefactor
        
        dsmix_dlogt = dsmix_dxh2 * dxh2_dlogt + dsmix_dxh * dxh_dlogt + dsmix_dxhe * dxhe_dlogt + dsmix_dxhep * dxhep_dlogt
        dsmix_dlogp = dsmix_dxh2 * dxh2_dlogp + dsmix_dxh * dxh_dlogp + dsmix_dxhe * dxhe_dlogp + dsmix_dxhep * dxhep_dlogp
        
        dlogsmix_dlogt = dsmix_dlogt / smix
        dlogsmix_dlogp = dsmix_dlogp / smix
        
        res['st'] = (1. - y) * s_h / s * res_h['st'] + y * s_he / s * res_he['st'] + smix / s * dlogsmix_dlogt
        res['sp'] = (1. - y) * s_h / s * res_h['sp'] + y * s_he / s * res_he['sp'] + smix / s * dlogsmix_dlogp
        
        res['grada'] = -1. * res['sp'] / res['st']

        logp, logt = pair
        dpdt_const_rho = - 10 ** logp / 10 ** logt * res['rhot'] / res['rhop']
        dudt_const_rho = s * (res['st'] - res['sp'] * res['rhot'] / res['rhop'])
        dpdu_const_rho = dpdt_const_rho / 10 ** res['logrho'] / dudt_const_rho
        gamma3 = 1. + dpdu_const_rho # cox and giuli 9.93a
        gamma1 = (gamma3 - 1.) / res['grada']
        res['gamma3'] = gamma3
        res['gamma1'] = gamma1
        res['chirho'] = res['rhop'] ** -1 # rhop = dlogrho/dlogp|t
        res['chit'] = dpdt_const_rho * 10 ** logt / 10 ** logp
        
        # from mesa's scvh in mesa/eos/eosPT_builder/src/scvh_eval.f
        # 1005:      Cv = chiT * P / (rho * T * (gamma3 - 1)) ! C&G 9.93
        # 1006:      Cp = Cv + P * chiT**2 / (Rho * T * chiRho) ! C&G 9.86
        res['cv'] = res['chit'] * 10 ** logp / (10 ** res['logrho'] * 10 ** logt * (gamma3 - 1.)) # erg g^-1 K^-1
        res['cp'] = res['cv'] + 10 ** logp * res['chit'] ** 2 / (10 ** res['logrho'] * 10 ** logt * res['chirho']) # erg g^-1 K^-1
                        
        return res
    
    def plot_pt_coverage(self, ax, symbol, **kwargs):
        for logt in self.logtvals:
            logp = self.h_data[logt]['logp']
            ax.plot(10 ** logp, np.ones_like(logp) * 10 ** logt, symbol, **kwargs)
    
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(3e-2, 5e19)
        ax.set_ylim(ymin=3e1)
        ax.set_xlabel(r'$P$')
        ax.set_ylabel(r'$T$')
        
    def plot_rhot_coverage(self, ax, symbol, **kwargs):
        for logt in self.logtvals:
            logrho = self.h_data[logt]['logrho']
            ax.plot(10 ** logrho, np.ones_like(logrho) * 10 ** logt, symbol, **kwargs)
    
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_xlim(3e-2, 5e19)
        ax.set_ylim(ymin=3e1)
        ax.set_xlabel(r'$\rho$')
        ax.set_ylabel(r'$T$')
