import numpy as np
import time
import numpy as np
from scipy.optimize import root, brentq
'''
   H2OFIT : Given the mass density RHO(g/cm^3^) and temperature T(K),
             returns the fits to the thermodynamic functions:
             PnkT - pressure, normalized to (kT times number density)
             FNkT - Helmholtz free energy per atom, normalized by kT
             UNkT - internal energy per atom, normalized by kT
             CV - specific heat per atom in units of Boltzmann constant k
             CHIT - logarithmic derivative of pressure over temperature
             CHIR - logarithmic derivative of pressure over density
             PMbar - pressure in Mbar
             USPEC - specific internal energy in ergs/gram
'''
   
names = 'rho', 't', 'p', 'p_nkT', 'f_NkT', 'u_NkT', 'cv_Nk', 'chit', 'chirho', 's_Nk', 'u'
import subprocess
for rhoval in 1e-3, 1e0, 1e3:
    for tval in 1e2, 1e3, 1e4, 1e5:
        p = subprocess.run(['./eos', str(rhoval), str(tval)], capture_output=True)
        # print(p.stdout)
        data = p.stdout.split()
        # print(np.float64(data[0]))

def zero_me(logrhoval, tval, pval):
    rhoval = 10 ** logrhoval
    p = subprocess.run(['./eos', str(rhoval), str(tval)], capture_output=True)
    try:
        p_out = np.float64(p.stdout.split()[2])
    except IndexError:
        raise ValueError('got bad result from eos')
    return p_out - pval
        
npts = 100
results = {}
results['rho'] = np.zeros((npts, npts))
results['u'] = np.zeros((npts, npts))
results['chit'] = np.zeros((npts, npts))
results['chirho'] = np.zeros((npts, npts))
results['p'] = np.logspace(-3, 5, npts)
results['t'] = np.logspace(2, 6, npts)

t0 = time.time()
for ip, pval in enumerate(results['p']): # 1e-3 to 1e3 GPa -> 10 bar to 10^9 bar -> 1e7 to 1e15 cgs
    et = time.time() - t0
    rate = (1. + ip) / et
    eta = (npts - ip) / rate
    print('\r{:3n}/{:3n}, p={:8.2e}, {:5.1f} s elapsed, {:5.1f} s remain'.format(ip, npts, pval, et, eta), end='')
    for it, tval in enumerate(results['t']):
        x = brentq(zero_me, -8, 2, args=(tval, pval))
        rho = 10. ** x
        results['rho'][ip, it] = rho

        # one more call to evaluate other quantities of interest at our rho value
        p = subprocess.run(['./eos', str(rho), str(tval)], capture_output=True)
        results['u'][ip, it] = np.float64(p.stdout.split()[10])
        results['chit'][ip, it] = np.float64(p.stdout.split()[7])
        results['chirho'][ip, it] = np.float64(p.stdout.split()[8])        
        
np.save('table', results)