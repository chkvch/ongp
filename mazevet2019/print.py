import numpy as np
data = np.load('table.npy').item()
npts = len(data['p'])
for ip in np.arange(npts):
    for it in np.arange(npts):
        numbers = data['p'][ip], data['t'][it], data['rho'][ip, it], data['u'][ip, it], data['chirho'][ip, it], data['chit'][ip, it]
        fmt = '{:10.3e} ' * len(numbers)
        print(fmt.format(*numbers))