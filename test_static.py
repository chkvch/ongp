from importlib import reload
import defaults; reload(defaults)
import ongp; reload(ongp)

evol_params = defaults.params['sat']['evol']
evolve_params = defaults.params['sat']['evolve']
mesh_params = defaults.mesh_params

evolve_params['t1'] = 160.
evolve_params['transition_pressure'] = 2.
evolve_params['z1'] = 0.02
evolve_params['mcore'] = 10.
evolve_params['rainout_verbosity'] = 2

e = ongp.evol(evol_params, **mesh_params)
e.static(evolve_params)
