from importlib import reload
import defaults; reload(defaults)
import ongp; reload(ongp)
import time

planet = 'jup'

evol_params = defaults.params[planet]['evol']
evolve_params = defaults.params[planet]['evolve']
mesh_params = defaults.mesh_params # empty at present

evol_params['atm_jupiter_modified_teq'] = 102.5
# evol_params['max_iters_static_before_rain'] = 8

# "evolve params" even though we're only calling static, reflecting the way that evolve
# passes its whole params dict each time it calls static
evolve_params['t1'] = 163.6 # 165.
evolve_params['transition_pressure'] = 2.
evolve_params['z1'] = 0.02
evolve_params['mcore'] = 5.
evolve_params['rrho_where_have_helium_gradient'] = 1e-5
evolve_params['rainout_verbosity'] = 2
evolve_params['phase_t_offset'] = -1250.

evolve_params['debug_iterations'] = True


if False: # ignore helium rain
    evol_params.pop('hhe_phase_diagram', None)
    evolve_params.pop('rrho_where_have_helium_gradient', None)

e = ongp.evol(evol_params, mesh_params)

t0 = time.time()
e.static(evolve_params)

print('{:>14s} {:>14s} {:>14s} {:>14s} {:>14s} {:>10s}'.format('rtot', 'teff', 't10', 'y1', 'lint', 'et_ms'))
print('{:>14e} {:>14e} {:>14e} {:>14e} {:>14e} {:>10f}'.format(e.rtot, e.teff, e.t10, e.y[-1], e.lint, (time.time() - t0)*1e3))
print()
for p, t in zip(e.p[-10:], e.t[-10:]):
    print('{:>14e} {:>14e}'.format(p, t))
print('{} zones outside 10 bars'.format(len(e.p[e.p<1e7])))

with open('profile.dat', 'w') as f:
    for p, t, y in zip(e.p, e.t, e.y):
        f.write('{:14g} {:14g} {:14g}\n'.format(p, t, y))
    print('wrote profile.dat')
