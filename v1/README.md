# An environment for making one-dimensional static and evolutionary models of giant planets.
The code is adapted from Daniel P. Thorngren's evolutionary program evolve.py [2016ApJ...831...64T], but extended to include a more general equation of state for H-He-Z mixtures, a H-He phase diagram for the purposes of modelling H-He phase separation, and model atmospheres specific to Jupiter and Saturn. The code depends on data (equations of state, etc.) from various other researchers that I'm not at liberty to distribute, but if you'd like to use the code contact me to be put in touch.

### Getting started
Set up an `ongp.evol` object with
```
python
import ongp

evol_params = {
    'hhe_eos_option':'mh13_scvh',
    'z_eos_option':'reos water',
    'atm_option':'f11_tables',
    'atm_planet':'jup',
    'path_to_data':'./data',
    'max_iters_static':100,
}
e = ongp.evol(evol_params)
```
and test a static model:
```
static_params = {
    'mtot':'jup',
    'model_type':'three_layer',
    't1':165.,
    'z1':0.07,
    'z2':0.1,
    'y1':0.265,
    'y2':0.280,
    'transition_pressure':3.35,
    'mcore':3.25
}

e.static(static_params)
print ('Rtot = %10.5g cm' % e.rtot)
print ('Teff = %3.2f K' % e.teff)
import matplotlib.pyplot as plt
plt.loglog(e.p, e.rho)
```
An evolutionary model could then be computed as
```
evolve_params = {
    'mtot':'jup',
    'model_type':'three_layer',
    'start_t':1e3,
    'end_t':160.,
    'which_t':'t1',
    'y1':0.265,
    'y2':0.280,
    'z1':0.07,
    'z2':0.1,
    'mcore':3.25,
    'transition_pressure':3.35,
    'stdout_interval':1,
    'max_timestep':1e9,
    'target_timestep':1e7,
}
e.evolve(evolve_params)
```
