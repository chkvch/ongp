# An environment for making one-dimensional static and evolutionary models of giant planets. 
The code is adapted from Daniel P. Thorngren's evolutionary program evolve.py [2016ApJ...831...64T], but extended to include a more general equation of state for H-He-Z mixtures, a H-He phase diagram for the purposes of modelling H-He phase separation, and model atmospheres specific to Jupiter and Saturn. The code depends on data (equations of state, etc.) from various other researchers that I'm not at liberty to distribute, but if you'd like to use the code contact me to be put in touch.

### Getting started
A basic usage pattern might look like

```python
import ongp

evol_params = {
    'hhe_eos_option':'scvh',
    'z_eos_option':'reos water',
    'atm_option':'f11_tables jup', 
    'path_to_data':'/Users/chris/Dropbox/planet_models/ongp/data'
}
e = ongp.evol(evol_params)

static_params = {
    'mtot':'jup', 
    't1':165., 
    'z1':0.07, 
    'z2':0.1, 
    'y1':0.265, 
    'y2':0.280, 
    'transition_pressure':3.35, 
    'mcore':3.25
}

e.static(static_params)
print 'Rtot = %10.5g cm' % e.rtot
print 'Teff = %3.2f K' % e.teff
```

An evolutionary model could then be computed as
```python
evolve_params = {
    'mtot':'jup',
    'start_t':2e3,
    'end_t':160.,
    'which_t':'t1',
    'y1':0.265,
    'y2':0.280,
    'z1':0.07,
    'z2':0.1,
    'mcore':3.25,
    'transition_pressure':3.35,
    'stdout_interval':5
}
e.evolve(evolve_params)
```

