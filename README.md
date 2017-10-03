# An environment for making one-dimensional static and evolutionary models of giant planets. 
The code is adapted from Daniel P. Thorngren's evolutionary program evolve.py, but extended to include a more general equation of state for H-He-Z mixtures, a H-He phase diagram for the purposes of modelling H-He phase separation, and model atmospheres specific to Jupiter and Saturn. The code depends on data (equations of state, etc.) from various other researchers that I'm not at liberty to distribute, but if you'd like to use the code contact me to be put in touch.

### Getting started
A simple instantiation of the ongp.evol class and calculation of a static model in Python 2.7 might look like

```python
import ongp
e = ongp.evol()
e.static(mtot='jup', yenv=0.275, zenv=0.1, mcore=10., t1=169.)
print 'Rtot = %10.5g cm' % e.rtot
print 'Teff = %3.2f K' % e.teff
```

An evolutionary model could similarly be computed as
```python
history = e.run(mtot='jup', yenv=0.275, zenv=0.1, mcore=10., starting_t1=2e3, min_t1=160.)
```

