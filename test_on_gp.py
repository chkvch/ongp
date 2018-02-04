# Tests for on_GP
import ongp
import f11_atm
import const
import numpy as np


#def test_uranus_0():
#    f = ongp.evol(z_eos_option='reos water', atm_option='f11_tables u')
#    f.static(mtot='u', mcore=12., zenv=0.1, t10=150.)

def test_uranus_1():
    f = ongp.evol(z_eos_option='reos water', atm_option='f11_tables u')
    f.static(mtot='u', mcore=13., zenv=0.1, t10=150.)

def test_uranus_2():
    f = ongp.evol(z_eos_option='reos water', atm_option='f11_tables u')
    f.static(mtot='u', mcore=13., zenv=0.1, t10=840.)
