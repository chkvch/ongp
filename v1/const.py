import numpy as np

pi = np.pi
one = 1
radians_per_cycle = pi * 2
degrees_per_cycle = 360.
degrees_per_radian = degrees_per_cycle / radians_per_cycle
radians_per_degree = radians_per_cycle / degrees_per_cycle
deg_per_day = 180. / np.pi * 60 * 60 * 24

msun = 1.9892e33
rsun = 6.9598e10
lsun = 3.8418e33

mearth = 5.9764e27
rearth = 6.37e8

mj = mjup = 1.8986112e30 # (15) ! Guillot+2003 Table 3.1
dmj = dmjup = 0.0000015e30

# Seidelmann et al. 2007
rj = rjup = 6.9911e9 # volumetric mean radius
drjup = 6e5
drjup_eq = 4e5
rj_eq = rjup_eq = 7.1492e9
rj_pol = rjup_pol = 6.6854e9
drjup_pol = 10e5

ms = msat = 568.34e27
rs = rsat = 58232e5 # volumetric
drsat = 6e5
rs_eq = rsat_eq = 60268e5 # equatorial
drsat_eq = 4e5
rs_pol = rsat_pol = 54364e5 # polar
drsat_pol = 10e5

cgrav = 6.67408e-8 # this value is NIST / codata reccommended 2014. mesa has 6.67428.
h = 6.62606896e-27
hbar = h / 2 / np.pi
qe = 4.80320440e-10

c = clight = 2.99792458e10
k = kb = 1.3806504e-16
kb_ev = 8.617385e-5
avogadro = 6.02214179e23
rgas = cgas = k * avogadro

amu = 1.660538782e-24
mn = 1.6749286e-24 # neutron mass (g)
mp = 1.6726231e-24
me = 9.10938291e-28
mh = 1.00794 * amu
mhe = 4.002602 * amu
boltz_sigma = sigma_sb = 5.670400e-5
arad = crad = boltz_sigma * 4 / c
au = 1.495978921e13

ev = 1.602176487e-12
ry = ryd = rydberg = 13.605698140 * ev

secyear = secyer = seconds_per_year = 3.15569e7
secday = seconds_per_day = 86400

neptune_semimajor_axis = 19.2 * au
uranus_semimajor_axis = 30.1 * au
neptune_incident_flux = lsun / 4. / np.pi / neptune_semimajor_axis ** 2
uranus_incident_flux = lsun / 4. / np.pi / uranus_semimajor_axis ** 2

uranus_mass = mura = 86.813e27
uranus_gm = 5.794e6 * 1e15 # G*M. conversion is for km^3 to cm^3
uranus_req = 25559e5
uranus_rpol = 24973e5
uranus_rvol = 25362e5
uranus_j2 = 3343.43e-6
uranus_rotation_period = 17.24 * 60 * 60
uranus_omega_rot = 2. * np.pi / uranus_rotation_period
uranus_omega_dyn = np.sqrt(uranus_gm / uranus_rvol ** 3)

neptune_mass = mnep = 102.413e27
neptune_gm = 6.8351e6 * 1e15
neptune_req = 24764e5
neptune_rpol = 24341e5
neptune_rvol = 24622e5
neptune_j2 = 3411.e-6
neptune_rotation_period = 16.11 * 60 * 60
neptune_omega_rot = 2. * np.pi / neptune_rotation_period
neptune_omega_dyn = np.sqrt(neptune_gm / neptune_rvol ** 3)

jupiter_gm = 126.687e6 * 1e15
jupiter_rvol = 69911e5
jupiter_rotation_period = 9.9259 * 60 * 60
jupiter_omega_rot = 2. * np.pi / jupiter_rotation_period
jupiter_omega_dyn = np.sqrt(jupiter_gm / jupiter_rvol ** 3)

saturn_gm = 37.931e6 * 1e15
saturn_rvol = 58232e5
saturn_rotation_period = 10.656 * 60 * 60 # lol
saturn_omega_rot = 2. * np.pi / saturn_rotation_period
saturn_omega_dyn = np.sqrt(saturn_gm / saturn_rvol ** 3)
