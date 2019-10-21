import driver
from importlib import reload
reload(driver)

print('homogeneous saturn')
params = {'mcore':10, 'z1':0.02, 'start_t':1e3}
driver.run('sat', params, homog=True, dump=False, custom_evol_params={}, custom_mesh_params={})
print()

reload(driver)
print('adiabatic saturn rainout')
params.pop('rrho_where_have_helium_gradient', None)
params['phase_t_offset'] = 0
driver.run('sat', params, homog=False, dump=False)
print()

reload(driver)
print('superadiabatic saturn rainout')
params['rrho_where_have_helium_gradient'] = 3e-2
driver.run('sat', params, homog=False, dump=False)
