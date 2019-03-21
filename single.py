# this is derived from ~/archive_planet_models/gravity/tof/single_master/single.py
from importlib import reload
import tof; reload(tof)

import numpy as np
import const
import saturn_data
q_sat = saturn_data.voyager_omega ** 2 * saturn_data.rsat_eq_n13 ** 3 / saturn_data.gmsat_j06
m_sat = saturn_data.voyager_omega ** 2 * saturn_data.r_vol ** 3 / saturn_data.gmsat_j06

params = {
            'nz':4096,
            'small':m_sat,
            'model_type':'evol',
            'evol_outer_method':'adjust_mcore',
            'max_iters_inner':3,
            'max_iters_outer':150,
            'mtot':const.msat,
            'req':60268e5,
            't1':140.,
            'ym':0.275,
            'z_eos_option':'reos water',
            'atm_option':'f11_tables sat',
            'j2n_rtol':1e-4,
            'mtot_rtol':1e-4,
            'verbosity':1,
            'evol_verbosity':2
         }
mesh_params = {
            'mesh_func_type':'flat_with_surface_exponential_core_gaussian',
            'amplitude_core_mesh_boost':5e0,
            'width_core_mesh_boost':1.5e-1,
            'fmean_core_bdy_mesh_boost':2e-1,
}

# params['path_to_eos_data'] = '/pfs/home/cmankovich/saturn_gravity/ongp/data'
params['method_for_aa2n_solve'] = 'cubic 32'
params['mcore'] = 16.

def run(z1, z2, y1, ptrans):
    params['z1'] = z1
    params['z2'] = z2
    params['y1'] = y1
    params['ptrans'] = ptrans
    t = tof.run_one_tof4(params, mesh_params)
    return t
