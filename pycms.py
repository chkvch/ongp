import matlab.engine
import numpy as np
from utils import hms

# for reference, check out
# https://www.mathworks.com/help/matlab/matlab-engine-for-python.html

def downsample_density_profile(r, rho, nz):
    from scipy.interpolate import splrep, splev
    tck = splrep(r, rho, k=1) # linear interpolant for rho(r)
        
    # evaluate density at a linear grid in mean radius. could do more complex grid choices if desired.
    ri = max(r) * np.arange(nz + 1.)[1::] / nz # layer mean radii # center point non-zero, surface point -> Rtot
    rhoi = splev(ri, tck)    
    
    return ri, rhoi
    
def downsample_from_model(ev, nz):
    assert not np.any(ev.r[1:] == ev.r[:-1]), 'duplicate radii in model; spline representation will not work.'
    from scipy.interpolate import splrep, splev
    tck = splrep(ev.r, ev.rho, k=1)
    
    si = np.zeros(nz)
    rhoi = np.zeros(nz)
    
    points_in_core = int(np.floor(1. / 3 * nz))
    points_in_env = int(np.ceil(2. / 3 * nz))
    
    si[:points_in_core] = ev.r[ev.kcore-1] * (1 + np.arange(points_in_core)) / points_in_core
    rhoi[:points_in_core] = splev(si[:points_in_core], tck)    
    
    if not ev.zenv_inner:
        si[-points_in_env:] = ev.r[ev.kcore] + (ev.r[-1] - ev.r[ev.kcore]) * (1 + np.arange(points_in_env)) / points_in_env
        rhoi[-points_in_env:] = splev(si[-points_in_env:], tck)
    else: # two-layer envelope; ensure density is sampled at the interface
        points_in_inner_env = int(np.floor(1. / 2 * points_in_env))
        points_in_outer_env = int(np.ceil(1. / 2 * points_in_env))
        
        si[-points_in_env:-points_in_outer_env] = ev.r[ev.kcore] + (ev.r[ev.ktrans-1] - ev.r[ev.kcore]) * (1. + np.arange(points_in_inner_env)) / points_in_inner_env
        rhoi[-points_in_env:-points_in_outer_env] = splev(si[-points_in_env:-points_in_outer_env], tck)
        
        si[-points_in_outer_env:] = ev.r[ev.ktrans] + (ev.r[-1] - ev.r[ev.ktrans]) * (1 + np.arange(points_in_outer_env)) / points_in_outer_env
        rhoi[-points_in_outer_env:] = splev(si[-points_in_outer_env:], tck)
        
    assert not np.any(np.isnan(rhoi)), 'downsample got nans in rhoi.'
    
    return si, rhoi
    

import saturn_data, const
qrot_cassini = saturn_data.cassini_omega ** 2 * saturn_data.rsat_eq_j06 ** 3 / saturn_data.gmsat_j06 # q = omega_sat^2 * a0_sat^3 / G / m_sat;
qrot_voyager = saturn_data.voyager_omega ** 2 * saturn_data.rsat_eq_j06 ** 3 / saturn_data.gmsat_j06 # q = omega_sat^2 * a0_sat^3 / G / m_sat;

qrots = {'cassini':qrot_cassini, 'voyager':qrot_voyager}

def run_cms_for_model(ev, nz, rot=None, output_folder=None, relax_qrot=True):
    """make a CMSPlanet instance with nz zones in a new or existing matlab engine, and iterate to find an oblate model in hydrostatic equilibrium
    with layer mean radii and densities matching those of the 1d input model.
    inputs: model ev (an instance of class evolve.Evolver), layer count nz (integer), rotation rate (one of 'voyager' or 'cassini'), 
            output_folder (destination for output), relax_qrot (logical, whether qrot should be adjusted during iterations or left alone)
    outputs: matlab engine e (an instance of matlab.engine.matlabenging.MatlabEngine) and cmp (a CMSPlanet instance)"""
    try:
        e.quit()
    except:
        pass

    if matlab.engine.find_matlab():
        # if want to find existing engine, run matlab.engine.shareEngine in matlab
        print 'connecting to one of existing matlab instances', matlab.engine.find_matlab()    
        e = matlab.engine.connect_matlab()
    else:
        print 'no matlab instance found. if you want to connect to an existing matlab engine, run'
        print 'matlab.engine.shareEngine in matlab.'
        ans = raw_input('do you want to start a new matlab engine? (takes a minute or two) [y/n]')
        if ans == 'y' or ans == 'yes':
            e = matlab.engine.start_matlab()
        else:
            assert False, 'quit'

    assert type(e) is matlab.engine.matlabengine.MatlabEngine, 'something went wrong initializing the matlab engine.'

    e.cd('/Users/chris/Dropbox/cms/CMS-planet/') # work from where the scripts are
    e.setws(nargout=0) # expects an output argument, will throw an error if there isn't one

    mtot = float(ev.mtot) # need python float, not np.float64
    a0 = 60330e3
    if not rot:
        print 'no rotation rate specified; defaulting to Cassini value'
        rot = 'cassini'
    qrot = qrots[rot]

    # make a new CMSPlanet instance with a handle in python
    cmp = e.CMSPlanet(float(nz), 'djtol', 9e-8, 'verb', 0) # note the float
    e.workspace['cmp'] = cmp

    e.setfield(cmp, 'M', mtot)
    e.setfield(cmp, 'a0', a0)
    e.setfield(cmp, 'qrot', qrot)

    # downsample the ndarrays to ones of length nz. so far do this with linear interpolation onto a grid linear in model radius.
    print 'downsampling %i-zone model to %i cms zones' % (len(ev.r), nz)
    # si, rhoi = downsample_density_profile(ev.r, ev.rho, nz)
    si, rhoi = downsample_from_model(ev, nz)
    if output_folder:
        import os
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        np.savetxt('%s/profile_nz%i.data' % (output_folder, nz), np.array([si, rhoi]))

    # CMSPlanet expects mks
    si *= 1e-2 # 1 cm = 10^-2 m
    rhoi *= 1e3 # 1 g cm^-3 = 10^3 kg m^-3

    # CMSPlanet wants descending radii
    si = si[::-1]
    rhoi = rhoi[::-1]
    # print '%10s' % 'si', ('%10g ' * nz) % tuple(si)
    # print '%10s' % 'rhoi', ('%10g ' * nz) % tuple(rhoi)
    # print

    # pretend si are equatorial radii for the initial relaxation
    ai = matlab.double(list(si))
    lambdai = matlab.double(list(si / max(si)))

    # these are model truth to be matched through iterations
    si = matlab.double(list(si))
    rhoi = matlab.double(list(rhoi))

    si.reshape((nz, 1))
    rhoi.reshape((nz, 1))
    ai.reshape((nz, 1))
    lambdai.reshape((nz, 1))

    e.setfield(cmp, 'ai', si)
    e.setfield(cmp, 'rhoi', rhoi)

    import time
    t0 = time.time()
    header_line = '%5s %6s %12s %12s %12s %14s %14s %14s %14s %10s %10s %10s %10s' % ('i', 'ET', 'j2', 'j4', 'j6', 'rms_r_err', 'dj2_sigma_j2', 'dj4_sigma_j4', 'dj6_sigma_j6', 'a0', 'b0', 's0', 'qrot')
    print header_line
    max_iters = 30
    rms_mean_radius_tolerance = 1e-10
    dj2_tolerance = saturn_data.j2sat_j06[1] / 10. # 0th element is j2 itself
    verb = 2
    if output_folder:
        if rot == 'cassini':
            output_path = '%s/cms_nz%i.data' % (output_folder, nz)
        elif rot == 'voyager':
            output_path = '%s/cms_nz%i_voyager.data' % (output_folder, nz)
        else:
            raise ValueError('rot must be one of either cassini or voyager')
        fw = open(output_path, 'w')
        fw.write('%s\n' % header_line)
    for i in np.arange(max_iters):
        old_j2 = e.eval('cmp.J2')
        old_j4 = e.eval('cmp.J4')
        old_j6 = e.eval('cmp.J6')
        
        et = e.relax_to_HE(cmp)
        # print iter, time, resulting Js, rms error
        x = (si - np.array(e.eval('cmp.si'))) / e.eval('cmp.s0') # relative mean radius error as np.ndarray
        rms_x = np.sqrt(np.mean(x ** 2))
        dj2 = e.eval('cmp.J2') - old_j2
        dj4 = e.eval('cmp.J4') - old_j4
        dj6 = e.eval('cmp.J6') - old_j6
        output_line = '%5i %6.2f %12.8f %12.8f %12.8f %14g %14g %14g %14g %10.1f %10.1f %10.1f %10.6f' % (i, et, e.eval('cmp.J2'), e.eval('cmp.J4'), e.eval('cmp.J6'), rms_x, \
            dj2 / saturn_data.j2sat_j06[1], dj4 / saturn_data.j4sat_j06[1], dj6 / saturn_data.j6sat_j06[1], e.eval('cmp.a0'), e.eval('cmp.b0'), e.eval('cmp.s0'), e.eval('cmp.qrot'))
        print output_line
        if output_folder: fw.write('%s\n' % output_line)
        
        if rms_x < rms_mean_radius_tolerance: 
            if verb >= 2: print 'satisfied rms mean radius error tolerance %g < %g after %s' % (rms_x, rms_mean_radius_tolerance, hms(time.time() - t0))
            break
        if abs(dj2) <= dj2_tolerance:
            if verb >= 2: print 'satisfied delta_j2 tolerance |%g| < %g after %s' % (dj2, dj2_tolerance, hms(time.time() - t0))
            break
    
        new_ai = np.array(e.eval('cmp.ai')) + x * e.eval('cmp.s0')
        # acrobatics to pass new ai to cms as column vector
        new_ai.reshape((nz, 1))
        new_ai_as_list_of_lists = [[x] for x in list(new_ai.flatten())]    
        new_ai = matlab.double(new_ai_as_list_of_lists)            
        
        e.setfield(cmp, 'ai', new_ai)

        if relax_qrot:
            if rot is 'cassini':
                new_qrot = saturn_data.cassini_omega ** 2 * (e.eval('cmp.a0') * 1e2) ** 3 / saturn_data.gmsat_j06
            elif rot is 'voyager':
                new_qrot = saturn_data.voyager_omega ** 2 * (e.eval('cmp.a0') * 1e2) ** 3 / saturn_data.gmsat_j06
            else:
                raise ValueError('rot must be one of cassini, voyager')
            e.setfield(cmp, 'qrot', new_qrot)
    
    if output_folder:
        print 'wrote progress to %s' % output_path
        fw.close()
    print
    return e, cmp
    