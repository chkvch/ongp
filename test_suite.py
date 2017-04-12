import evolve
reload(evolve)
import matplotlib.pyplot as plt
import const
import time

def static_jupiter_and_saturn():
    
    print 'test static_jupiter_and_saturn'
    fmt = '%20s %20s %20s %20s %20s'
    print fmt % ('case', 'iters', 'walltime (s)', 'rtot', 'teff')
    t0 = time.time()
    jup_homog = evolve.Evolver()
    jup_homog.staticModel(mtot=1., yenv=0.27, zenv=0.0465, mcore=10., t10=300.)
    print fmt % ('jup_homog', jup_homog.iters, time.time() - t0, jup_homog.rtot, jup_homog.teff)

    t0 = time.time()
    jup_rain = evolve.Evolver(phase_t_offset=-700.)
    jup_rain.staticModel(mtot=1., yenv=0.27, zenv=0.05, mcore=10., t10=308., include_he_immiscibility=True)
    print fmt % ('jup_rain', jup_rain.iters, time.time() - t0, jup_rain.rtot, jup_rain.teff)
    
    t0 = time.time()
    sat_homog = evolve.Evolver()
    sat_homog.staticModel(mtot=const.msat/const.mjup, yenv=0.27, zenv=0.05, mcore=20., t10=200)
    print fmt % ('sat_homog', sat_homog.iters, time.time() - t0, sat_homog.rtot, sat_homog.teff)
    
    t0 = time.time()
    sat_rain = evolve.Evolver(phase_t_offset=-700.)
    sat_rain.staticModel(mtot=const.msat/const.mjup, yenv=0.27, zenv=0.05, mcore=20., t10=200., include_he_immiscibility=True)
    print fmt % ('sat_rain', sat_rain.iters, time.time() - t0, sat_rain.rtot, sat_rain.teff)
    
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 4), gridspec_kw={'wspace':0.4})

    ax[0].loglog(jup_homog.p, jup_homog.t)
    ax[0].loglog(sat_homog.p, sat_homog.t)
    # ax[0].loglog(jup_homog_no_core.p, jup_homog_no_core.t)
    ax[0].loglog(jup_rain.p, jup_rain.t)
    ax[0].loglog(sat_rain.p, sat_rain.t)
    ax[0].set_xlabel(r'$P\ \ (\rm dyne\ cm^{-2})$')
    ax[0].set_ylabel(r'$T\ \ (\rm K)$')
    
    ax[1].loglog(jup_homog.p, jup_homog.rho)
    ax[1].loglog(sat_homog.p, sat_homog.rho)
    # ax[1].loglog(jup_homog_no_core.p, jup_homog_no_core.rho)
    ax[1].loglog(jup_rain.p, jup_rain.rho)
    ax[0].loglog(sat_rain.p, sat_rain.t)
    ax[1].set_xlabel(r'$P\ \ (\rm dyne\ cm^{-2})$')
    ax[1].set_ylabel(r'$\rho\ \ (\rm g\ cm^{-3})$')
    
    ax[2].plot(jup_rain.rf, jup_rain.y)
    ax[2].plot(sat_rain.rf, sat_rain.y)
    ax[2].set_ylabel(r'$Y$')
    ax[2].set_xlabel(r'$r/R$')
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 4), gridspec_kw={'wspace':0.4})
    
    try:
        ax[0].semilogy(jup_rain.rf, jup_rain.brunt_b)
        ax[0].semilogy(sat_rain.rf, sat_rain.brunt_b)
    except ValueError:
        '''no positive brunt_b.'''
        pass
        # raise ValueError('no positive brunt_b in the helium rain models.')
    ax[0].set_xlabel(r'$r/R$')
    ax[0].set_ylabel(r'$B$')
    
    ax[1].semilogy(jup_rain.rf, jup_rain.brunt_n2)
    ax[1].semilogy(sat_rain.rf, sat_rain.brunt_n2)
    ax[1].set_xlabel(r'$r/R$')
    ax[1].set_ylabel(r'$N^2\ \ (\rm s^{-2})$')
    ax[1].set_ylim(1e-10, 1e-3)
    
    ax[2].semilogy(jup_rain.rf, jup_rain.lamb_s12)
    ax[2].semilogy(sat_rain.rf, sat_rain.lamb_s12)
    ax[2].set_xlabel(r'$r/R$')
    ax[2].set_ylabel(r'$S_{\ell=1}^2\ \ {\rm s^{-2}}$')
    ax[2].set_ylim(1e-10, 1e-3)
    
def evolve_jupiter():
    
    print 'test evolve_jupiter'
        
    t0 = time.time()
    # must use aneos for evolutionary models since REOS water does not cover high T.
    jup_homog = evolve.Evolver(z_eos_option='aneos ice')
    try:
        jup_homog.run(mtot=1., yenv=0.27, zenv=0.05, mcore=10., starting_t10=1e3, min_t10=300., nsteps=100., 
                        stdout_interval=10,  output_prefix='test_suite/jup_homog')
    except:
        print 'case jup_homog failed.'
        print
        pass
    print

    t0 = time.time()
    jup_rain = evolve.Evolver(z_eos_option='aneos ice', phase_t_offset=-700.)
    try:
        jup_rain.run(mtot=1., yenv=0.27, zenv=0.05, mcore=10., starting_t10=1e3, min_t10=300., nsteps=100., include_he_immiscibility=True,
                        stdout_interval=10,  output_prefix='test_suite/jup_rain')
    except:
        print 'case jup_rain failed.'
        print
        pass
    print
    
def evolve_saturn():
    
    print 'test evolve_saturn'
        
    t0 = time.time()
    sat_homog = evolve.Evolver(z_eos_option='aneos ice')
    try:
        sat_homog.run(mtot=const.msat/const.mjup, yenv=0.27, zenv=0.05, mcore=20., starting_t10=1e3, min_t10=200., nsteps=100., 
                        stdout_interval=10,  output_prefix='test_suite/sat_homog')
    except:
        print 'case sat_homog failed.'
        print
        pass
    print

    t0 = time.time()
    sat_rain = evolve.Evolver(z_eos_option='aneos ice', phase_t_offset=-700.)
    try:
        sat_rain.run(mtot=const.msat/const.mjup, yenv=0.27, zenv=0.05, mcore=20., starting_t10=1e3, min_t10=200., nsteps=100., include_he_immiscibility=True,
                        stdout_interval=10,  output_prefix='test_suite/sat_rain')
    except:
        print 'case sat_rain failed.'
        print
        pass
    print
        