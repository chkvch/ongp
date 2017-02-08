import evolve
reload(evolve)
import matplotlib.pyplot as plt
import const

def static_jupiter_and_saturn():
    
    jup_homog = evolve.Evolver()
    jup_homog.staticModel(mtot=1., yenv=0.27, zenv=0.0465, mcore=10., t10=300.)

    # jup_homog_no_core = evolve.Evolver()
    # jup_homog_no_core.staticModel(mtot=1., yenv=0.27, zenv=0.07, mcore=0., t10=300.)

    jup_rain = evolve.Evolver(phase_t_offset=-370.)
    jup_rain.staticModel(mtot=1., yenv=0.27, zenv=0.05, mcore=10., t10=308., include_he_immiscibility=True)
    
    sat_homog = evolve.Evolver()
    sat_homog.staticModel(mtot=const.msat/const.mjup, yenv=0.27, zenv=0.05, mcore=20., t10=200)
    
    sat_rain = evolve.Evolver(phase_t_offset=-370.)
    sat_rain.staticModel(mtot=const.msat/const.mjup, yenv=0.27, zenv=0.05, mcore=20., t10=200., include_he_immiscibility=True)
    
    
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
        raise ValueError('no positive brunt_b in the helium rain models.')
    ax[0].set_xlabel(r'$r/R$')
    ax[0].set_ylabel(r'$B$')
    
    ax[1].semilogy(jup_rain.rf, jup_rain.brunt_n2)
    ax[1].semilogy(sat_rain.rf, sat_rain.brunt_n2)
    ax[1].set_xlabel(r'$r/R$')
    ax[1].set_ylabel(r'$N^2\ \ (\rm s^{-2})$')
    ax[2].set_ylim(1e-10, 1e-3)
    
    ax[2].semilogy(jup_rain.rf, jup_rain.lamb_s12)
    ax[2].semilogy(sat_rain.rf, sat_rain.lamb_s12)
    ax[2].set_xlabel(r'$r/R$')
    ax[2].set_ylabel(r'$S_{\ell=1}^2\ \ {\rm s^{-2}}$')
    ax[2].set_ylim(1e-10, 1e-3)
    
    