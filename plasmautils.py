import numpy as np
from ase import units
from scipy.stats import linregress
from mpmath import mp
mp.dps = 50
# Spectroscopic constants of nitrogen
# we, wexe, weye, weze cite:heijkers-2015-co2con-microw

ws = np.array([2372.45, 18.1017, 1.27552e-2, -7.95949])  # in cm-1
cm1eV = 0.00012398426
E_Nvibs = cm1eV * ws


def make_PES(Hs,
             Eas,
             ntimes,
             names,
             col='C0',
             width=1,
             fontsize=22,
             label=None,
             axis_labels=True,
             Eref=0.,
             ls='-',
             IS_start=0.,
             legend=False):
    import matplotlib.pyplot as plt
    # Initialize
    x = width
    E = Eref
    line, = plt.plot([IS_start, x], [E, E], c=col, ls=ls)
    # col = line.get_color()
    for i, nEaHt in enumerate(zip(names, Eas, Hs, ntimes)):
        name, Ea, H, times = nEaHt

        for time in range(1, times+1):

            if Ea != 0:
                # barrier
                xs = [x, x + 0.5 * width, x + width]
                es = [E, E + Ea, E+H]
                # Fitting polynomial
                p = np.polyfit(xs, es, 2)

                xfit = np.linspace(x, x + width, 10000)
                efit = np.polyval(p, xfit)

                plt.plot(xfit, efit, c=col, ls=ls)

                plt.text(x + 0.8 * width,
                         E + Ea - 0.1, name,
                         fontsize=fontsize)
            else:
                # No barrier
                plt.plot([x, x + width], [E, E+H], c=col, ls=ls)
                plt.text(x, E + H / 2. - 0.12, name, fontsize=fontsize)
            x += width
            E += H
            plt.plot([x, x + width], [E, E], c=col, ls=ls)
            x += width

    if axis_labels:
        plt.ylabel('Potential Energy (eV)', fontsize=fontsize + 2)
        plt.xlabel('Reaction Coordinate', fontsize=fontsize + 2)

    if label:
        line, = plt.plot([], [], ls=ls, label=label)
        if legend:
            plt.legend(fontsize=fontsize)
        return line


def plot_rates(ENs, Rsab, Ro, Rss, prefix='images/'):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-talk')
    plt.figure()
    # print Ro
    plt.semilogy(ENs, Rsab, label='Sabatier')
    if len(Ro):
        plt.semilogy(ENs, Ro, label='single-site ODE')
    plt.semilogy(ENs, Rss, '.', label='Steady state roots')

    plt.xlabel('$E_{\mathrm{N}}$ (eV)')
    plt.ylabel('TOF (1 / s)')
    plt.title(prefix)
    plt.ylim(1e-20, 1e2)
    plt.legend()
    plt.savefig('{0}-rates.png'.format(prefix), dpi=200)


def plot_coverages(ENs, to, tss, prefix='images/'):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-talk')

    if len(to):
        plt.figure()
        plt.plot(ENs, to)

        plt.xlabel('$E_{\mathrm{N}}$ (eV)')
        plt.ylabel('Coverages')
        plt.title(prefix)
        plt.legend(['N', 'H', 'NH', 'NH$_{2}$', '*'])
        plt.savefig('{0}-coverages-ode.png'.format(prefix), dpi=200)

    plt.figure()
    plt.plot(ENs, tss)
    plt.legend(['N', 'H', 'NH', 'NH$_{2}$', '*'])
    plt.xlabel('$E_{\mathrm{N}}$ (eV)')
    plt.ylabel('Coverages')
    plt.title(prefix)
    plt.legend(['N', 'H', 'NH', 'NH$_{2}$', '*'])
    plt.savefig('{0}-coverages-steady-state.png'.format(prefix), dpi=200)


def store_variables(filename,
                    ENs,
                    Rsab,
                    Ro,
                    Rss,
                    to,
                    tss,
                    T, p0,
                    N2_frac,
                    X,
                    site,
                    Evib=0.,
                    metals=[]):

    import json
    d = {}
    d['ENs'] = ENs.tolist()
    d['R_sabatier'] = Rsab
    d['R_steadystate'] = Rss
    d['R_ode'] = Ro
    d['theta_ode'] = to
    d['theta_steadystate'] = tss
    d['T'] = T
    d['p0'] = p0
    d['N2_frac'] = N2_frac
    d['conversion'] = X
    d['site'] = site
    d['Evib'] = Evib
    d['metals'] = metals
    d['prefix'] = filename

    with open('{0}.json'.format(filename), 'w') as f:
        json.dump(d, f)


def load_variables(filename):
    import json

    with open(filename) as f:
        d = json.load(f)
    return d


def plots_from_json(filename, prefix='images'):
    d = load_variables(filename)

    ENs = d['ENs']
    Rsab = d['R_sabatier']
    Rss = d['R_steadystate']
    Ro = d['R_ode']
    to = d['theta_ode']
    tss = d['theta_steadystate']

    plot_rates(ENs, Rsab, Ro, Rss, prefix)
    plot_coverages(ENs, to, tss, prefix)


def perform_rate_analysis(prefix,
                          ENs, Evib=0., T=473,
                          p0=1, N2_frac=0.25, X=0.01,
                          site='step',
                          plot=True,
                          alkali_promotion=False,
                          use_alpha=False,
                          model='effective',
                          nlevels=1,
                          dist='Treanor',
                          Tv=3500,
                          metals=[],
                          Agg=False):

    if Agg:
        import matplotlib
        matplotlib.use('Agg')
    from mkm import NH3mkm

    Rs = []
    Ro = []
    Rss = []
    thetao = []
    thetass = []

    theta0 = np.zeros(4)
    for i, EN in enumerate(ENs):

        mod = NH3mkm(T, EN, Evib, p0, N2_frac, X=X,
                     site=site,
                     alkali_promotion=alkali_promotion,
                     model=model,
                     dist=dist,
                     Tv=Tv,
                     nlevels=nlevels,
                     use_alpha=use_alpha)

        Rds, rsab = mod.get_sabatier_rate()
        Rs.append(rsab)

        theta = mod.integrate_odes(theta0=theta0,
                                   h0=1e-24,
                                   rtol=1e-12,
                                   atol=1e-15,
                                   timespan=[0, 1e9],
                                   mxstep=200000)
        if not len(metals) == 0:
            theta0 = theta

        r = mod.get_rates(theta)
        theta_and_empty = list(theta) + [1 - sum(theta)]
        thetao.append(theta_and_empty)
        Ro.append(r[0])

        try:
            theta = mod.find_steady_state_roots(theta0=theta)
            theta_and_empty = list(theta) + [1 - sum(theta)]
            theta_and_empty = [mp.nstr(t, 25) for t in theta_and_empty]
            if (np.array(theta) > 1).any() or (np.array(theta) < 0.).any():
                theta_and_empty = [np.nan] * 5
        except ValueError:
            # In case there is a tolerance issue
            theta_and_empty = [np.nan] * 5

        thetass.append(theta_and_empty)

        r = mod.get_rates(theta)

        Rss.append(r[0])

    store_variables(prefix,
                    ENs,
                    Rs,
                    Ro,
                    Rss,
                    thetao,
                    thetass,
                    T, p0,
                    N2_frac,
                    X, site,
                    Evib,
                    metals=metals)

    if not len(metals) and plot:
        plot_rates(ENs, Rs, Ro, Rss, prefix)
        plot_coverages(ENs, thetao, thetass, prefix)


def perform_ss_rate_analysis(prefix,
                             ENs, Evib=0., T=473,
                             p0=1, N2_frac=0.25, X=0.01,
                             site='step',
                             plot=True,
                             alkali_promotion=False,
                             use_alpha=False,
                             model='effective',
                             nlevels=1,
                             dist='Treanor',
                             Tv=3500,
                             metals=[],
                             Agg=False):
    if Agg:
        import matplotlib
        matplotlib.use('Agg')
    from mkm import NH3mkm

    Rs = []
    Rss = []
    thetas = []

    theta0 = np.zeros(4)  # [0, 0, 0, 0]
    for i, EN in enumerate(ENs):

        # print type(EN), EN
        mod = NH3mkm(T, EN, Evib, p0, N2_frac, X=X,
                     site=site,
                     alkali_promotion=alkali_promotion,
                     model=model,
                     dist=dist,
                     Tv=Tv,
                     nlevels=nlevels,
                     use_alpha=use_alpha)

        Rds, rsab = mod.get_sabatier_rate()
        Rs.append(rsab)

        if i == 0:
            theta = mod.integrate_odes(theta0=theta0,
                                       h0=1e-24,
                                       rtol=1e-12,
                                       atol=1e-15,
                                       timespan=[0, 1e9],
                                       mxstep=200000)
        try:
            theta_ss = mod.find_steady_state_roots(theta)
        except:
            # If there is an issue re-solve ode
            theta = mod.integrate_odes(theta0=theta,
                                       h0=1e-24,
                                       rtol=1e-12,
                                       atol=1e-15,
                                       timespan=[0, 1e9],
                                       mxstep=200000)
            # Try one more time with reduced tolerance
            try:
                theta_ss = mod.find_steady_state_roots(theta, tol=1e-15)
            except:
                # set it to the ode theta
                theta_ss = theta

        # Now check if theta_ss has unphysical values
        if (np.array(theta) <= 1).all() and (np.array(theta) >= 0.).all():
            theta = theta_ss

        theta_and_empty = list(theta) + [1 - sum(theta)]
        theta_and_empty = [mp.nstr(t, 25) for t in theta_and_empty]

        r = mod.get_rates(theta)
        thetas.append(theta_and_empty)
        Rss.append(r[0])

    # This needs to be updated
    store_variables(prefix, ENs,
                    Rs, [], Rss, [], thetas,
                    T, p0, N2_frac, X, site, Evib,
                    metals=metals)

    if plot:
        plot_rates(ENs, Rs, [], Rss, prefix)
        plot_coverages(ENs, [], thetas, prefix)


def get_prefix(site, T, p, Evib=0., Xrc_calc=False):

    prefix = 'site={0}-T={1}K-P={2}atm-Ev={3}eV'.format(site,
                                                        T,
                                                        p,
                                                        Evib)
    if Xrc_calc:
        prefix = 'Xrc-{0}'.format(prefix)

    return prefix


def get_prefix_explicit(site, T, p, Tv, dist, nlevels,
                        use_alpha, Xrc_calc=False):
    prefix = 'site={site}-Tg={T}K-Tv={Tv}K-P={p}atm-dist={dist}-nlevels={nlevels}-alpha={use_alpha}'
    prefix = prefix.format(**locals())
    if Xrc_calc:
        prefix = 'Xrc-{0}'.format(prefix)
    return prefix


def treanor_dist(Tg, Tv, truncate=10):
    ws = np.array([2372.45, 18.1017])
    cm1eV = 0.00012398426
    EN_vibs = cm1eV * ws

    levels = np.arange(truncate)
    Ei_harmonic = levels * EN_vibs[0]  # energy of i_th vibrational state, eV
    Ei_anharmonic = EN_vibs[0] * levels - EN_vibs[1] * levels ** 2

    P_T = np.exp(-Ei_harmonic / units.kB / Tv
                 + (Ei_harmonic - Ei_anharmonic) / units.kB / Tg)

    P_T = P_T / sum(P_T)
    return Ei_anharmonic, P_T
