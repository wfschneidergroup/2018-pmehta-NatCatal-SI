"""Module for ammonia synthesis microkinetic model
Inspired by Grabow, Computational Catalyst Screening,
RSC Catalysis Series, 1-58, 2013
Copyright 2017, Prateek Mehta
"""
import numpy as np
from mpmath import mp, findroot
from scipy.integrate import odeint
from ase.thermochemistry import HarmonicThermo as HT
from ase import units

mp.dps = 50
# mp.pretty = True

# Physical constants and conversion factors
J2eV = 6.24150974E18            # eV/J
Na = 6.0221415E23             # mol-1
h = 6.626068E-34 * J2eV      # in eV*s
kb = 1.3806503E-23 * J2eV     # in eV/K

kJ_mol2eV = 0.01036427
wave2eV = 0.00012398

# Standard entropy of gas phase species
SN2g = 191.61e-3 * kJ_mol2eV
SH2g = 130.68e-3 * kJ_mol2eV
SNH3g = 192.77e-3 * kJ_mol2eV

# Use gas phase N2 and H2 energies as reference
EN2g = 0
EH2g = 0

# From NIST
ZPEN2g = 0.144
ZPEH2g = 0.267
ZPENH3g = 0.903


class NH3mkm:

    def __init__(self,
                 T,
                 EN,
                 Evib=0.0,
                 p0=1.,
                 N2_frac=0.25,
                 dps=25,
                 X=0.01,
                 model='effective',
                 alkali_promotion=False,
                 use_HT=False,
                 nlevels=1,
                 Tv=3000,
                 dist='Treanor',
                 use_alpha=False,
                 site='step'):

        if dps is not None:
            mp.dps = dps

        # From NIST
        self.ZPEN2g = ZPEN2g
        self.ZPEH2g = ZPEH2g
        self.ZPENH3g = ZPENH3g

        self.model = model
        self.nlevels = nlevels
        self.T = T
        self.Tv = Tv
        self.use_alpha = use_alpha

        if dist == 'Treanor':
            self.Elevels, self.plevels = self.treanor_dist()
        elif dist == 'Boltzmann':
            self.Elevels, self.plevels = self.boltzmann_dist()

        self.EN = EN
        self.Evib = Evib
        self.p0 = p0
        self.N2_frac = N2_frac
        self.pN2_0 = N2_frac * p0
        self.pH2_0 = (1 - N2_frac) * p0
        self.X = X
        self.alkali_promotion = alkali_promotion

        pN2, pH2, pNH3 = self.get_pressures()
        self.pN2 = pN2
        self.pH2 = pH2
        self.pNH3 = pNH3

        self.use_HT = use_HT
        self.site = site

        # Vib freq in cm^-1 from vojvodic cpl
        freqN = np.array([434, 465, 508]) * wave2eV
        freqNH = np.array([372, 404, 511, 686, 714, 3430]) * wave2eV
        freqNH2 = np.array([94, 310, 454, 612,
                            661, 674, 1480, 3466, 3568]) * wave2eV
        freqNN = np.array([378, 408, 444, 479, 527]) * wave2eV

        # The values are in meV. These are h\mu. From Honakala Science 2005
        freqH = np.array([44, 121, 161]) / 1000.

        self.freqN = freqN
        self.freqNN = freqNN
        self.freqNH2 = freqNH2
        self.freqNH = freqNH
        self.freqH = freqH

        self.ZPEN = sum(freqN) * 0.5
        self.ZPENH = sum(freqNH) * 0.5
        self.ZPENH2 = sum(freqNH2) * 0.5
        self.ZPENN = sum(freqNN) * 0.5
        self.ZPEH = sum(freqH) * 0.5
        # We can set this later
        self.Xrc_calc = False

    def get_pressures(self):
        N2_frac = self.N2_frac
        H2_frac = 1 - N2_frac
        X = self.X

        # N is limiting
        nN2 = N2_frac - X * N2_frac
        nH2 = H2_frac - 3 * X * N2_frac
        nNH3 = 2 * X * N2_frac

        ntot = nN2 + nH2 + nNH3

        pN2 = self.p0 * nN2 / ntot
        pH2 = self.p0 * nH2 / ntot
        pNH3 = self.p0 * nNH3 / ntot

        return pN2, pH2, pNH3

    def boltzmann_dist(self):
        Tv = self.Tv

        ws = np.array([2372.45, 18.1017])
        cm1eV = 0.00012398426
        EN_vibs = cm1eV * ws

        nlevels = self.nlevels
        levels = np.arange(nlevels)
        Ei_harmonic = levels * EN_vibs[0]  # energy of ith vibrational state
        Ei_anharmonic = EN_vibs[0] * levels - EN_vibs[1] * levels ** 2
        P_B = np.exp(-Ei_harmonic / units.kB / Tv)
        P_B = P_B / sum(P_B)
        return Ei_anharmonic, P_B

    def treanor_dist(self):
        Tv = self.Tv
        Tg = self.T
        nlevels = self.nlevels

        ws = np.array([2372.45, 18.1017])
        cm1eV = 0.00012398426
        EN_vibs = cm1eV * ws

        levels = np.arange(nlevels)
        Ei_harmonic = levels * EN_vibs[0]  # energy of ith vibrational state
        Ei_anharmonic = EN_vibs[0] * levels - EN_vibs[1] * levels ** 2

        P_T = np.exp(-Ei_harmonic / units.kB / Tv
                     + (Ei_harmonic - Ei_anharmonic) / units.kB / Tg)

        P_T = P_T / sum(P_T)
        return Ei_anharmonic, P_T

    def get_effective_Evib(self):
        pi = self.plevels
        Ei = self.Elevels
        return np.dot(Ei, pi)

    def get_N2_pressures(self, use_conversion=True):
        if use_conversion:
            return self.plevels * self.pN2
        else:
            return self.plevels * self.pN2_0

    def get_rxn_energies(self):

        EN = self.EN
        Evib = self.Evib

        if self.model == 'effective':
            dE = np.zeros(5)          # array initialization
            if self.site is 'step':
                dE[0] = 2 * (EN) - Evib    # N adsorption
                dE[1] = 2 * (0.10 * EN - 0.44)  # H adsorption
                dE[2] = -0.19 * EN - 0.16  # NH formation
                dE[3] = -0.39 * EN - 0.37  # NH2 formation
                dE[4] = -0.72 * EN + 1.01  # NH3 formation

            elif self.site is 'terrace':
                dE[0] = 2 * EN - Evib
                dE[1] = 2 * (0.0 * EN - 0.32)
                dE[2] = -0.20 * EN - 0.33
                dE[3] = -0.35 * EN + 0.24
                dE[4] = -0.45 * EN + 0.21

        elif self.model == 'explicit':
            nlevels = self.nlevels
            Elevels = self.Elevels
            dE = np.zeros(nlevels + 4)

            for level in range(nlevels):
                dE[level] = 2 * (EN) - Elevels[level]  # N adsorption

            # Index is now level
            if self.site is 'step':
                dE[level + 1] = 2 * (0.10 * EN - 0.44)  # H adsorption
                dE[level + 2] = -0.19 * EN - 0.16  # NH formation
                dE[level + 3] = -0.39 * EN - 0.37  # NH2 formation
                dE[level + 4] = -0.72 * EN + 1.01  # NH3 formation

            elif self.site is 'terrace':
                dE[level + 1] = 2 * (0.00 * EN - 0.32)
                dE[level + 2] = -0.20 * EN - 0.33
                dE[level + 3] = -0.35 * EN + 0.24
                dE[level + 4] = -0.45 * EN + 0.21
        return dE

    def get_rxn_entropies(self):
        # Surface entropies are assumed to be zero
        # SN = SH = SNH = SNH2 = SNH3 = 0
        # Frozen adsorbate approximation

        # Entropy changes
        # dS = np.zeros(5)
        nlevels = self.nlevels
        dS = np.zeros(nlevels + 4)

        if not self.use_HT:
            for level in range(nlevels):
                dS[level] = -SN2g

            dS[level + 1] = -SH2g
            dS[level + 4] = SNH3g
        else:
            SN = HT(2 * self.freqN).get_entropy(self.T, verbose=False)
            SH = HT(2 * self.freqH).get_entropy(self.T, verbose=False)
            SNH = HT(2 * self.freqNH).get_entropy(self.T, verbose=False)
            SNH2 = HT(2 * self.freqNH2).get_entropy(self.T, verbose=False)

            for level in range(nlevels):
                dS[level] = 2 * SN - SN2g
            dS[level + 1] = 2 * SH - SH2g
            dS[level + 2] = SNH - SN - SH
            dS[level + 3] = SNH2 - SNH - SH
            dS[level + 4] = SNH3g - SNH2 - SH

        return dS

    def get_Eacts(self):
        dE = self.get_rxn_energies()
        # Activation energy barriers
        Ea = np.zeros(5)          # array initialization

        EN = self.EN

        if self.model == 'effective':
            Evib = self.Evib
            if self.site is 'step':
                Ea[0] = 1.57 * EN + 1.56  # N dissoc
                alpha = self.calculate_alpha(Ea[0])
                Ea[0] -= alpha * Evib
                Ea[2] = 0.14 * dE[2] + 1.38  # NH
                Ea[3] = 0.85 * dE[3] + 1.74  # NH2
                Ea[4] = 0.24 * dE[4] + 1.48  # NH3

            elif self.site is 'terrace':
                Ea[0] = 1.58 * EN + 2.33
                alpha = self.calculate_alpha(Ea[0])
                Ea[0] -= alpha * Evib
                Ea[2] = 0.56 * dE[2] + 1.23
                Ea[3] = 0.44 * dE[3] + 1.16
                Ea[4] = 0.92 * dE[4] + 0.79

        elif self.model == 'explicit':
            nlevels = self.nlevels
            Elevels = self.Elevels
            Ea = np.zeros(nlevels + 4)

            if self.site is 'step':
                for level in range(nlevels):
                    Ea0 = 1.57 * EN + 1.56
                    alpha = self.calculate_alpha(Ea0)
                    Ea[level] = Ea0 - alpha * Elevels[level]
                Ea[level + 2] = 0.14 * dE[level + 2] + 1.38  # NH
                Ea[level + 3] = 0.85 * dE[level + 3] + 1.74  # NH2
                Ea[level + 4] = 0.24 * dE[level + 4] + 1.48  # NH3

            elif self.site is 'terrace':
                for level in range(nlevels):
                    Ea0 = 1.58 * EN + 2.33
                    alpha = self.calculate_alpha(Ea0)
                    Ea[level] = Ea0 - alpha * Elevels[level]
                Ea[level + 2] = 0.56 * dE[level + 2] + 1.23
                Ea[level + 3] = 0.44 * dE[level + 3] + 1.16
                Ea[level + 4] = 0.92 * dE[level + 4] + 0.79
        return Ea

    def get_TS_entropies(self):

        SN2g = 192.77 / Na * J2eV    # eV/K
        SH2g = 130.0 / Na * J2eV
        SNH3g = 191.0 / Na * J2eV

        nlevels = self.nlevels

        # Entropy changes to the transition state
        STS = np.zeros(nlevels + 4)
        for level in range(nlevels):
            STS[level] = -SN2g
        STS[level + 1] = -SH2g
        STS[level + 4] = SNH3g
        return STS

    def get_ZPEs_rxn(self):
        # Vib freq in cm^-1

        nlevels = self.nlevels
        ZPEs = np.zeros(nlevels + 4)
        for level in range(nlevels):
            ZPEs[level] = 2 * self.ZPEN - ZPEN2g
        ZPEs[level + 1] = 2 * self.ZPEH - ZPEH2g
        ZPEs[level + 2] = self.ZPENH - self.ZPEH - self.ZPEN
        ZPEs[level + 3] = self.ZPENH2 - self.ZPENH - self.ZPEH
        ZPEs[level + 4] = ZPENH3g - self.ZPENH2 - self.ZPEH
        return ZPEs

    def get_ZPEs_act(self, TSstate='initial'):

        nlevels = self.nlevels

        # Vib freq in cm^-1
        ZPEs = np.zeros(nlevels + 4)
        for level in range(nlevels):
            ZPEs[level] = self.ZPENN - ZPEN2g

        if TSstate is 'final':
            # Full ZPE change
            ZPEs[level + 1] = 2 * self.ZPEH - ZPEH2g
            ZPEs[level + 2] = self.ZPENH - self.ZPEH - self.ZPEN
            ZPEs[level + 3] = self.ZPENH2 - self.ZPENH - self.ZPEH
            ZPEs[level + 4] = ZPENH3g - self.ZPENH2 - self.ZPEH

        elif TSstate is 'average':
            ZPEs[level + 1] = (2 * self.ZPEH + ZPEH2g) / 2. - ZPEH2g
            ZPEs[level + 2] = self.ZPENH - (self.ZPEH + self.ZPEN) / 2.
            ZPEs[level + 3] = self.ZPENH2 - (self.ZPENH + self.ZPEH) / 2.
            ZPEs[level + 4] = ZPENH3g - (self.ZPENH2 + self.ZPEH) / 2.
        return ZPEs

    def get_rate_constants(self):
        """This function calculates all rate constants based on
        the DFT energies for each species.
        No coverage dependence is included.
        """

        # Gas phase entropies
        kbT = kb * self.T                   # in eV

        dE = self.get_rxn_energies()
        dS = self.get_rxn_entropies()
        Ea = self.get_Eacts()
        STS = self.get_TS_entropies()
        ZPEs = self.get_ZPEs_rxn()
        TSZPEs = self.get_ZPEs_act()

        dE += ZPEs
        Ea += TSZPEs
        nlevels = self.nlevels
        if self.alkali_promotion:
            for level in range(nlevels):
                dE[level] += 0.02  # N2 inc by 0.01
                Ea[level] -= 0.15  # Barrier lowered by 0.15
            dE[level + 2] += 0.24 - 0.01  # NH inc by 0.24
            dE[level + 3] += 0.27 - 0.24  # NH2 inc by 0.27
            dE[level + 4] += 0 - 0.27  # NH3 inc by 0.54, but not adsorbed

        self.dE = dE
        self.dS = dS
        self.Ea = Ea
        self.STS = STS
        self.ZPEs = ZPEs
        self.TSZPEs = TSZPEs

        # Calculate equilibrium and rate constants
        K = np.zeros(len(dE))           # equilibrium constants
        kf = np.zeros(len(dE))                # forward rate constants
        kr = np.zeros(len(dE))             # reverse rate constants

        for i in range(len(dE)):
            dG = dE[i] - self.T*dS[i]
            K[i] = mp.exp(-dG / kbT)
            Ea[i] = max([0, dE[i], Ea[i]])   # Enforce Ea > 0, and Ea > dE
            kf[i] = kbT/h * mp.exp(STS[i]/kb) * mp.exp(-Ea[i]/kbT)
            kr[i] = kf[i]/K[i]       # enforce thermodynamic consistency

        self.kf = kf
        self.kr = kr
        return (kf, kr)

    def get_sabatier_rate(self, use_conversion=False,
                          N2_only=False, non_N2=False):

        kf, kr = self.get_rate_constants()
        sabatier_rates = np.zeros(5)

        if not use_conversion:
            pN2 = self.pN2_0
            pH2 = self.pH2_0

        else:
            pN2 = self.pN2
            pH2 = self.pH2

        if self.model == 'effective':
            sabatier_rates[0] = kf[0] * pN2 * 1
            sabatier_rates[1] = kf[1] * pH2 * 1
            sabatier_rates[2] = 0.5 * kf[2] * 0.5 * 0.5
            sabatier_rates[3] = 0.5 * kf[3] * 0.5 * 0.5
            sabatier_rates[4] = 0.5 * kf[4] * 0.5 * 0.5

        elif self.model == 'explicit':
            sabatier_rates[0] = 0
            nlevels = self.nlevels
            pressures = self.get_N2_pressures(use_conversion=use_conversion)

            for level in range(nlevels):
                p = pressures[level]
                sabatier_rates[0] += kf[level] * p

            sabatier_rates[1] = kf[level + 1] * pH2 * 1
            sabatier_rates[2] = 0.5 * kf[level + 2] * 0.5 * 0.5
            sabatier_rates[3] = 0.5 * kf[level + 3] * 0.5 * 0.5
            sabatier_rates[4] = 0.5 * kf[level + 4] * 0.5 * 0.5

        if N2_only:
            sabatier_rates = np.array([sabatier_rates[0]])
        elif non_N2:
            sabatier_rates = np.array(sabatier_rates[1:])
        rds = np.argmin(sabatier_rates)
        sabatier_rate = sabatier_rates[rds]

        self.sabatier_rates = sabatier_rates
        self.rds = rds
        self.sabatier_rate = sabatier_rate

        return rds, sabatier_rate

    def get_rates(self, theta):
        if not self.Xrc_calc:
            kf, kr = self.get_rate_constants()
        else:
            kf = self.kf
            kr = self.kr

        tN = theta[0]
        tH = theta[1]        # theta of H
        tNH = theta[2]             # theta of NH
        tNH2 = theta[3]             # theta of NH2
        tstar = 1.0 - tN - tH - tNH - tNH2   # site balance for tstar

        rate = np.zeros(5)
        # Rates
        if self.model == 'effective':
            rate[0] = kf[0] * self.pN2 * tstar ** 2 - kr[0] * tN ** 2
            rate[1] = kf[1] * self.pH2 * tstar ** 2 - kr[1] * tH ** 2
            rate[2] = kf[2] * tN * tH - kr[2] * tNH * tstar
            rate[3] = kf[3] * tNH * tH - kr[3] * tNH2 * tstar
            rate[4] = kf[4] * tNH2 * tH - kr[4] * self.pNH3 * tstar ** 2

        elif self.model == 'explicit':
            nlevels = self.nlevels
            pressures = self.get_N2_pressures()
            # Could use vector algebra here
            rate[0] = 0
            for level in range(nlevels):
                p = pressures[level]
                rate[0] += kf[level] * p * tstar ** 2 - kr[level] * tN ** 2

            rate[1] += kf[level + 1] * self.pH2 * tstar ** 2 \
                       - kr[level + 1] * tH ** 2
            rate[2] = kf[level + 2] * tN * tH - kr[level + 2] * tNH * tstar
            rate[3] = kf[level + 3] * tNH * tH - kr[level + 3] * tNH2 * tstar
            rate[4] = kf[level + 4] * tNH2 * tH \
                      - kr[level + 4] * self.pNH3 * tstar ** 2

        return rate

    @staticmethod
    def get_odes(theta, t, self):
        """This needs to be a staticmethod to work well
        with odeint"""

        rate = self.get_rates(theta)

        # Time derivatives of theta
        dt = [0] * 4
        dt[0] = 2 * rate[0] - rate[2]                          # d(tN)/dt
        dt[1] = 2 * rate[1] - rate[2] - rate[3] - rate[4]      # d(tH)/dt
        dt[2] = rate[2] - rate[3]                              # d(tNH)/dt
        dt[3] = rate[3] - rate[4]                              # d(tNH2)/dt
        return dt

    @staticmethod
    def check_site_type(site_type, theta0):
        if site_type == 'single':
            func = NH3mkm.get_odes
            if not len(theta0) == 4:
                theta0 = mp.zeros(1, 4)

        elif site_type == 'two-site':
            func = NH3mkm.get_odes_two_site
            if not len(theta0) == 6:
                theta0 = mp.zeros(1, 6)
        return func, theta0

    def integrate_odes(self,
                       site_type='single',
                       theta0=mp.zeros(1, 4),
                       timespan=[0, 1e8],
                       h0=1e-20,
                       mxstep=200000,          # maximum number of steps
                       rtol=1E-12,            # relative tolerance
                       atol=1E-15,           # Absolute tolerance
                       full_output=1):

        func, theta0 = NH3mkm.check_site_type(site_type, theta0)
        self.theta0 = theta0
        args = self,

        # Integrate the ODEs for 1E10 sec (enough to reach steady-state)
        theta, out = odeint(func,         # system of ODEs
                            theta0,                  # initial guess
                            timespan,                 # time span
                            args=args,
                            h0=h0,              # initial time step
                            mxstep=mxstep,          # maximum number of steps
                            rtol=rtol,            # relative tolerance
                            atol=atol,           # Absolute tolerance
                            full_output=full_output)
        self.integrated_thetas = theta
        self.ode_output = out
        
        # Return the value of theta at the last timestep
        return theta[-1, :]

    def find_steady_state_roots(self,
                                theta0=[0., 0., 0., 0.],
                                tol=1e-22,
                                solver='secant'):
        mp.dps = 25
        mp.pretty = True

        def get_findroot_eqns(*args):
            return self.get_odes(args, 0, self)

        theta = findroot(get_findroot_eqns,
                         tuple(theta0),
                         solver=solver,
                         tol=tol,
                         multidimensional=True)

        self.steady_state_theta = theta
        return theta

    def get_rates_two_site(self, theta):
        '''
        Compute reaction rates based on the two-site model
        used by Vojvodic et al, Chem Phys Lett, 2014, 598, 108
        '''
        kf, kr = self.get_rate_constants()

        tstepN = theta[0]
        tstepH = theta[1]        # theta of H
        tstepNH = theta[2]             # theta of NH
        tstepNH2 = theta[3]             # theta of NH2
        tstep = 0.5 - tstepN - tstepH - tstepNH - tstepNH2  # steps

        tstarN = theta[4]
        tstarNH = theta[5]
        tstar = 0.5 - tstarN - tstarNH

        # Rates
        rate = np.zeros(7)
        rate[0] = kf[0] * self.pN2 * tstep * tstar - kr[0] * tstepN * tstarN
        rate[1] = kf[1] * self.pH2 * tstep ** 2 - kr[1] * tstepH ** 2
        rate[2] = kf[2] * tstarN * tstepH - kr[2] * tstarNH * tstep
        rate[3] = kf[2] * tstepN * tstepH - kr[2] * tstepNH * tstep
        rate[4] = kf[3] * tstarNH * tstepH - kr[3] * tstepNH2 * tstar
        rate[5] = kf[3] * tstepNH * tstepH - kr[3] * tstepNH2 * tstep
        rate[6] = kf[4] * tstepNH2 * tstepH - kr[4] * self.pNH3 * tstep ** 2
        return rate

    @staticmethod
    def get_odes_two_site(theta, t, self):
        """This needs to be a staticmethod to work well
        with odeint"""

        rate = self.get_rates_two_site(theta)

        # Time derivatives of theta
        dt = np.zeros(6)
        # d(tstepN)/dt
        dt[0] = rate[0] - rate[3]
        # d(tstepH)/dt
        dt[1] = 2 * rate[1] - rate[2] - rate[3] - rate[4] - rate[5] - rate[6]
        # d(tstepNH)/dt
        dt[2] = rate[3] - rate[5]
        # d(tstepNH2)/dt
        dt[3] = rate[4] + rate[5] - rate[6]
        # d(tstarN)/dt
        dt[4] = rate[0] - rate[2]
        # d(tstarNH)/dt
        dt[5] = rate[2] - rate[4]
        return dt

    def get_N2_rds_rate(self, site_type='single'):
        '''
        Compute rates assuming N2 dissociation is rate
        limiting
        '''
        kf, kr = self.get_rate_constants()
        K = kf / kr
        self.K = K

        lambdaH = np.sqrt(K[1] * self.pH2)
        lambdaNH2 = self.pNH3 / K[4] / lambdaH
        lambdaNH = lambdaNH2 / K[3] / lambdaH
        lambdaN = lambdaNH / K[2] / lambdaH

        if site_type is 'single':

            theta_star = 1 / (1 + lambdaN + lambdaH
                              + lambdaNH + lambdaNH2)

            thetaH = lambdaH * theta_star
            thetaN = lambdaN * theta_star
            thetaNH = lambdaNH * theta_star
            thetaNH2 = lambdaNH2 * theta_star

            theta_N2_rds = [thetaN, thetaH, thetaNH, thetaNH2]
            rates_N2_rds = self.get_rates(theta_N2_rds)

        elif site_type == 'two-site':
            theta_star = 0.5 / (1 + lambdaN + lambdaNH)
            theta_step = 0.5 / (1 + lambdaN + lambdaH
                                + lambdaNH + lambdaNH2)

            thetaH_step = lambdaH * theta_step
            thetaN_step = lambdaN * theta_step
            thetaNH_step = lambdaNH * theta_step
            thetaNH2_step = lambdaNH2 * theta_step

            thetaN_star = lambdaN * theta_star
            thetaNH_star = lambdaNH * theta_star

            theta_N2_rds = [thetaN_step,
                            thetaH_step,
                            thetaNH_step,
                            thetaNH2_step,
                            thetaN_star,
                            thetaNH_star]

            rates_N2_rds = self.get_rates_two_site(theta_N2_rds)

        self.theta_N2_rds = theta_N2_rds
        self.rates_N2_rds = rates_N2_rds

        return theta_N2_rds, rates_N2_rds

    def get_NH_rds_rate(self, site_type='single'):
        '''
        Compute rates assuming NH formation is rate
        limiting
        '''
        kf, kr = self.get_rate_constants()
        K = kf / kr
        self.K = K

        lambdaN = np.sqrt(K[0] * self.pN2)
        lambdaH = np.sqrt(K[1] * self.pH2)
        lambdaNH2 = self.pNH3 / K[4] / lambdaH
        lambdaNH = lambdaNH2 / K[3] / lambdaH

        if site_type is 'single':

            theta_star = 1 / (1 + lambdaN + lambdaH
                              + lambdaNH + lambdaNH2)

            thetaH = lambdaH * theta_star
            thetaN = lambdaN * theta_star
            thetaNH = lambdaNH * theta_star
            thetaNH2 = lambdaNH2 * theta_star

            theta_NH_rds = [thetaN, thetaH, thetaNH, thetaNH2]
            rates_NH_rds = self.get_rates(theta_NH_rds)

        self.theta_NH_rds = theta_NH_rds
        self.rates_NH_rds = rates_NH_rds

        return theta_NH_rds, rates_NH_rds

    def calculate_Xrc(self):
        '''
        Degree of rate control
        '''
        kf0, kr0 = self.get_rate_constants()
        theta0 = self.integrate_odes()
        r0 = self.get_rates(theta0)[0]

        # Currently only implemented for single site
        # and no excitation
        delta = 0.1  # change of 10%
        Xrc_rates = np.zeros(5)

        self.kf0 = kf0
        self.kr0 = kr0

        for s in range(len(Xrc_rates)):
            kf = kf0.copy()
            kr = kr0.copy()

            kf[s] = (1 + delta) * kf0[s]
            kr[s] = (1 + delta) * kr0[s]

            self.kf = kf
            self.kr = kr
            self.Xrc_calc = True

            theta = self.integrate_odes(theta0=theta0)
            rates = self.get_rates(theta)
            Xrc_rates[s] = rates[0]

        Xrc = (Xrc_rates - r0) / (delta * r0)
        self.Xrc_rates = Xrc_rates
        self.Xrc = Xrc
        return Xrc

    def calculate_alpha(self, Ea):
        '''
        Calculate Fridman alpha
        '''
        if not self.use_alpha:
            alpha = 1
        else:
            Eaf = Ea
            Eab = Eaf - 2 * self.EN
            alpha = Eaf / (Eaf + Eab)
        alpha = max(0, alpha)
        self.alpha = alpha
        return alpha
