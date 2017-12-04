import numpy as np
from plasmautils import perform_rate_analysis

metals = 'Re, Ru, Rh, Co, Ni, Pt, Pd'

if not len(metals):
    ENs = np.linspace(-1.5, 1.0, 30)
else:
    ENs = np.array([-1.385, -0.535, -0.125, -0.045, 0.155, 0.725, 0.995])
    metals = 'Re, Ru, Rh, Co, Ni, Pt, Pd'.split(', ')

perform_rate_analysis('site=terrace-Tg=473.0K-Tv=3000.0K-P=1.0atm-dist=Treanor-nlevels=11-alpha=True', ENs, T=473.0, p0=1.0,
    site='terrace', Evib=0.0, alkali_promotion=False,
    X=0.01, use_alpha=True, model='explicit',
    nlevels=11, dist='Treanor', Tv=3000.0,
    metals=metals,
    Agg=True,
    N2_frac=0.25)
