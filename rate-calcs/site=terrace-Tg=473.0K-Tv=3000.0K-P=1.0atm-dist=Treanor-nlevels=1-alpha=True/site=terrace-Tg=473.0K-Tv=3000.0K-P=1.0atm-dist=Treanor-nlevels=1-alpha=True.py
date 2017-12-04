import numpy as np
from plasmautils import perform_ss_rate_analysis

metals = ''

if not len(metals):
    ENs = np.linspace(1, -1.5, 120)
else:
    ENs = np.array([])
    metals = ''.split(', ')

perform_ss_rate_analysis('site=terrace-Tg=473.0K-Tv=3000.0K-P=1.0atm-dist=Treanor-nlevels=1-alpha=True', ENs, T=473.0, p0=1.0,
    site='terrace', Evib=0.0, alkali_promotion=False,
    X=0.01, use_alpha=True, model='explicit',
    nlevels=1, dist='Treanor', Tv=3000.0,
    metals=metals,
    Agg=True,
    N2_frac=0.25)
