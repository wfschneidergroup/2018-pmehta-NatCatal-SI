import numpy as np
from plasmautils import perform_rate_analysis

metals = 'Ru, Co, Fe, Rh, Ni, Pd, Pt'

if not len(metals):
    ENs = np.linspace(-1.5, 1.0, 30)
else:
    ENs = np.array([-0.46, -0.18, -1.27, -0.34, -0.06, 0.91, 0.61])
    metals = 'Ru, Co, Fe, Rh, Ni, Pd, Pt'.split(', ')

perform_rate_analysis('site=step-Tg=473.0K-Tv=3000.0K-P=1.0atm-dist=Treanor-nlevels=1-alpha=True', ENs, T=473.0, p0=1.0,
    site='step', Evib=0.0, alkali_promotion=False,
    X=0.01, use_alpha=True, model='explicit',
    nlevels=1, dist='Treanor', Tv=3000.0,
    metals=metals,
    Agg=True,
    N2_frac=0.25)
