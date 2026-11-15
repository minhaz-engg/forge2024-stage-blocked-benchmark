"""Fair ablation: apply the identical Gamma-Poisson online rescaling to the GBM and RF forecasts."""
import numpy as np, pandas as pd, zp, warnings, sys
warnings.filterwarnings('ignore')
sys.argv = ['x', '0']
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 's06_forecast.py')).read().split('rows, series = [], []')[0]      # import class defs only
ns = {}; exec(src, ns)
from scipy.stats import poisson
st = pd.read_csv('results/stage_table.csv'); S16A = st[st.well=='16A'].stage.tolist(); S16B = st[st.well=='16B'].stage.tolist()
W_GRID = [1, 3, 10, 30, 100]
def rescale(lam0, n, w):
    cn = np.concatenate([[0], np.cumsum(n)[:-1]]); cl = np.concatenate([[0], np.cumsum(lam0)[:-1]])
    return (w + cn) / (w + cl) * lam0
rows = []
for cat in ['Niemz', 'Zhu']:
    d = ns['add_ar'](pd.read_csv(f'results/binned_{cat}.csv')) if 'n_lag1' not in pd.read_csv(f'results/binned_{cat}.csv', nrows=1) else pd.read_csv(f'results/binned_{cat}.csv')
    for mname in ['GBM', 'RF']:
        for scheme, pairs in [('LOSO', [([x for x in S16A if x != s], s) for s in S16A]),
                              ('FC', [(S16A[:j], S16A[j]) for j in range(2, len(S16A))]),
                              ('16A->16B', [(S16A, s) for s in S16B if (d.stage == s).sum() >= 6])]:
            for trs, s in pairs:
                tr, te = d[d.stage.isin(trs)], d[d.stage == s]
                if mname == 'GBM':
                    m = ns['GBM'](zp.FEATS).fit(tr); mk = lambda: ns['GBM'](zp.FEATS, params=m.p_, inner=False)
                else:
                    m = ns['RF'](zp.FEATS).fit(tr); mk = lambda: ns['RF'](zp.FEATS)
                # choose w by inner LOSO with fixed hyper-parameters
                ll = {w: 0.0 for w in W_GRID}
                for s2 in tr.stage.unique():
                    a, b = tr[tr.stage != s2], tr[tr.stage == s2]
                    lam0 = np.maximum(mk().fit(a).predict(b), 1e-9)
                    for w in W_GRID: ll[w] += zp.pois_ll(b.n.values, rescale(lam0, b.n.values, w))
                w = max(ll, key=ll.get)
                lam = rescale(np.maximum(m.predict(te), 1e-9), te.n.values, w); null = zp.NullModel().fit(tr).predict(te)
                N, L = te.n.sum(), lam.sum()
                rows.append(dict(catalog=cat, scheme=scheme, stage=s, model=f'{mname}-adaptive', w=w, LL=zp.pois_ll(te.n.values, lam),
                                 IG=zp.info_gain_bits(te.n.values, lam, null), N_obs=N, N_pred=L,
                                 Ntest_p=min(poisson.cdf(N, L), 1 - poisson.cdf(N - 1, L)), rmse=np.sqrt(np.mean((te.n.values - lam)**2))))
            print(cat, mname, scheme, 'done', flush=True)
pd.DataFrame(rows).to_csv('results/forecast_scores_ml_adaptive.csv', index=False)
