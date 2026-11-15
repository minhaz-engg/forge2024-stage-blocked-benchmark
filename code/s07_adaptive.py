"""Bayesian-adaptive SI-memory forecast: productivity A updated online within each stage (Gamma-Poisson),
prior from training stages. Uses only counts observed BEFORE the bin being forecast."""
import numpy as np, pandas as pd, zp
from scipy.stats import poisson, wilcoxon
st = pd.read_csv('results/stage_table.csv'); S16A = st[st.well=='16A'].stage.tolist(); S16B = st[st.well=='16B'].stage.tolist()
W_GRID = [1, 3, 10, 30, 100]          # prior weight (pseudo-events) of the training-based forecast

def adaptive_predict(phys, te, w_abs):
    """lambda_k = phi_k * lambda0_k,  phi_k = (w + sum_{j<k} n_j) / (w + sum_{j<k} lambda0_j)  (Gamma-Poisson)."""
    lam0 = phys.predict(te); n = te.n.values
    cn = np.concatenate([[0], np.cumsum(n)[:-1]]); cl = np.concatenate([[0], np.cumsum(lam0)[:-1]])
    return (w_abs + cn) / (w_abs + cl) * lam0

def fit_w(tr):
    stg = tr.stage.unique(); best = None
    for w in W_GRID:
        ll = 0
        for s in stg:
            a, b = tr[tr.stage != s], tr[tr.stage == s]
            ph = zp.SIMemModel().fit(a)
            scale = 1.0
            ll += zp.pois_ll(b.n.values, adaptive_predict(ph, b, w * scale))
        if best is None or ll > best[0]: best = (ll, w)
    return best[1]

rows = []
for cat in ['Niemz', 'Zhu']:
    d = pd.read_csv(f'results/binned_{cat}.csv')
    for scheme, pairs in [('LOSO', [([x for x in S16A if x != s], s) for s in S16A]),
                          ('FC', [(S16A[:j], S16A[j]) for j in range(2, len(S16A))]),
                          ('16A->16B', [(S16A, s) for s in S16B if (d.stage == s).sum() >= 6])]:
        for trs, s in pairs:
            tr, te = d[d.stage.isin(trs)], d[d.stage == s]
            w = fit_w(tr); ph = zp.SIMemModel().fit(tr)
            scale = 1.0
            lam = adaptive_predict(ph, te, w * scale); null = zp.NullModel().fit(tr).predict(te)
            N, L = te.n.sum(), lam.sum()
            rows.append(dict(catalog=cat, scheme=scheme, stage=s, model='SI-mem-adaptive', w=w, LL=zp.pois_ll(te.n.values, lam),
                             IG=zp.info_gain_bits(te.n.values, lam, null), N_obs=N, N_pred=L,
                             Ntest_p=min(poisson.cdf(N, L), 1 - poisson.cdf(N - 1, L)), rmse=np.sqrt(np.mean((te.n.values - lam)**2))))
    print(cat, 'done', flush=True)
A = pd.DataFrame(rows); A.to_csv('results/forecast_scores_adaptive.csv', index=False)
R = pd.concat([pd.read_csv('results/forecast_scores_seed0.csv'), A], ignore_index=True)
R.to_csv('results/forecast_scores_all.csv', index=False)
pd.set_option('display.width', 250)
for sch in ['LOSO', 'FC']:
    t = R[R.scheme == sch].groupby(['catalog', 'model']).agg(LL=('LL','sum'), IG_mean=('IG','mean'), IG_med=('IG','median'),
        Ntest_pass=('Ntest_p', lambda p: (p > 0.025).mean())).round(3)
    print(f'\n=== {sch} ==='); print(t.to_string())
print(R[R.scheme=='16A->16B'].pivot_table(index=['catalog','stage','N_obs'], columns='model', values='N_pred').round(1).to_string())
L = R[R.scheme == 'LOSO'].pivot_table(index=['catalog', 'stage'], columns='model', values='IG')
for a, b in [('SI-mem-adaptive', 'SI-mem'), ('SI-mem-adaptive', 'GBM+AR'), ('SI-mem-adaptive', 'SI-mem+AR'), ('SI-mem-adaptive','GBM')]:
    dl = L[a] - L[b]; print(f'  {a} - {b}: median {dl.median():+.3f}, wins {int((dl>0).sum())}/{len(dl)}, p={wilcoxon(dl).pvalue:.4f}')
print('chosen w:', A.groupby(['catalog','scheme']).w.agg(lambda x: x.value_counts().to_dict()).to_string())
