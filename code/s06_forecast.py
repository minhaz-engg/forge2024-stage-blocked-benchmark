"""Stage-wise, leakage-free forecasting benchmark of 10-min event counts (M >= Mc).
Schemes: leave-one-stage-out (LOSO) and forward-chaining (FC) over the nine 16A stages; transfer to 16B."""
import numpy as np, pandas as pd, zp, itertools, sys, warnings
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from scipy.stats import poisson, wilcoxon
warnings.filterwarnings('ignore')
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
st = pd.read_csv('results/stage_table.csv'); cs = pd.read_csv('results/catalog_summary.csv').set_index('catalog')
inj = zp.load_injection()
S16A = st[st.well == '16A'].stage.tolist(); S16B = st[st.well == '16B'].stage.tolist()
CAT_END = {}

GBM_GRID = [dict(max_depth=d, learning_rate=lr, max_iter=it, min_samples_leaf=msl)
            for d, lr, it, msl in itertools.product([2, 3], [0.05], [100, 300], [10, 30])]

def add_ar(d):
    d = d.copy(); g = d.groupby('stage', sort=False)
    for L in [1, 2, 3]:
        d[f'n_lag{L}'] = g.n.shift(L).fillna(0)
    d['n_roll6'] = g.n.transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean()).fillna(0)
    return d

AR = ['n_lag1', 'n_lag2', 'n_lag3', 'n_roll6']

class GBM:
    def __init__(self, feats, params=None, inner=True, seed=0): self.feats, self.params, self.inner, self.seed = feats, params, inner, seed
    def _mk(self, p): return HistGradientBoostingRegressor(loss='poisson', random_state=self.seed, **p)
    def fit(self, d):
        p = self.params
        if p is None and self.inner:      # inner leave-one-stage-out on the training stages only
            stg = d.stage.unique(); best = None
            for p_ in GBM_GRID:
                ll = 0
                for s in stg:
                    tr, te = d[d.stage != s], d[d.stage == s]
                    m = self._mk(p_).fit(tr[self.feats], tr.n)
                    ll += zp.pois_ll(te.n.values, m.predict(te[self.feats]))
                if best is None or ll > best[0]: best = (ll, p_)
            p = best[1]
        self.p_ = p; self.m = self._mk(p).fit(d[self.feats], d.n); return self
    def predict(self, d): return self.m.predict(d[self.feats])

class RF:
    def __init__(self, feats, seed=0): self.feats, self.seed = feats, seed
    def fit(self, d):
        self.m = RandomForestRegressor(n_estimators=300, criterion='poisson', min_samples_leaf=10, max_features=0.5,
                                       random_state=self.seed, n_jobs=2).fit(d[self.feats], d.n); return self
    def predict(self, d): return np.maximum(self.m.predict(d[self.feats]), 1e-6)

class Hybrid(GBM):
    """GBM that also receives the physics model's (SI-memory) rate, fitted on training stages only."""
    def fit(self, d):
        self.phys = zp.SIMemModel().fit(d); d = d.copy(); d['phys'] = self.phys.predict(d)
        return super().fit(d)
    def predict(self, d):
        d = d.copy(); d['phys'] = self.phys.predict(d); return super().predict(d)

class SIMemAR:
    """lambda = A*(Q*h_tau) + c + alpha * n_{k-1}  (physics + short-term clustering)."""
    def fit(self, d):
        from scipy.optimize import minimize
        self.base = zp.SIMemModel().fit(d); x = self.base.transform(d); n = d.n.values; n1 = d.n_lag1.values
        f = lambda p: -zp.pois_ll(n, np.exp(p[0]) * x + np.exp(p[1]) + np.exp(p[2]) * n1)
        r = minimize(f, [np.log(self.base.A), np.log(self.base.c), np.log(0.3)], method='Nelder-Mead', options=dict(maxiter=3000))
        self.A, self.c, self.al = np.exp(r.x); return self
    def predict(self, d): return self.A * self.base.transform(d) + self.c + self.al * d.n_lag1.values

def models():
    fx = zp.FEATS
    return {'Null': lambda: zp.NullModel(), 'SI': lambda: zp.SIModel(), 'SI-mem': lambda: zp.SIMemModel(),
            'RF': lambda: RF(fx, SEED), 'GBM': lambda: GBM(fx, seed=SEED), 'Hybrid': lambda: Hybrid(fx + ['phys'], seed=SEED),
            'SI-mem+AR': lambda: SIMemAR(), 'GBM+AR': lambda: GBM(fx + AR, seed=SEED)}

def evaluate(d, train_stages, test_stage, M):
    tr, te = d[d.stage.isin(train_stages)], d[d.stage == test_stage]
    out = {}
    null = zp.NullModel().fit(tr).predict(te)
    for name, mk in M.items():
        lam = np.maximum(mk().fit(tr).predict(te), 1e-9)
        N = te.n.sum(); Lt = lam.sum()
        out[name] = dict(LL=zp.pois_ll(te.n.values, lam), IG=zp.info_gain_bits(te.n.values, lam, null), N_obs=N, N_pred=Lt,
                         Ntest_p=min(poisson.cdf(N, Lt), 1 - poisson.cdf(N - 1, Lt)),   # two-sided-ish quantile (Zechar 2010)
                         rmse=np.sqrt(np.mean((te.n.values - lam)**2)), lam=lam)
    return out

rows, series = [], []
for cat in ['Niemz', 'Zhu']:
    c = pd.read_csv(f'results/catalog_{cat}_filtered.csv'); mc = cs.loc[cat, 'mc']
    cm = c[c.M >= mc - 1e-9]
    d = zp.add_features(zp.binned_series(inj, st, cm, S16A + S16B))
    d = add_ar(d)
    end = zp.load_catalog(cat).t_s.max()
    d = d[d.t0 + zp.DT <= end].reset_index(drop=True)     # bins beyond catalog end are not observed
    d.to_csv(f'results/binned_{cat}.csv', index=False)
    M = models()
    # LOSO
    for s in S16A:
        r = evaluate(d, [x for x in S16A if x != s], s, M)
        for m, v in r.items():
            rows.append(dict(catalog=cat, scheme='LOSO', stage=s, model=m, **{k: v[k] for k in v if k != 'lam'}))
            series.append(pd.DataFrame(dict(catalog=cat, scheme='LOSO', stage=s, model=m, k=d[d.stage == s].k.values, n=d[d.stage == s].n.values, lam=v['lam'])))
    # forward chaining: predict stage j from stages < j (j >= 3rd stage)
    for j in range(2, len(S16A)):
        s = S16A[j]; r = evaluate(d, S16A[:j], s, M)
        for m, v in r.items():
            rows.append(dict(catalog=cat, scheme='FC', stage=s, model=m, **{k: v[k] for k in v if k != 'lam'}))
    # transfer: all 16A -> 16B stages (observed part only)
    for s in S16B:
        if (d.stage == s).sum() < 6: continue
        r = evaluate(d, S16A, s, M)
        for m, v in r.items():
            rows.append(dict(catalog=cat, scheme='16A->16B', stage=s, model=m, **{k: v[k] for k in v if k != 'lam'}))
    print(cat, 'done', flush=True)

R = pd.DataFrame(rows); R.to_csv(f'results/forecast_scores_seed{SEED}.csv', index=False)
pd.concat(series).to_csv(f'results/forecast_series_seed{SEED}.csv', index=False)
pd.set_option('display.width', 250)
for sch in ['LOSO', 'FC']:
    t = R[R.scheme == sch].groupby(['catalog', 'model']).agg(LL=('LL', 'sum'), IG_mean=('IG', 'mean'),
        IG_med=('IG', 'median'), Ntest_pass=('Ntest_p', lambda p: (p > 0.025).mean()), rmse=('rmse', 'mean')).round(3)
    print(f'\n=== {sch} ==='); print(t.to_string())
t = R[R.scheme == '16A->16B'][['catalog', 'stage', 'model', 'N_obs', 'N_pred']].round(1)
print('\n=== 16A -> 16B transfer (total counts) ==='); print(t.pivot_table(index=['catalog', 'stage', 'N_obs'], columns='model', values='N_pred').round(1).to_string())
# paired tests on LOSO IG (both catalogs pooled, 18 stage-forecasts)
L = R[R.scheme == 'LOSO'].pivot_table(index=['catalog', 'stage'], columns='model', values='IG')
print('\nPaired Wilcoxon (LOSO, IG per stage, 2 catalogs x 9 stages):')
for a, b in [('SI-mem', 'SI'), ('SI-mem', 'GBM'), ('Hybrid', 'SI-mem'), ('Hybrid', 'GBM'), ('SI-mem', 'RF'), ('GBM+AR', 'SI-mem+AR'), ('SI-mem+AR', 'SI-mem')]:
    dlt = L[a] - L[b]
    print(f'  {a:10s} - {b:10s}: median diff {dlt.median():+.3f} bits/event, wins {int((dlt > 0).sum())}/{len(dlt)}, p={wilcoxon(dlt).pvalue:.4f}')
