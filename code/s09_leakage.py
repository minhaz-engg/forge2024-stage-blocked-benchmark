"""Re-test of the v9 set-up: per-event magnitude regression from location/time/injection features.
Random 80/20 split (as in v9) vs. stage-blocked (leave-one-stage-out) vs. the trivial mean predictor."""
import numpy as np, pandas as pd, zp, warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')
inj = zp.load_injection(); st = pd.read_csv('results/stage_table.csv'); S16A = st[st.well=='16A'].stage.tolist()
tg = np.arange(st.t_on.min() - 3600, st[st.well=='16A'].t_end.max(), 10.0)
q = np.zeros_like(tg); P = np.full_like(tg, np.nan)
for s, g in inj.groupby('stage'):
    m = (tg >= g.t_s.min()) & (tg <= g.t_s.max())
    q[m] = np.interp(tg[m], g.t_s, g.q); P[m] = np.interp(tg[m], g.t_s, g.P)
P = pd.Series(P).ffill().fillna(0).values; Ps = savgol_filter(P, 51, 3)
Vc = np.cumsum(q) * 10.0
def feats(c):
    i = np.clip(np.searchsorted(tg, c.t_s.values), 0, len(tg) - 1)
    ref = c[['x','y','z']].median().values
    X = pd.DataFrame(dict(x=c.x - ref[0], y=c.y - ref[1], z=c.z - ref[2], t=c.t_s / 3600, P=Ps[i], V=Vc[i], q=q[i], power=Ps[i] * q[i]))
    X['r'] = np.sqrt(X.x**2 + X.y**2 + X.z**2); return X
MODELS = {'Mean': None,
          'RF (v9 best)': lambda: RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=2, random_state=0, n_jobs=1),
          'GBR (v9 best)': lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0),
          'kNN on (x,y,z,t) only': lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(15))}
cs = pd.read_csv('results/catalog_summary.csv').set_index('catalog')
rows = []
for cat, cut in [('Niemz','all'), ('Zhu','all'), ('Niemz','M>=Mc'), ('Zhu','M>=Mc')]:
    c = pd.read_csv(f'results/catalog_{cat}_filtered.csv'); c = c[c.stage.isin(S16A)].reset_index(drop=True)
    if cut != 'all': c = c[c.M >= cs.loc[cat,'mc'] - 1e-9].reset_index(drop=True)
    X = feats(c); y = c.M.values; grp = c.stage.values
    for mname, mk in MODELS.items():
        cols = ['x','y','z','t'] if 'kNN' in mname else list(X.columns)
        # random split, 10 repeats
        rng = np.random.default_rng(0); r2r, rm = [], []
        for rep in range(10):
            idx = rng.permutation(len(y)); te, tr = idx[:len(y)//5], idx[len(y)//5:]
            pred = np.full(len(te), y[tr].mean()) if mk is None else mk().fit(X.iloc[tr][cols], y[tr]).predict(X.iloc[te][cols])
            r2r.append(r2_score(y[te], pred)); rm.append(np.sqrt(mean_squared_error(y[te], pred)))
        # leave-one-stage-out
        pred = np.zeros(len(y))
        for s in S16A:
            te = grp == s; tr = ~te
            pred[te] = y[tr].mean() if mk is None else mk().fit(X[tr][cols], y[tr]).predict(X[te][cols])
        r2s = [r2_score(y[grp == s], pred[grp == s]) for s in S16A]
        rows.append(dict(catalog=cat, subset=cut, model=mname, n=len(y), R2_random_mean=np.mean(r2r), R2_random_sd=np.std(r2r),
                         RMSE_random=np.mean(rm), R2_LOSO_pooled=r2_score(y, pred), R2_LOSO_stage_median=np.median(r2s),
                         RMSE_LOSO=np.sqrt(mean_squared_error(y, pred))))
        pass
R = pd.DataFrame(rows); R.to_csv('results/leakage_magnitude_regression.csv', index=False)
print(R.round(3).to_string())
