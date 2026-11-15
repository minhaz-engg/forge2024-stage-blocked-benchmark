"""Core functions for the reworked zeroth paper (Utah FORGE 2024 stimulation).
Times: seconds since 2024-04-03 00:00 UTC. Lengths: m. Volumes: m^3. Rates: m^3/s."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

BBL = 0.158987294928          # m^3 per barrel
BPM = BBL / 60.0              # m^3/s per bbl/min
DT = 600.0                    # forecast bin (s)

# ---------------------------------------------------------------- data
def load_injection(path='data/injection_2024_stages.csv'):
    inj = pd.read_csv(path)
    inj['q'] = inj.q_bpm.clip(0, 100).fillna(0) * BPM
    inj['P'] = inj.P_psi * 6894.757e-6          # MPa (surface treating pressure)
    return inj

def stage_table(inj, q_on_bpm=5.0):
    rows = []
    for s, g in inj.groupby('stage'):
        on = g[g.q_bpm > q_on_bpm]
        rows.append(dict(stage=s, well=g.well.iloc[0], t_on=on.t_s.min(), t_off=on.t_s.max(),
                         V=np.trapezoid(g.q.values, g.t_s.values),
                         qmax=g.q.max(), Pmax=g.P.max()))
    st = pd.DataFrame(rows).sort_values('t_on').reset_index(drop=True)
    # analysis window: pump-on to next stage pump-on (16A); 16B: pump-on + 24 h
    nxt = st.t_on.shift(-1).fillna(st.t_on.iloc[-1] + 86400).values
    st['t_end'] = np.minimum(nxt, st.t_off.values + 24 * 3600)
    return st

def load_catalog(name):
    f = {'Niemz': 'data/FORGE2024_Niemz_surface_compact.csv', 'Zhu': 'data/FORGE2024_Zhu_phasepick_compact.csv'}[name]
    c = pd.read_csv(f)
    c.columns = ['t_s', 'x', 'y', 'z', 'M']
    return c.sort_values('t_s').reset_index(drop=True)

def spatial_filter(c, ref=None, rmax=1500.0, zlim=(1500., 3600.)):
    if ref is None:
        ref = c[['x', 'y', 'z']].median().values
    r = np.sqrt((c.x - ref[0])**2 + (c.y - ref[1])**2)
    keep = (r < rmax) & (c.z > zlim[0]) & (c.z < zlim[1])
    return c[keep].reset_index(drop=True), ref

def assign_stage(c, st):
    lab = np.full(len(c), None, dtype=object)
    for _, s in st.iterrows():
        m = (c.t_s >= s.t_on) & (c.t_s < s.t_end)
        lab[m.values] = s.stage
    c = c.copy(); c['stage'] = lab
    return c

# ---------------------------------------------------------------- magnitudes
def mc_maxc(M, dm=0.1, corr=0.2):
    e = np.arange(np.floor(M.min() / dm) * dm, M.max() + dm, dm)
    h, e = np.histogram(M, e)
    return round(e[np.argmax(h)] + dm / 2 + corr, 2)

def b_aki(M, mc, dm=0.1):
    m = M[M >= mc - 1e-9]
    b = np.log10(np.e) / (m.mean() - (mc - dm / 2))
    sb = 2.3 * b**2 * np.sqrt(((m - m.mean())**2).sum() / (len(m) * (len(m) - 1)))  # Shi & Bolt (1982)
    return b, sb, len(m)

def mc_gof(M, dm=0.1, level=90):
    """Wiemer & Wyss (2000) goodness-of-fit Mc: lowest Mc with residual R <= 100-level %."""
    out = []
    for mc in np.arange(np.round(np.quantile(M, 0.02), 1), np.quantile(M, 0.8), dm):
        m = M[M >= mc - 1e-9]
        if len(m) < 50: break
        b, _, n = b_aki(M, mc, dm)
        a = np.log10(n) + b * mc
        bins = np.arange(mc, m.max() + dm, dm)
        obs = np.array([(m >= x - 1e-9).sum() for x in bins])
        pred = 10**(a - b * bins)
        R = 100 * np.abs(obs - pred).sum() / obs.sum()
        out.append((round(mc, 2), b, R))
    out = pd.DataFrame(out, columns=['mc', 'b', 'R'])
    ok = out[out.R <= 100 - level]
    return (ok.mc.iloc[0] if len(ok) else np.nan), out

# ---------------------------------------------------------------- binned series
def binned_series(inj, st, c_mc, stages, dt=DT):
    """One row per dt-bin inside each stage window."""
    rows = []
    for s in stages:
        S = st[st.stage == s].iloc[0]
        g = inj[inj.stage == s]
        edges = np.arange(S.t_on, S.t_end + dt, dt)
        tt = g.t_s.values
        # fine grid for injection (1 s data) -> per-bin volume & mean P
        tf = np.arange(S.t_on, S.t_end, 10.0)
        qf = np.interp(tf, tt, g.q.values, left=0, right=0)
        Pf = np.interp(tf, tt, g.P.values, left=np.nan, right=np.nan)
        k = ((tf - S.t_on) // dt).astype(int)
        nb = len(edges) - 1
        Q = np.bincount(k, qf * 10.0, minlength=nb)[:nb]
        Pm = pd.Series(Pf).groupby(k).mean().reindex(range(nb)).values
        n = np.histogram(c_mc.t_s.values, edges)[0]
        df = pd.DataFrame(dict(stage=s, k=np.arange(nb), t0=edges[:-1], n=n, Q=Q, P=Pm))
        df['tsince_on'] = (df.t0 - S.t_on) / 3600
        df['tsince_off'] = np.clip((df.t0 - S.t_off) / 3600, 0, None)
        df['pumping'] = (df.t0 < S.t_off).astype(int)
        df['Vcum'] = df.Q.cumsum()
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

def add_features(df):
    df = df.copy()
    g = df.groupby('stage', sort=False)
    df['qbin'] = df.Q / DT
    for L in [1, 2, 3, 6, 12]:
        df[f'Q_lag{L}'] = g.Q.shift(L).fillna(0)
    df['Q_roll6'] = g.Q.transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['Q_roll18'] = g.Q.transform(lambda x: x.rolling(18, min_periods=1).mean())
    df['P'] = g.P.transform(lambda x: x.ffill().fillna(0))
    df['dP'] = g.P.diff().fillna(0)
    df['P_roll6'] = g.P.transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['Pmax_sofar'] = g.P.cummax()
    return df

FEATS = ['Q', 'P', 'dP', 'Vcum', 'tsince_on', 'tsince_off', 'pumping', 'Q_lag1', 'Q_lag2', 'Q_lag3', 'Q_lag6',
         'Q_lag12', 'Q_roll6', 'Q_roll18', 'P_roll6', 'Pmax_sofar']

# ---------------------------------------------------------------- likelihoods
def pois_ll(n, lam):
    lam = np.maximum(lam, 1e-12)
    return (n * np.log(lam) - lam - gammaln(n + 1)).sum()

def kernel_conv(Q_by_stage, tau_bins):
    """causal exponential memory: x_k = sum_j Q_{k-j} w_j, w_j = (1-a) a^j, a = exp(-1/tau)."""
    a = np.exp(-1.0 / max(tau_bins, 1e-6))
    out = []
    for Q in Q_by_stage:
        x = np.zeros_like(Q, dtype=float); acc = 0.0
        for i, v in enumerate(Q):
            acc = a * acc + (1 - a) * v
            x[i] = acc
        out.append(x)
    return out

class NullModel:
    def fit(self, d): self.r = d.n.sum() / len(d); return self
    def predict(self, d): return np.full(len(d), self.r)

class SIModel:
    """lambda = A*Q + c  (seismogenic-index rate law, Shapiro et al. 2010, plus background)."""
    def fit(self, d):
        n, Q = d.n.values, d.Q.values
        f = lambda p: -pois_ll(n, np.exp(p[0]) * Q + np.exp(p[1]))
        r = minimize(f, [np.log(n.sum() / Q.sum() + 1e-9), np.log(0.1)], method='Nelder-Mead')
        self.A, self.c = np.exp(r.x); return self
    def predict(self, d): return self.A * d.Q.values + self.c

class SIMemModel:
    """lambda = A*(Q * h_tau) + c ; h_tau = causal exponential kernel (diffusive delay / post-shut-in decay)."""
    def fit(self, d):
        self.stages = None
        parts = [g.Q.values for _, g in d.groupby('stage', sort=False)]
        n = np.concatenate([g.n.values for _, g in d.groupby('stage', sort=False)])
        def f(p):
            x = np.concatenate(kernel_conv(parts, np.exp(p[2])))
            return -pois_ll(n, np.exp(p[0]) * x + np.exp(p[1]))
        best = None
        for t0 in [1, 3, 6, 12, 24]:
            r = minimize(f, [np.log(n.sum() / d.Q.sum() + 1e-9), np.log(0.1), np.log(t0)], method='Nelder-Mead',
                         options=dict(maxiter=2000, xatol=1e-4, fatol=1e-4))
            if best is None or r.fun < best.fun: best = r
        self.A, self.c, self.tau = np.exp(best.x); return self
    def transform(self, d):
        idx = [g.index for _, g in d.groupby('stage', sort=False)]
        parts = [d.loc[i, 'Q'].values for i in idx]
        x = np.concatenate(kernel_conv(parts, self.tau))
        return pd.Series(x, index=np.concatenate(idx)).reindex(d.index).values
    def predict(self, d): return self.A * self.transform(d) + self.c

def info_gain_bits(n, lam_model, lam_ref):
    N = max(n.sum(), 1)
    return (pois_ll(n, lam_model) - pois_ll(n, lam_ref)) / N / np.log(2)
