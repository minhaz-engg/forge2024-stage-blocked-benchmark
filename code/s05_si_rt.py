import numpy as np, pandas as pd, zp
from scipy.optimize import brentq
st = pd.read_csv('results/stage_table.csv'); cs = pd.read_csv('results/catalog_summary.csv').set_index('catalog')
inj = zp.load_injection()
G = 3.0e10   # shear modulus (Pa), McGarr (2014)
SI, RT = [], []
for name in ['Niemz', 'Zhu']:
    c = pd.read_csv(f'results/catalog_{name}_filtered.csv'); mc, b = cs.loc[name, 'mc'], cs.loc[name, 'b']
    for _, s in st.iterrows():
        e = c[c.stage == s.stage]; ep = e[e.t_s <= s.t_off]
        n_inj = (ep.M >= mc).sum(); n_win = (e.M >= mc).sum()
        Sig = np.log10(n_inj) - np.log10(s.V) + b * mc if n_inj > 0 else np.nan
        Sig_w = np.log10(n_win) - np.log10(s.V) + b * mc if n_win > 0 else np.nan
        Mmax = e.M.max() if len(e) else np.nan
        Mexp = mc + np.log10(n_win) / b if n_win > 0 else np.nan            # expected largest from GR
        Mw_mcgarr = (2/3) * (np.log10(G * s.V) - 9.1)
        SI.append(dict(catalog=name, stage=s.stage, V_m3=s.V, n_all=len(e), n_inj_mc=n_inj, n_win_mc=n_win,
                       Sigma_inj=Sig, Sigma_win=Sig_w, frac_post_shutin=1 - n_inj / n_win if n_win else np.nan,
                       Mmax=Mmax, Mmax_GR_expected=Mexp, Mw_McGarr=Mw_mcgarr))
        # --- r-t analysis (16A stages with >= 40 events during window)
        if s.well != '16A' or len(e) < 40: continue
        e = e.sort_values('t_s'); t = (e.t_s - s.t_on).values
        x0 = e.iloc[:10][['x', 'y', 'z']].median().values           # injection-point proxy: first 10 events
        r = np.sqrt(((e[['x', 'y', 'z']].values - x0)**2).sum(1))
        sel = t > 0
        tt, rr = t[sel][10:], r[sel][10:]                             # exclude the 10 events defining x0
        # envelope diffusivity: D such that 95% / 90% of events lie inside r_t = sqrt(4 pi D t)
        Dq = {}
        for qq in [0.90, 0.95]:
            f = lambda lD: np.mean(rr <= np.sqrt(4*np.pi*10**lD*tt)) - qq
            Dq[qq] = 10**brentq(f, -6, 3)
        # power-law growth of the cloud: r_90(t) in log-time bins during pumping
        tb = np.logspace(np.log10(max(tt.min(), 120)), np.log10(min(tt.max(), s.t_off - s.t_on)), 9)
        r90 = []; tm = []
        for a, bb in zip(tb[:-1], tb[1:]):
            m = (tt >= a) & (tt < bb)
            if m.sum() >= 5: r90.append(np.percentile(rr[:np.searchsorted(tt, bb)], 90)); tm.append(np.sqrt(a*bb))
        nexp = np.polyfit(np.log10(tm), np.log10(r90), 1)[0] if len(tm) >= 4 else np.nan
        # bootstrap D95
        rng = np.random.default_rng(1); Db = []
        for _ in range(300):
            i = rng.integers(0, len(tt), len(tt))
            f = lambda lD: np.mean(rr[i] <= np.sqrt(4*np.pi*10**lD*tt[i])) - 0.95
            Db.append(10**brentq(f, -6, 3))
        RT.append(dict(catalog=name, stage=s.stage, n=len(tt), D90=Dq[0.90], D95=Dq[0.95],
                       D95_lo=np.percentile(Db, 2.5), D95_hi=np.percentile(Db, 97.5), growth_exponent=nexp,
                       r_med=np.median(rr), x0=x0[0], y0=x0[1], z0=x0[2]))
si = pd.DataFrame(SI); rt = pd.DataFrame(RT)
si.to_csv('results/seismogenic_index.csv', index=False); rt.to_csv('results/diffusivity_rt.csv', index=False)
pd.set_option('display.width', 250)
print(si.round(3).to_string()); print(); print(rt.round(4).to_string())
for name in ['Niemz','Zhu']:
    s = si[(si.catalog==name) & si.stage.str.startswith('16A')]
    print(name, '16A Sigma_inj median', s.Sigma_inj.median().round(2), 'range', s.Sigma_inj.min().round(2), s.Sigma_inj.max().round(2),
          ' pooled Sigma (all 16A):', round(np.log10(s.n_inj_mc.sum()) - np.log10(s.V_m3.sum()) + cs.loc[name,'b']*cs.loc[name,'mc'], 2))
    r = rt[rt.catalog==name]; print('   D95 median', r.D95.median().round(4), 'range', r.D95.min().round(4), r.D95.max().round(4), ' growth exp median', r.growth_exponent.median().round(2))
