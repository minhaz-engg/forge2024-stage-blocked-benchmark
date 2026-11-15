import pandas as pd, numpy as np
inj = pd.read_csv('data/injection_2024_stages.csv')
cats = {'Niemz': pd.read_csv('data/FORGE2024_Niemz_surface_compact.csv'), 'Zhu': pd.read_csv('data/FORGE2024_Zhu_phasepick_compact.csv')}
# 1-min injection rate series on a global grid (offset applied later by shifting)
t = np.arange(0, 16*86400, 60.0)
q = np.zeros_like(t)
for s, g in inj.groupby('stage'):
    g = g.dropna(subset=['q_bpm'])
    qi = np.interp(t, g.t_s.values, g.q_bpm.clip(0, 100).values, left=0, right=0)
    q = np.maximum(q, qi)
for name, c in cats.items():
    n, _ = np.histogram(c.t_s.values, bins=np.append(t, t[-1]+60))
    print(name, 'events', len(c))
    for off_h in [5, 6, 7]:
        # injection file was built with +6 h; shift by (off_h-6)
        sh = int((off_h - 6) * 60)
        qs = np.roll(q, sh)
        # fraction of events within 0..30 min after pumping >5 bpm vs. 30 min before pumping starts
        pumping = qs > 5
        print(f'  UTC offset {off_h}h: frac events while pumping = {n[pumping].sum()/n.sum():.3f}; '
              f'corr(q, count, lag 0..30min max) = {max(np.corrcoef(np.roll(qs, L), n)[0,1] for L in range(0, 31)):.3f}')
    # per-stage onset: first event after pump start (offset 6)
for s, g in inj.groupby('stage', sort=False):
    g = g[g.q_bpm > 5]
    if len(g)==0: continue
    t_on, t_off = g.t_s.min(), g.t_s.max()
    out = []
    for name, c in cats.items():
        tt = c.t_s.values
        out.append(f"{name}: n[-2h,0]={((tt>t_on-7200)&(tt<t_on)).sum():3d} n[0,end]={((tt>=t_on)&(tt<=t_off)).sum():4d} first_after={(tt[tt>=t_on].min()-t_on)/60 if (tt>=t_on).any() else np.nan:6.1f}min")
    print(f'{s:7s} pump {t_on/3600:7.2f}-{t_off/3600:7.2f} h  ' + ' | '.join(out))
