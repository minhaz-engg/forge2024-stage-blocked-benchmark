import pandas as pd, numpy as np
inj = pd.read_csv('data/injection_2024_stages.csv')
for s, g in inj.groupby('stage'):
    q = g.q_bpm; dV = g.Vstage_bbl.diff()/g.t_s.diff()*60
    print(f'{s:7s} q>90: {(q>90).sum():5d}  q p99={q.quantile(.99):6.1f}  median(q|q>5)={q[q>5].median():5.1f}  '
          f'V from integral(q)={np.trapezoid(q.fillna(0).clip(0,100), g.t_s)/60:8.0f} vs Vstage max={g.Vstage_bbl.max():8.0f}')
for name in ['Niemz_surface','Zhu_phasepick']:
    c = pd.read_csv(f'data/FORGE2024_{name}_compact.csv'); M = c.iloc[:,4].values
    print(name, len(c), 'M range', M.min(), M.max(), 't range h', c.t_s.min()/3600, c.t_s.max()/3600,
          'z range', c.zloc_m.min(), c.zloc_m.max(), 'x', c.x_m.min(), c.x_m.max(), 'y', c.y_m.min(), c.y_m.max())
    h, e = np.histogram(M, np.arange(-2, 3, 0.1)); print('  hist from -2.0:', ' '.join(map(str, h)))
