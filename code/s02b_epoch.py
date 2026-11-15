import pandas as pd, numpy as np
inj = pd.read_csv('data/injection_2024_stages.csv')
c = pd.concat([pd.read_csv('data/FORGE2024_Niemz_surface_compact.csv').assign(cat='N'), pd.read_csv('data/FORGE2024_Zhu_phasepick_compact.csv').assign(cat='Z')])
edges = np.arange(-180, 181, 10)*60
for key, thr in [('start', 20)]:
    for catn in ['N','Z']:
        tt = c[c.cat==catn].t_s.values
        H_on = np.zeros(len(edges)-1); H_off = np.zeros(len(edges)-1)
        for s, g in inj[inj.well=='16A'].groupby('stage'):
            g = g.dropna(subset=['q_bpm']); hi = g[g.q_bpm > thr]
            if len(hi)==0: continue
            t_on, t_off = hi.t_s.min(), hi.t_s.max()
            H_on += np.histogram(tt - t_on, edges)[0]; H_off += np.histogram(tt - t_off, edges)[0]
        print(catn, 'stacked counts rel. to first q>20 bpm (10-min bins, -180..+180 min):')
        print('  ', ' '.join(f'{int(x)}' for x in H_on))
        print(catn, 'rel. to last q>20 bpm (shut-in):')
        print('  ', ' '.join(f'{int(x)}' for x in H_off))
