import numpy as np, pandas as pd, zp
inj = zp.load_injection(); st = zp.stage_table(inj)
print(st.assign(t_on_h=st.t_on/3600, t_off_h=st.t_off/3600, t_end_h=st.t_end/3600, V_m3=st.V.round(0))[['stage','t_on_h','t_off_h','t_end_h','V_m3','qmax','Pmax']].round(2).to_string())
st.to_csv('results/stage_table.csv', index=False)
res = []
for name in ['Niemz', 'Zhu']:
    c0 = zp.load_catalog(name)
    a16 = c0[(c0.t_s > st.t_on.iloc[0]) & (c0.t_s < st[st.well=='16A'].t_end.max())]
    ref = a16[['x','y','z']].median().values
    c, _ = zp.spatial_filter(c0, ref)
    c = zp.assign_stage(c, st)
    print(f'\n{name}: {len(c0)} -> {len(c)} after spatial filter; ref (x,y,z)={ref.round(0)}; in 16A windows: {c.stage.str.startswith("16A").sum()}  16B: {c.stage.str.startswith("16B").sum()}  unassigned: {c.stage.isna().sum()}')
    M = c[c.stage.notna() & c.stage.str.startswith('16A', na=False)].M.values
    mc1 = zp.mc_maxc(M); mc2, tab = zp.mc_gof(M)
    mc = max(mc1, mc2)
    b, sb, n = zp.b_aki(M, mc)
    # bootstrap
    rng = np.random.default_rng(0); bb = []
    for _ in range(1000):
        mm = rng.choice(M, len(M)); bb.append(zp.b_aki(mm, mc)[0])
    print(f'  Mc(MAXC+0.2)={mc1}  Mc(GOF90)={mc2}  -> Mc={mc}; b={b:.3f} +/- {sb:.3f} (Shi-Bolt), boot 95% [{np.percentile(bb,2.5):.3f},{np.percentile(bb,97.5):.3f}], N>=Mc={n}')
    res.append(dict(catalog=name, n_raw=len(c0), n_filt=len(c), mc_maxc=mc1, mc_gof=mc2, mc=mc, b=b, b_sb=sb, b_lo=np.percentile(bb,2.5), b_hi=np.percentile(bb,97.5), n_mc=n, ref_x=ref[0], ref_y=ref[1], ref_z=ref[2]))
    c.to_csv(f'results/catalog_{name}_filtered.csv', index=False)
pd.DataFrame(res).to_csv('results/catalog_summary.csv', index=False)
