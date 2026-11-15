import pandas as pd, numpy as np, json
F = {}
cs = pd.read_csv('results/catalog_summary.csv').set_index('catalog'); si = pd.read_csv('results/seismogenic_index.csv'); rt = pd.read_csv('results/diffusivity_rt.csv')
st = pd.read_csv('results/stage_table.csv')
F['V16A_total_m3'] = st[st.well=='16A'].V.sum(); F['V16B_total_m3'] = st[st.well=='16B'].V.sum()
F['V16A_range'] = [st[st.well=='16A'].V.min(), st[st.well=='16A'].V.max()]
F['V16B_range'] = [st[st.well=='16B'].V.min(), st[st.well=='16B'].V.max()]
for c in ['Niemz','Zhu']:
    s = si[(si.catalog==c)&si.stage.str.startswith('16A')]; b = si[(si.catalog==c)&si.stage.str.startswith('16B')]
    r = rt[rt.catalog==c]
    cat = pd.read_csv(f'results/catalog_{c}_filtered.csv')
    F[c] = dict(n_raw=int(cs.loc[c,'n_raw']), n_filt=int(cs.loc[c,'n_filt']), n_16A=int(cat.stage.str.startswith('16A',na=False).sum()),
        n_16B=int(cat.stage.str.startswith('16B',na=False).sum()), mc=cs.loc[c,'mc'], b=round(cs.loc[c,'b'],2), b_ci=[round(cs.loc[c,'b_lo'],2), round(cs.loc[c,'b_hi'],2)],
        n_mc=int(cs.loc[c,'n_mc']), Sig_win_16A_med=round(s.Sigma_win.median(),2), Sig_win_16A_rng=[round(s.Sigma_win.min(),2), round(s.Sigma_win.max(),2)],
        Sig_win_16B=b.Sigma_win.round(2).tolist(), n16B_win_mc=b.n_win_mc.tolist(),
        post_shutin_frac_med=round(s.frac_post_shutin.median(),2), post_rng=[round(s.frac_post_shutin.min(),2), round(s.frac_post_shutin.max(),2)],
        Mmax=s.Mmax.max(), gap_to_mcgarr_min=round((s.Mw_McGarr - s.Mmax).min(),2), gap_max=round((s.Mw_McGarr - s.Mmax).max(),2),
        Mmax_minus_GR=round((s.Mmax-s.Mmax_GR_expected).median(),2), Mmax_minus_GR_rng=[round((s.Mmax-s.Mmax_GR_expected).min(),2), round((s.Mmax-s.Mmax_GR_expected).max(),2)],
        D95_med=round(r.D95.median(),2), D95_rng=[round(r.D95.min(),2), round(r.D95.max(),2)], growth_med=round(r.growth_exponent.median(),2))
L = pd.read_csv('results/leakage_magnitude_regression.csv')
F['leak'] = L.round(3).to_dict('records')
b = pd.read_csv('results/binned_Niemz.csv'); F['n_bins_16A'] = int(b.stage.str.startswith('16A').sum())
print(json.dumps(F, indent=1, default=float))
json.dump(F, open('results/facts.json','w'), indent=1, default=float)
