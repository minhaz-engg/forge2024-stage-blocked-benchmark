"""All manuscript figures (300 dpi PNG). Run after s04-s09."""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, zp, os
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'legend.frameon': False, 'savefig.dpi': 300,
                     'lines.linewidth': 1.4})
C1, C2, C3, C4, C5, C6, C7, C8 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'
GREY, INK, INK2 = '#9a9993', '#0b0b0b', '#52514e'
CATC = {'Niemz': C1, 'Zhu': C2}
os.makedirs('figs', exist_ok=True)
st = pd.read_csv('results/stage_table.csv'); cs = pd.read_csv('results/catalog_summary.csv').set_index('catalog')
inj = zp.load_injection(); S16A = st[st.well == '16A'].stage.tolist(); S16B = st[st.well == '16B'].stage.tolist()
cats = {c: pd.read_csv(f'results/catalog_{c}_filtered.csv') for c in ['Niemz', 'Zhu']}
def lab(ax, s): ax.text(-0.09, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom')
import os; os.makedirs('figs', exist_ok=True)
def save(f, name): f.savefig(f'figs/{name}.png', bbox_inches='tight'); plt.close(f); print('saved', name)

# ---------------- Fig 2: overview of the 16A stimulation (time series)
f, ax = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1.2]))
t0h = st.t_on.min() / 3600 - 2; t1h = st[st.well == '16A'].t_end.max() / 3600
for s, g in inj[inj.well == '16A'].groupby('stage'):
    ax[0].plot(g.t_s / 3600, np.minimum(g.q * 60, 14.0), color=C1, lw=0.7)
    ax[1].plot(g.t_s / 3600, g.P, color=INK2, lw=0.6)
for _, s in st[st.well == '16A'].iterrows():
    for a in ax: a.axvspan(s.t_on / 3600, s.t_off / 3600, color='#f1f0ec', zorder=0, lw=0)
    ax[0].text((s.t_on + s.t_off) / 7200, 14.2, s.stage.replace('16A-', ''), ha='center', fontsize=7, color=INK2)
ax[0].set_ylabel('Injection rate\n(m$^3$/min)'); ax[0].set_ylim(0, 15.5)
ax[1].set_ylabel('Surface treating\npressure (MPa)')
for c in ['Niemz', 'Zhu']:
    cc = cats[c]; m = cc[(cc.M >= cs.loc[c, 'mc'])]
    e = np.arange(t0h, t1h + 1, 1.0); h = np.histogram(m.t_s / 3600, e)[0]
    ax[2].step(e[:-1], h, where='post', color=CATC[c], lw=1.1, label=f'{c} catalogue (M ≥ {cs.loc[c, "mc"]:.2f})')
ax[2].set_ylabel('Events per hour\n(M ≥ Mc)'); ax[2].legend(loc='upper left', fontsize=7.5)
ax[2].set_xlabel('Hours since 2024-04-03 00:00 UTC'); ax[2].set_xlim(t0h, 126)
for a, s in zip(ax, 'abc'): lab(a, s)
save(f, 'Fig02_overview_16A')

# ---------------- Fig 3: hypocentres
f, ax = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
for j, c in enumerate(['Niemz', 'Zhu']):
    cc = cats[c][cats[c].stage.isin(S16A + S16B)]; tt = cc.t_s / 3600
    sc = ax[0, j].scatter(cc.x, cc.y, c=tt, s=2.5, cmap='Blues', vmin=-20, vmax=130, lw=0)
    ax[0, j].set_xlim(200, 1800); ax[0, j].set_ylim(-1100, 500); ax[0, j].set_aspect('equal')
    ax[0, j].set_xlabel('Easting rel. 16A wellhead (m)'); ax[0, j].set_ylabel('Northing (m)')
    ax[0, j].set_title(f'{c} catalogue: plan view', fontsize=8.5, loc='left')
    ax[1, j].scatter(cc.x, cc.z, c=tt, s=2.5, cmap='Blues', vmin=-20, vmax=130, lw=0)
    ax[1, j].set_xlim(200, 1800); ax[1, j].set_ylim(3100, 1700); ax[1, j].set_aspect('equal')
    ax[1, j].set_xlabel('Easting (m)'); ax[1, j].set_ylabel('Depth below wellhead (m)')
    ax[1, j].set_title('E–W section', fontsize=8.5, loc='left')
f.colorbar(sc, ax=ax, shrink=0.5, label='Hours since 2024-04-03 00:00 UTC')
for a_, s_ in zip(ax.flat, 'abcd'): lab(a_, s_)
save(f, 'FigS1_hypocentres')

# ---------------- Fig 4: frequency-magnitude
f, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for j, c in enumerate(['Niemz', 'Zhu']):
    M = cats[c][cats[c].stage.isin(S16A)].M.values; mc, b = cs.loc[c, 'mc'], cs.loc[c, 'b']
    bins = np.arange(np.floor(M.min() * 10) / 10, M.max() + 0.1, 0.1)
    cum = np.array([(M >= x - 1e-9).sum() for x in bins]); inc = np.histogram(M, np.append(bins, bins[-1] + 0.1))[0]
    ax[j].semilogy(bins, cum, 'o', ms=3, color=CATC[c], label='cumulative')
    ax[j].semilogy(bins, np.where(inc > 0, inc, np.nan), 's', ms=2.5, mfc='none', color=INK2, label='incremental')
    n = (M >= mc).sum(); a = np.log10(n) + b * mc; xx = np.linspace(mc, M.max(), 50)
    ax[j].semilogy(xx, 10**(a - b * xx), color=INK, lw=1, label=f'b = {b:.2f} [{cs.loc[c,"b_lo"]:.2f}, {cs.loc[c,"b_hi"]:.2f}]')
    ax[j].axvline(mc, color=GREY, ls='--', lw=0.8); ax[j].text(mc + 0.03, 1.5, f'Mc = {mc:.2f}', fontsize=7, color=INK2)
    ax[j].set_xlabel('Magnitude'); ax[j].set_ylabel('Number of events'); ax[j].set_title(f'{c} catalogue (16A stages)', loc='left', fontsize=8.5)
    ax[j].legend(fontsize=7)
for a, s in zip(ax, 'ab'): lab(a, s)
save(f, 'FigS2_FMD')

# ---------------- Fig 5: seismogenic index and Mmax
si = pd.read_csv('results/seismogenic_index.csv'); order = S16A + S16B[:3]
f, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
for k, c in enumerate(['Niemz', 'Zhu']):
    s_ = si[si.catalog == c].set_index('stage').reindex(order); x = np.arange(len(order)) + (k - 0.5) * 0.25
    ax[0].plot(x, s_.Sigma_win, 'o', ms=4, color=CATC[c], label=c)
ax[0].axvline(len(S16A) - 0.5, color=GREY, lw=0.8)
ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels([o.replace('16A-', 'A').replace('16B-', 'B') for o in order], fontsize=7)
ax[0].set_ylabel('Seismogenic index Σ'); ax[0].legend(fontsize=7, loc='lower left'); ax[0].set_xlabel('Stage (A = well 16A, B = well 16B)')
VV = np.logspace(2, 4, 50)
ax[1].semilogx(VV, (2/3) * (np.log10(3e10 * VV) - 9.1), color=INK, lw=1, label='McGarr (2014) bound')
for c in ['Niemz', 'Zhu']:
    s_ = si[(si.catalog == c) & si.stage.isin(S16A)]
    ax[1].semilogx(s_.V_m3, s_.Mmax, 'o', ms=4, color=CATC[c], label=f'{c}')
ax[1].set_xlabel('Injected volume per stage (m$^3$)'); ax[1].set_ylabel('Largest magnitude'); ax[1].legend(fontsize=6.5, loc='upper left'); ax[1].set_ylim(0, 4)
for c in ['Niemz', 'Zhu']:
    s_ = si[(si.catalog == c) & si.stage.isin(S16A)]
    ax[2].plot(s_.Mmax_GR_expected, s_.Mmax, 'o', ms=4, color=CATC[c])
ax[2].plot([0, 2.6], [0, 2.6], color=GREY, lw=0.8); ax[2].set_xlim(0, 2.6); ax[2].set_ylim(0, 2.6)
ax[2].set_xlabel('GR-expected largest, Mc + log$_{10}$N / b'); ax[2].set_ylabel('Observed largest')
for a_, s_ in zip(ax, 'abc'): lab(a_, s_)
save(f, 'Fig05_SI_Mmax')

# ---------------- Fig 6: r-t plots
rt = pd.read_csv('results/diffusivity_rt.csv')
f, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for a, sname in zip(ax, ['16A-3R', '16A-8', '16A-9']):
    S = st[st.stage == sname].iloc[0]; r_ = rt[(rt.catalog == 'Niemz') & (rt.stage == sname)].iloc[0]
    e = cats['Niemz'][cats['Niemz'].stage == sname].sort_values('t_s'); t = (e.t_s - S.t_on) / 3600
    r = np.sqrt((e.x - r_.x0)**2 + (e.y - r_.y0)**2 + (e.z - r_.z0)**2)
    a.plot(t, r, 'o', ms=2.2, color=C1, alpha=0.7)
    tt = np.linspace(0.01, t.max(), 200)
    a.plot(tt, np.sqrt(4 * np.pi * r_.D95 * tt * 3600), color=INK, lw=1, label=f'D$_{{95}}$ = {r_.D95:.2f} m$^2$/s')
    a.axvline((S.t_off - S.t_on) / 3600, color=GREY, ls='--', lw=0.8)
    a.set_title(sname, loc='left', fontsize=8.5); a.set_xlabel('Time since pump-on (h)'); a.legend(fontsize=7, loc='upper left')
    a.set_ylim(0, 1200)
ax[0].set_ylabel('Distance from early-event centroid (m)')
for a, s in zip(ax, 'abc'): lab(a, s)
save(f, 'Fig06_rt')

# ---------------- Fig 7: example forecasts (LOSO)
ser = pd.read_csv('results/forecast_series_seed0.csv'); bn = pd.read_csv('results/binned_Niemz.csv')
f, ax = plt.subplots(2, 1, figsize=(7.2, 4.8), constrained_layout=True)
for a, sname in zip(ax, ['16A-8', '16A-9']):
    s = ser[(ser.catalog == 'Niemz') & (ser.stage == sname) & (ser.scheme == 'LOSO')]
    obs = s[s.model == 'SI'].n.values; th = np.arange(len(obs)) / 6
    a.bar(th, obs, width=1/6 * 0.85, color='#c9c8c2', label='observed (M ≥ Mc)')
    for m, col in [('SI-mem', C1), ('GBM', C2)]:
        a.plot(th, s[s.model == m].lam.values, color=col, lw=1.3, label=m)
    S = st[st.stage == sname].iloc[0]; a.axvline((S.t_off - S.t_on) / 3600, color=INK2, ls='--', lw=0.8)
    a.text((S.t_off - S.t_on) / 3600 + 0.1, a.get_ylim()[1] * 0.97, 'shut-in', fontsize=7, color=INK2, va='top')
    a.set_ylabel('Events per 10 min'); a.set_title(f'{sname} (held out; Niemz catalogue)', loc='left', fontsize=8.5)
ax[1].set_xlabel('Hours since pump-on'); ax[0].legend(fontsize=7, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.12))
for a in ax: a.set_ylim(0, a.get_ylim()[1] * 1.08)
for a, s in zip(ax, 'ab'): lab(a, s)
save(f, 'Fig07_forecast_examples')

# ---------------- Fig 8: benchmark summary (information gain)
R = pd.read_csv('results/forecast_scores_final.csv')
MODELS = ['SI', 'SI-mem', 'RF', 'GBM', 'Hybrid', 'SI-mem+AR', 'GBM+AR', 'RF-adaptive', 'GBM-adaptive', 'SI-mem-adaptive']
f, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
for a, sch in zip(ax, ['LOSO', 'FC']):
    for k, c in enumerate(['Niemz', 'Zhu']):
        for i, m in enumerate(MODELS):
            v = R[(R.scheme == sch) & (R.catalog == c) & (R.model == m)].IG.values
            y = i + (k - 0.5) * 0.3
            a.plot(v, np.full(len(v), y), 'o', ms=2.2, color=CATC[c], alpha=0.35, lw=0)
            a.plot([np.mean(v)], [y], 'D', ms=5, color=CATC[c], label=c if i == 0 else None)
    a.axvline(0, color=GREY, lw=0.8); a.set_yticks(range(len(MODELS))); a.set_yticklabels(MODELS)
    a.set_xlabel('Information gain over null (bits/event)'); a.set_title({'LOSO': 'Leave-one-stage-out', 'FC': 'Forward chaining'}[sch], loc='left', fontsize=8.5)
    a.axhspan(6.5, 9.5, color='#f1f0ec', zorder=0, lw=0)
ax[0].legend(fontsize=7, loc='lower right'); ax[0].invert_yaxis()
for a, s in zip(ax, 'ab'): lab(a, s)
save(f, 'Fig08_benchmark_IG')

# ---------------- Fig 9: transfer to 16B
T = R[R.scheme == '16A->16B']
f, ax = plt.subplots(figsize=(7.2, 3.0))
ms = ['Null', 'SI', 'SI-mem', 'RF', 'GBM', 'GBM+AR', 'GBM-adaptive', 'SI-mem-adaptive']
xs = np.arange(len(ms))
for k, c in enumerate(['Niemz', 'Zhu']):
    t = T[T.catalog == c]
    for i, m in enumerate(ms):
        v = t[t.model == m]
        ax.plot(np.full(len(v), i + (k - 0.5) * 0.3), v.N_pred, 'o', ms=4, color=CATC[c], label=c if i == 0 else None)
    ax.axhline(t.drop_duplicates('stage').N_obs.max() + 0.5, color=CATC[c], ls=':', lw=1)
ax.set_yscale('log'); ax.set_xticks(xs); ax.set_xticklabels(ms, fontsize=7.5, rotation=20)
ax.set_ylabel('Forecast events in 24 h (M ≥ Mc)'); ax.text(len(ms) - 0.6, 7.5, 'max. observed\n(dotted)', fontsize=7, color=INK2, ha='right')
ax.legend(fontsize=7, loc='upper right'); ax.set_title('Models trained on all nine 16A stages, forecasting the 16B stages', loc='left', fontsize=8.5)
save(f, 'Fig09_transfer_16B')

# ---------------- Fig 10: leakage
L = pd.read_csv('results/leakage_magnitude_regression.csv'); L = L[L.model != 'Mean']
f, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for a, sub in zip(ax, ['all', 'M>=Mc']):
    l = L[L.subset == sub]; lbl = [f'{r.catalog}\n' + {'RF (v9 best)': 'RF', 'GBR (v9 best)': 'GBR', 'kNN on (x,y,z,t) only': 'kNN-xyzt'}[r.model] for _, r in l.iterrows()]; x = np.arange(len(l))
    a.bar(x - 0.18, l.R2_random_mean, 0.34, color=C4, label='random 80/20 split')
    a.bar(x + 0.18, l.R2_LOSO_pooled, 0.34, color=C7, label='stage-blocked (leave-one-stage-out)')
    a.axhline(0, color=INK2, lw=0.8); a.set_xticks(x); a.set_xticklabels(lbl, fontsize=6.5)
    a.set_title({'all': 'All events', 'M>=Mc': 'Events above completeness (M ≥ Mc)'}[sub], loc='left', fontsize=8.5)
ax[0].set_ylabel('R$^2$ of per-event magnitude'); ax[0].legend(fontsize=7, loc='lower left', bbox_to_anchor=(0, 1.1), ncol=2); ax[0].set_ylim(-0.08, 0.34)
for a, s in zip(ax, 'ab'): lab(a, s)
save(f, 'Fig10_leakage')
