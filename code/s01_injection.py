"""Parse Calfrac stage treatment files -> one injection table (UTC seconds since 2024-04-03 00:00 UTC).
Local time offset is a parameter; it is checked against seismicity onset in s02."""
import pandas as pd, numpy as np, glob, re, sys
UTC_OFFSET_H = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0   # MDT = UTC-6 (April 2024)
import os
base = os.environ.get('FORGE_2024_DIR', os.path.join('data', 'raw', 'GDR_1611')) + os.sep   # folder holding '16A Frac Data' and '16B Frac Data'
fs = sorted(glob.glob(base + '1*Frac Data/*.xlsx'))
T0 = pd.Timestamp('2024-04-03 00:00:00')
rows = []
for f in fs:
    name = f.split('/')[-1]
    well = '16A' if name.startswith('16A') else '16B'
    stage = re.search(r'Stage (\w+)_', name).group(1)
    df = pd.read_excel(f, header=None, skiprows=5)
    d = pd.to_datetime(df[0]).dt.normalize()
    tt = pd.to_timedelta(df[1].astype(str))
    ts = d + tt + pd.Timedelta(hours=UTC_OFFSET_H)
    t = (ts - T0).dt.total_seconds()
    P = pd.to_numeric(df[2], errors='coerce')
    q = pd.to_numeric(df[6], errors='coerce')      # clean rate bpm
    V = pd.to_numeric(df[7], errors='coerce')      # clean stage total bbl
    out = pd.DataFrame({'t_s': t, 'P_psi': P, 'q_bpm': q, 'Vstage_bbl': V})
    out['well'] = well; out['stage'] = f'{well}-{stage}'
    rows.append(out)
    print(f'{well}-{stage:>3}: {len(out):6d} rows  start {ts.iloc[0]}  dur {(t.iloc[-1]-t.iloc[0])/3600:5.2f} h  '
          f'Vmax {V.max():7.0f} bbl  qmax {q.max():5.1f} bpm  Pmax {P.max():6.0f} psi  dt~{np.median(np.diff(t)):.0f}s')
inj = pd.concat(rows).sort_values('t_s').reset_index(drop=True)
inj.to_csv('data/injection_2024_stages.csv', index=False)
print(inj.shape, 'total clean volume bbl', inj.groupby('stage').Vstage_bbl.max().sum())
