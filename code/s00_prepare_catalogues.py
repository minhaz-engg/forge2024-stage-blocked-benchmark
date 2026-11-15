"""Convert the two public catalogue files into the compact tables used by all later steps.

Inputs (download from the sources and pass their paths, or keep the default names in data/raw/):
  Niemz et al. (2024b), https://doi.org/10.5281/zenodo.12817881   -> columns time(UTC), lat, lon, depth[m_local], magnitude
  Zhu et al. (2026),  https://gdr.openei.org/submissions/1817     -> columns time, latitude, longitude, depth_km, magnitude
Output columns: t_s (s since 2024-04-03 00:00 UTC), x_m / y_m (east / north of the 16A wellhead, m),
zloc_m (depth below the 16A wellhead, m), magnitude (ML for Niemz, M for Zhu).
"""
import sys, numpy as np, pandas as pd
LAT0, LON0, R = 38.50402147, -112.8963897, 6371008.8   # 16A wellhead; mean Earth radius
T0 = pd.Timestamp('2024-04-03 00:00:00')
rnd = lambda v: np.floor(np.asarray(v, float) + 0.5).astype(int)

def local_xy(lat, lon):
    x = np.radians(lon - LON0) * R * np.cos(np.radians(LAT0))
    y = np.radians(lat - LAT0) * R
    return x, y

def niemz(path):
    c = pd.read_csv(path)
    t = (pd.to_datetime(c['time(UTC)']) - T0).dt.total_seconds()
    x, y = local_xy(c['lat'].values, c['lon'].values)
    return pd.DataFrame({'t_s': rnd(t), 'x_m': rnd(x), 'y_m': rnd(y), 'zloc_m': rnd(c['depth[m_local]']), 'ML': c['magnitude'].round(2)})

def zhu(path):
    c = pd.read_csv(path)
    t = (pd.to_datetime(c['time']) - T0).dt.total_seconds()
    x, y = local_xy(c['latitude'].values, c['longitude'].values)
    return pd.DataFrame({'t_s': rnd(t), 'x_m': rnd(x), 'y_m': rnd(y), 'zloc_m': rnd(1000 * c['depth_km'] + 1650), 'M': c['magnitude'].round(2)})

if __name__ == '__main__':
    pn = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/FORGE2024_Niemz2024_surface_catalog.csv'
    pz = sys.argv[2] if len(sys.argv) > 2 else 'data/raw/FORGE2024_Zhu2026_phasepick_catalog.csv'
    n, z = niemz(pn), zhu(pz)
    n.to_csv('data/FORGE2024_Niemz_surface_compact.csv', index=False)
    z.to_csv('data/FORGE2024_Zhu_phasepick_compact.csv', index=False)
    print('Niemz', n.shape, 'Zhu', z.shape)
