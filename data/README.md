# Data

All inputs are public. Download them into `data/raw/` as follows (about 60 MB in total).

| Put here | Download from | Licence |
|---|---|---|
| `data/raw/FORGE2024_Niemz2024_surface_catalog.csv` | Niemz et al. (2024), Zenodo, https://doi.org/10.5281/zenodo.12817881 (the event catalogue CSV, renamed) | see record |
| `data/raw/FORGE2024_Zhu2026_phasepick_catalog.csv` | Zhu et al. (2026), Geothermal Data Repository, https://gdr.openei.org/submissions/1817 (the event catalogue CSV, renamed) | see record |
| `data/raw/GDR_1611/16A Frac Data/*.xlsx` and `data/raw/GDR_1611/16B Frac Data/*.xlsx` | England (2024), Geothermal Data Repository, https://doi.org/10.15121/2371032 (unzip `16A Frac Data.zip` and `16B Frac Data.zip`) | CC BY 4.0 |

`code/s00_prepare_catalogues.py` converts the two catalogues to the compact tables used by all later steps (time in seconds since 2024-04-03 00:00 UTC; east/north/depth relative to the 16A wellhead). Different file names can be passed as arguments: `python code/s00_prepare_catalogues.py <niemz.csv> <zhu.csv>`.

`code/s01_injection.py` reads the Calfrac stage files from `data/raw/GDR_1611` (or from the folder in the environment variable `FORGE_2024_DIR`) and converts local time to UTC with UTC = local + 6 h, the offset tested in `s02_timecheck.py`.
