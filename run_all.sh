#!/usr/bin/env bash
# Reproduce all results from the public data. Run from the repository root.
# Expects the raw files described in data/README.md. Total run time: about 1 h on a laptop (s06-s09 dominate).
set -euo pipefail
mkdir -p results figs
python code/s00_prepare_catalogues.py
python code/s01_injection.py
for s in s02_timecheck s02b_epoch s03_qc s04_catalog s05_si_rt s06_forecast s07_adaptive s08_gbm_adaptive s09_leakage s10_summary s11_facts; do
  echo "== $s"; python code/$s.py
done
python code/make_figures.py
