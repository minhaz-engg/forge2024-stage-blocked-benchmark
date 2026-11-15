import pandas as pd, numpy as np
from scipy.stats import wilcoxon
R = pd.concat([pd.read_csv('results/forecast_scores_all.csv'), pd.read_csv('results/forecast_scores_ml_adaptive.csv')], ignore_index=True)
R.to_csv('results/forecast_scores_final.csv', index=False)
pd.set_option('display.width', 250)
order = ['Null','SI','SI-mem','RF','GBM','Hybrid','SI-mem+AR','GBM+AR','RF-adaptive','GBM-adaptive','SI-mem-adaptive']
for sch in ['LOSO','FC']:
    t = R[R.scheme==sch].groupby(['model','catalog']).agg(LL=('LL','sum'), IG=('IG','mean'), Npass=('Ntest_p', lambda p:(p>0.025).mean())).unstack('catalog').reindex(order).round(3)
    print(f'\n=== {sch} ==='); print(t.to_string()); t.to_csv(f'results/table_{sch}.csv')
T = R[R.scheme=='16A->16B'].pivot_table(index=['catalog','stage','N_obs'], columns='model', values='N_pred').reindex(columns=order).round(1)
print(T.to_string()); T.to_csv('results/table_transfer.csv')
rows = []
for sch in ['LOSO','FC']:
    L = R[R.scheme == sch].pivot_table(index=['catalog','stage'], columns='model', values='IG')
    for a, b in [('SI-mem','SI'),('SI-mem','GBM'),('SI-mem','RF'),('Hybrid','GBM'),('Hybrid','SI-mem'),('SI-mem+AR','SI-mem'),('GBM+AR','SI-mem+AR'),
                 ('SI-mem-adaptive','SI-mem'),('SI-mem-adaptive','GBM-adaptive'),('SI-mem-adaptive','RF-adaptive'),('SI-mem-adaptive','GBM+AR'),('GBM-adaptive','GBM')]:
        d = (L[a]-L[b]).dropna()
        rows.append(dict(scheme=sch, A=a, B=b, n=len(d), median_diff=d.median(), mean_diff=d.mean(), wins=int((d>0).sum()), p=wilcoxon(d).pvalue))
W = pd.DataFrame(rows); W.to_csv('results/paired_tests.csv', index=False); print(W.round(4).to_string())
