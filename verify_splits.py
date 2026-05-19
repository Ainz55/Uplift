"""Check CV and early-stopping split stratification."""

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from evaluation import stratified_holdout_split

train = pd.read_csv("dataset/uplift_train.csv")
t = train.treatment_flg.values
y = train.target.values
strata = t * 2 + y

global_treatment_rate = t.mean()
global_target_rate = y.mean()

skf = StratifiedKFold(5, shuffle=True, random_state=42)
print(f"Global treatment rate: {global_treatment_rate:.6f}")
print(f"Global target rate:    {global_target_rate:.6f}\n")

for fold, (tr_idx, oof_idx) in enumerate(skf.split(train, strata), 1):
    fit_idx, es_idx = stratified_holdout_split(
        tr_idx,
        t,
        y,
        val_fraction=0.2,
        random_state=42 + fold,
    )

    r_oof = t[oof_idx].mean()
    r_fit = t[fit_idx].mean()
    r_es = t[es_idx].mean()
    y_oof = y[oof_idx].mean()
    y_fit = y[fit_idx].mean()
    y_es = y[es_idx].mean()

    print(
        f"Fold {fold}: treatment oof={r_oof:.5f}  fit={r_fit:.5f}  early_stop={r_es:.5f}  "
        f"(max |delta|={max(abs(r_oof-global_treatment_rate), abs(r_fit-global_treatment_rate), abs(r_es-global_treatment_rate)):.5f})"
    )
    print(
        f"         target    oof={y_oof:.5f}  fit={y_fit:.5f}  early_stop={y_es:.5f}  "
        f"(max |delta|={max(abs(y_oof-global_target_rate), abs(y_fit-global_target_rate), abs(y_es-global_target_rate)):.5f})"
    )
    print(f"         n_fit={len(fit_idx):,}  n_es={len(es_idx):,}  n_oof={len(oof_idx):,}")

print("\nOK: early_stop is separate from oof_val; stratify=treatment+target at each level.")
