# """Quick check for CV and early-stopping split stratification."""

# from __future__ import annotations

# import argparse
# from pathlib import Path

# import numpy as np

# from data import TARGET_COL, TREATMENT_COL, load_and_validate
# from evaluation import make_strata, stratified_holdout_split


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=Path, default=Path("dataset"))
#     parser.add_argument("--train", type=Path, default=None)
#     parser.add_argument("--test", type=Path, default=None)
#     parser.add_argument("--folds", type=int, default=5)
#     args = parser.parse_args()

#     train, _, _ = load_and_validate(args.dataset, train_path=args.train, test_path=args.test)
#     treatment = train[TREATMENT_COL].to_numpy()
#     y = train[TARGET_COL].to_numpy(dtype=float)
#     strata = make_strata(treatment, y)

#     print(f"Rows: {len(train):,}")
#     print(f"Treatment rate: {treatment.mean():.6f}")
#     print(f"Positive rec_spend rate: {(y > 0).mean():.6f}")
#     print(f"Strata counts: {dict(zip(*np.unique(strata, return_counts=True)))}")

#     indices = np.arange(len(train))
#     fit_idx, es_idx = stratified_holdout_split(
#         indices,
#         treatment,
#         y,
#         val_fraction=0.2,
#         random_state=42,
#     )
#     print(f"Fit rows: {len(fit_idx):,}; early-stop rows: {len(es_idx):,}")
#     print(f"Fit treatment rate: {treatment[fit_idx].mean():.6f}")
#     print(f"Early-stop treatment rate: {treatment[es_idx].mean():.6f}")


# if __name__ == "__main__":
#     main()
