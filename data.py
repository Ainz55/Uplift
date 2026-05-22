# """Data loading and validation for the official hackathon schema."""

# from __future__ import annotations

# import logging
# from dataclasses import dataclass
# from pathlib import Path

# import numpy as np
# import pandas as pd

# logger = logging.getLogger(__name__)

# ID_COL = "user_id"
# TREATMENT_COL = "treatment_flg"
# TARGET_COL = "rec_spend"
# COMMUNICATION_COL = "communication_type"

# TRAIN_REQUIRED = {ID_COL, COMMUNICATION_COL, TREATMENT_COL, TARGET_COL}
# TEST_REQUIRED = {ID_COL, COMMUNICATION_COL}


# @dataclass
# class DatasetInfo:
#     n_train: int
#     n_test: int
#     n_features_raw: int
#     treatment_rate: float
#     spend_mean: float
#     spend_zero_rate: float
#     spend_control_mean: float
#     spend_treatment_mean: float
#     avg_uplift: float
#     duplicate_user_ids_train: int
#     duplicate_user_ids_test: int


# class DatasetError(ValueError):
#     """Raised when input files do not match the expected hackathon schema."""


# def _read_table(path: Path) -> pd.DataFrame:
#     if not path.exists():
#         raise DatasetError(f"Input file not found: {path}")

#     suffix = path.suffix.lower()
#     if suffix == ".parquet":
#         return pd.read_parquet(path)
#     if suffix == ".csv":
#         logger.warning("Reading CSV input %s. Official submission format is Parquet.", path)
#         return pd.read_csv(path)
#     raise DatasetError(f"Unsupported input format for {path}. Expected .parquet or .csv")


# def _require_columns(df: pd.DataFrame, required: set[str], context: str) -> None:
#     missing = required - set(df.columns)
#     if missing:
#         raise DatasetError(
#             f"{context}: missing required columns {sorted(missing)}. "
#             f"Expected official schema with user_id, communication_type, treatment_flg and rec_spend."
#         )


# def _check_binary(series: pd.Series, name: str) -> None:
#     values = set(pd.unique(series.dropna()))
#     if not values <= {0, 1, False, True}:
#         raise DatasetError(f"{name} must contain only 0/1 values, got: {sorted(values)}")


# def _check_target(series: pd.Series) -> None:
#     y = pd.to_numeric(series, errors="coerce")
#     if y.isna().any():
#         raise DatasetError(f"{TARGET_COL} must be numeric and non-null")
#     if np.isinf(y.to_numpy(dtype=float)).any():
#         raise DatasetError(f"{TARGET_COL} must not contain infinite values")


# def _validate_feature_contract(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
#     reserved = {TREATMENT_COL, TARGET_COL}
#     train_features = [c for c in train.columns if c not in reserved]
#     test_features = list(test.columns)

#     missing_in_test = [c for c in train_features if c not in test.columns]
#     if missing_in_test:
#         raise DatasetError(
#             "test data is missing feature columns present in train: "
#             f"{missing_in_test[:20]}"
#         )

#     extra_test = [c for c in test_features if c not in train_features]
#     if extra_test:
#         logger.warning("Ignoring extra test-only columns: %s", extra_test[:20])

#     return train_features


# def load_and_validate(
#     dataset_dir: Path,
#     *,
#     train_path: Path | None = None,
#     test_path: Path | None = None,
# ) -> tuple[pd.DataFrame, pd.DataFrame, DatasetInfo]:
#     train_file = train_path or Path(dataset_dir) / "train.parquet"
#     test_file = test_path or Path(dataset_dir) / "test.parquet"

#     train = _read_table(Path(train_file))
#     test = _read_table(Path(test_file))

#     _require_columns(train, TRAIN_REQUIRED, str(train_file))
#     _require_columns(test, TEST_REQUIRED, str(test_file))
#     _check_binary(train[TREATMENT_COL], TREATMENT_COL)
#     _check_target(train[TARGET_COL])

#     if TREATMENT_COL in test.columns or TARGET_COL in test.columns:
#         raise DatasetError(f"test file must not contain {TREATMENT_COL} or {TARGET_COL}")

#     _validate_feature_contract(train, test)

#     train[TREATMENT_COL] = train[TREATMENT_COL].astype(np.int8)
#     train[TARGET_COL] = pd.to_numeric(train[TARGET_COL], errors="raise").astype(float)

#     dup_train = int(train[ID_COL].duplicated().sum())
#     dup_test = int(test[ID_COL].duplicated().sum())
#     if dup_train:
#         logger.warning("train has %d duplicated user_id values", dup_train)
#     if dup_test:
#         logger.warning("test has %d duplicated user_id values", dup_test)

#     treatment = train[TREATMENT_COL].to_numpy()
#     spend = train[TARGET_COL].to_numpy(dtype=float)
#     control = treatment == 0
#     treated = treatment == 1
#     control_mean = float(spend[control].mean()) if control.any() else 0.0
#     treated_mean = float(spend[treated].mean()) if treated.any() else 0.0

#     info = DatasetInfo(
#         n_train=len(train),
#         n_test=len(test),
#         n_features_raw=len([c for c in train.columns if c not in {ID_COL, TREATMENT_COL, TARGET_COL}]),
#         treatment_rate=float(treatment.mean()),
#         spend_mean=float(spend.mean()),
#         spend_zero_rate=float((spend == 0).mean()),
#         spend_control_mean=control_mean,
#         spend_treatment_mean=treated_mean,
#         avg_uplift=treated_mean - control_mean,
#         duplicate_user_ids_train=dup_train,
#         duplicate_user_ids_test=dup_test,
#     )
#     return train, test, info
