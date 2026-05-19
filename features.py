"""Построение признаков клиентов для uplift-модели."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import AGE_MAX, AGE_MIN


class FeatureBuilder:
    """Инженерия признаков из справочника clients.csv."""

    def __init__(self, reference_date: pd.Timestamp | None = None) -> None:
        self.reference_date = reference_date
        self.feature_columns_: list[str] = []
        self.age_median_: float = 45.0

    def fit(self, clients: pd.DataFrame) -> FeatureBuilder:
        ages = self._clean_age(clients["age"])
        self.age_median_ = float(ages.median())
        if self.reference_date is None:
            issue = pd.to_datetime(clients["first_issue_date"], errors="coerce")
            self.reference_date = issue.max()
        return self

    @staticmethod
    def _clean_age(age: pd.Series) -> pd.Series:
        age = pd.to_numeric(age, errors="coerce")
        valid = age.between(AGE_MIN, AGE_MAX)
        cleaned = age.where(valid)
        return cleaned

    def transform(self, clients: pd.DataFrame) -> pd.DataFrame:
        df = clients.copy()
        df["first_issue_date"] = pd.to_datetime(df["first_issue_date"], errors="coerce")
        df["first_redeem_date"] = pd.to_datetime(df["first_redeem_date"], errors="coerce")

        ref = self.reference_date
        df["days_since_issue"] = (ref - df["first_issue_date"]).dt.days
        df["days_since_redeem"] = (ref - df["first_redeem_date"]).dt.days
        df["days_issue_to_redeem"] = (df["first_redeem_date"] - df["first_issue_date"]).dt.days

        df["has_redeemed"] = df["first_redeem_date"].notna().astype(np.int8)
        df["issue_year"] = df["first_issue_date"].dt.year
        df["issue_month"] = df["first_issue_date"].dt.month
        df["issue_quarter"] = df["first_issue_date"].dt.quarter
        df["issue_dow"] = df["first_issue_date"].dt.dayofweek
        df["issue_is_weekend"] = df["issue_dow"].isin([5, 6]).astype(np.int8)

        age_raw = pd.to_numeric(df["age"], errors="coerce")
        age_clean = self._clean_age(df["age"])
        df["age"] = age_clean.fillna(self.age_median_)
        df["age_missing"] = age_clean.isna().astype(np.int8)
        df["age_outlier"] = (~age_raw.between(AGE_MIN, AGE_MAX) & age_raw.notna()).astype(np.int8)

        df["age_group_young"] = (df["age"] < 35).astype(np.int8)
        df["age_group_middle"] = ((df["age"] >= 35) & (df["age"] < 55)).astype(np.int8)
        df["age_group_senior"] = (df["age"] >= 55).astype(np.int8)
        df["age_bin_18_24"] = df["age"].between(18, 24, inclusive="both").astype(np.int8)
        df["age_bin_25_34"] = df["age"].between(25, 34, inclusive="both").astype(np.int8)
        df["age_bin_35_44"] = df["age"].between(35, 44, inclusive="both").astype(np.int8)
        df["age_bin_45_54"] = df["age"].between(45, 54, inclusive="both").astype(np.int8)
        df["age_bin_55_64"] = df["age"].between(55, 64, inclusive="both").astype(np.int8)
        df["age_bin_65_plus"] = (df["age"] >= 65).astype(np.int8)

        df["log_days_since_issue"] = np.log1p(df["days_since_issue"].clip(lower=0))
        df["log_days_issue_to_redeem"] = np.log1p(
            df["days_issue_to_redeem"].clip(lower=0).fillna(0)
        )

        df["redeem_before_issue"] = (
            df["days_issue_to_redeem"].notna() & (df["days_issue_to_redeem"] < 0)
        ).astype(np.int8)
        df["quick_redeem_7d"] = df["days_issue_to_redeem"].between(0, 7, inclusive="both").astype(np.int8)
        df["quick_redeem_30d"] = df["days_issue_to_redeem"].between(0, 30, inclusive="both").astype(np.int8)
        df["long_redeem_180d"] = (df["days_issue_to_redeem"] > 180).astype(np.int8)
        df["new_client_90d"] = (df["days_since_issue"] <= 90).astype(np.int8)
        df["old_client_720d"] = (df["days_since_issue"] >= 720).astype(np.int8)
        df["redeem_delay_ratio"] = (
            df["days_issue_to_redeem"].clip(lower=0)
            / (df["days_since_issue"].clip(lower=1) + 1.0)
        ).fillna(0)
        df["days_since_issue_x_age"] = df["log_days_since_issue"] * df["age"]
        df["has_redeemed_x_age"] = df["has_redeemed"] * df["age"]

        for col in ("days_since_redeem", "days_issue_to_redeem"):
            df[col] = df[col].fillna(-1)

        gender = pd.get_dummies(df["gender"].fillna("U"), prefix="gender", dtype=np.int8)
        df = pd.concat([df.drop(columns=["gender"]), gender], axis=1)
        for col in [c for c in df.columns if c.startswith("gender_")]:
            df[f"{col}_x_has_redeemed"] = df[col] * df["has_redeemed"]
            df[f"{col}_x_age"] = df[col] * df["age"]

        self.feature_columns_ = [
            "days_since_issue",
            "days_since_redeem",
            "days_issue_to_redeem",
            "log_days_since_issue",
            "log_days_issue_to_redeem",
            "has_redeemed",
            "issue_year",
            "issue_month",
            "issue_quarter",
            "issue_dow",
            "issue_is_weekend",
            "age",
            "age_missing",
            "age_outlier",
            "age_group_young",
            "age_group_middle",
            "age_group_senior",
            "age_bin_18_24",
            "age_bin_25_34",
            "age_bin_35_44",
            "age_bin_45_54",
            "age_bin_55_64",
            "age_bin_65_plus",
            "redeem_before_issue",
            "quick_redeem_7d",
            "quick_redeem_30d",
            "long_redeem_180d",
            "new_client_90d",
            "old_client_720d",
            "redeem_delay_ratio",
            "days_since_issue_x_age",
            "has_redeemed_x_age",
        ] + [c for c in df.columns if c.startswith("gender_")]

        return df[["client_id"] + self.feature_columns_]

    def fit_transform(self, clients: pd.DataFrame) -> pd.DataFrame:
        return self.fit(clients).transform(clients)


def prepare_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    clients: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    builder = FeatureBuilder()
    features = builder.fit_transform(clients)

    train_df = train.merge(features, on="client_id", how="left")
    test_df = test.merge(features, on="client_id", how="left")

    feature_cols = builder.feature_columns_
    for df in (train_df, test_df):
        df[feature_cols] = df[feature_cols].fillna(-1)

    return train_df, test_df, feature_cols
