"""Feature preparation for train/test tables with the official schema."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data import COMMUNICATION_COL, ID_COL, TARGET_COL, TREATMENT_COL

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Deterministic preprocessing without target leakage."""

    def __init__(self, *, max_categories: int = 40) -> None:
        self.max_categories = max_categories
        self.base_feature_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.numeric_medians_: pd.Series = pd.Series(dtype=float)
        self.log_columns_: list[str] = []
        self.category_levels_: dict[str, list[object]] = {}
        self.feature_columns_: list[str] = []

    def fit(self, train: pd.DataFrame, test: pd.DataFrame) -> FeatureBuilder:
        reserved = {ID_COL, TREATMENT_COL, TARGET_COL}
        self.base_feature_columns_ = [c for c in train.columns if c not in reserved]
        missing = [c for c in self.base_feature_columns_ if c not in test.columns]
        if missing:
            raise ValueError(f"test is missing train feature columns: {missing[:20]}")

        feature_df = train[self.base_feature_columns_].copy()

        forced_categorical = {COMMUNICATION_COL}
        self.categorical_columns_ = [
            c
            for c in self.base_feature_columns_
            if c in forced_categorical
            or pd.api.types.is_object_dtype(feature_df[c])
            or pd.api.types.is_categorical_dtype(feature_df[c])
            or pd.api.types.is_bool_dtype(feature_df[c])
        ]
        self.numeric_columns_ = [
            c for c in self.base_feature_columns_ if c not in self.categorical_columns_
        ]

        numeric = feature_df[self.numeric_columns_].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        self.numeric_medians_ = numeric.median().fillna(0.0)

        self.log_columns_ = []
        for col in self.numeric_columns_:
            s = numeric[col]
            clean = s.dropna()
            if clean.empty:
                continue
            non_negative = bool((clean >= 0).all())
            skewed = abs(float(clean.skew())) > 2.0 if len(clean) > 2 else False
            if non_negative and skewed:
                self.log_columns_.append(col)

        self.category_levels_ = {}
        for col in self.categorical_columns_:
            values = feature_df[col].astype("object").where(feature_df[col].notna(), "__MISSING__")
            levels = values.value_counts(dropna=False).head(self.max_categories).index.tolist()
            self.category_levels_[col] = levels

        transformed = self.transform(train)
        self.feature_columns_ = list(transformed.columns)
        logger.info(
            "Features: %d numeric, %d categorical, %d final columns",
            len(self.numeric_columns_),
            len(self.categorical_columns_),
            len(self.feature_columns_),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.base_feature_columns_ if c not in df.columns]
        if missing:
            raise ValueError(f"input is missing feature columns: {missing[:20]}")

        parts: list[pd.DataFrame] = []

        if self.numeric_columns_:
            numeric = df[self.numeric_columns_].apply(pd.to_numeric, errors="coerce")
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            missing_flags = numeric.isna().astype(np.float32)
            missing_flags.columns = [f"{c}__missing" for c in missing_flags.columns]
            numeric = numeric.fillna(self.numeric_medians_).astype(np.float32)
            parts.extend([numeric, missing_flags])

            if self.log_columns_:
                log_df = pd.DataFrame(index=df.index)
                for col in self.log_columns_:
                    log_df[f"{col}__log1p"] = np.log1p(numeric[col].clip(lower=0)).astype(np.float32)
                parts.append(log_df)

        for col in self.categorical_columns_:
            raw = df[col].astype("object").where(df[col].notna(), "__MISSING__")
            levels = self.category_levels_.get(col, [])
            cat_df = pd.DataFrame(index=df.index)
            for level in levels:
                safe_level = str(level).replace(" ", "_").replace("/", "_")
                cat_df[f"{col}__{safe_level}"] = (raw == level).astype(np.float32)
            cat_df[f"{col}__OTHER"] = (~raw.isin(levels)).astype(np.float32)
            parts.append(cat_df)

        if not parts:
            return pd.DataFrame(index=df.index)

        result = pd.concat(parts, axis=1)
        result = result.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return result.astype(np.float32, copy=False)

    def fit_transform(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        self.fit(train, test)
        x_train = self.transform(train)
        x_test = self.transform(test)
        x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
        self.feature_columns_ = list(x_train.columns)
        return x_train, x_test, self.feature_columns_


def prepare_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    max_categories: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], FeatureBuilder]:
    builder = FeatureBuilder(max_categories=max_categories)
    x_train, x_test, feature_cols = builder.fit_transform(train, test)
    return x_train, x_test, feature_cols, builder
