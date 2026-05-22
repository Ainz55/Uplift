# """Feature preparation for train/test tables with the official schema."""

# from __future__ import annotations

# import logging

# import numpy as np
# import pandas as pd

# from data import COMMUNICATION_COL, ID_COL, TARGET_COL, TREATMENT_COL

# logger = logging.getLogger(__name__)


# class FeatureBuilder:
#     """Deterministic preprocessing without target leakage."""

#     def __init__(self, *, max_categories: int = 40, feature_set: str = "baseline") -> None:
#         self.max_categories = max_categories
#         if feature_set not in {"baseline", "event_zero", "semantic", "enhanced"}:
#             raise ValueError("feature_set must be 'baseline', 'event_zero', 'semantic' or 'enhanced'")
#         self.feature_set = feature_set
#         self.base_feature_columns_: list[str] = []
#         self.numeric_columns_: list[str] = []
#         self.categorical_columns_: list[str] = []
#         self.numeric_medians_: pd.Series = pd.Series(dtype=float)
#         self.numeric_fill_values_: pd.Series = pd.Series(dtype=float)
#         self.log_columns_: list[str] = []
#         self.category_levels_: dict[str, list[object]] = {}
#         self.feature_columns_: list[str] = []

#     @staticmethod
#     def _is_spend_or_count_col(col: str) -> bool:
#         if col in {
#             "n_redeem",
#             "n_plastic_card",
#             "n_virtual_card",
#             "cus_mark_n_offers",
#             "cus_mark_n_view",
#             "cus_mark_n_rule",
#             "cus_mark_fav_cat_accept_flg_1_month_ago",
#             "cus_mark_fav_cat_accept_flg_2_month_ago",
#         }:
#             return True
#         if col.startswith("rto_format_"):
#             return True
#         if col.startswith("cus_cat_") and any(
#             token in col
#             for token in (
#                 "_rto",
#                 "_atv",
#                 "_std",
#                 "_n_days",
#                 "_n_sku",
#             )
#         ):
#             return True
#         return False

#     @staticmethod
#     def _is_event_zero_col(col: str) -> bool:
#         return col in {
#             "n_redeem",
#             "n_plastic_card",
#             "n_virtual_card",
#             "cus_mark_n_offers",
#             "cus_mark_n_view",
#             "cus_mark_n_rule",
#             "cus_mark_fav_cat_accept_flg_1_month_ago",
#             "cus_mark_fav_cat_accept_flg_2_month_ago",
#         }

#     @staticmethod
#     def _is_recency_col(col: str) -> bool:
#         return (
#             col == "n_days_last_visit"
#             or "_last_" in col
#             or col.endswith("_last_1_days")
#             or col.endswith("_last_2_days")
#             or col.endswith("_last_3_days")
#         )

#     @staticmethod
#     def _is_interval_col(col: str) -> bool:
#         return (
#             "days_btw_trn" in col
#             or "_diff_mean" in col
#             or "_diff_min" in col
#             or "_diff_max" in col
#             or "_diff_std" in col
#             or "_last_1_2_days" in col
#             or "_last_2_3_days" in col
#             or "_max_min_days_diff" in col
#         )

#     def _build_semantic_fill_values(self, numeric: pd.DataFrame) -> pd.Series:
#         fills = self.numeric_medians_.copy()
#         for col in numeric.columns:
#             if self._is_spend_or_count_col(col):
#                 fills[col] = 0.0
#             elif self._is_recency_col(col):
#                 max_value = numeric[col].max(skipna=True)
#                 fills[col] = float(max_value + 1.0) if pd.notna(max_value) else 999.0
#             elif self._is_interval_col(col):
#                 # Missing interval stats usually mean too few events. Keep that close
#                 # to zero and let the explicit missing flag carry the "not observed" signal.
#                 fills[col] = 0.0
#         return fills.fillna(0.0)

#     def _build_event_zero_fill_values(self, numeric: pd.DataFrame) -> pd.Series:
#         fills = self.numeric_medians_.copy()
#         for col in numeric.columns:
#             if self._is_event_zero_col(col):
#                 fills[col] = 0.0
#         return fills.fillna(0.0)

#     @staticmethod
#     def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
#         denom = denominator.replace(0, np.nan)
#         return (numerator / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

#     def _build_derived_numeric(self, numeric: pd.DataFrame) -> pd.DataFrame:
#         derived = pd.DataFrame(index=numeric.index)

#         def has(*cols: str) -> bool:
#             return all(col in numeric.columns for col in cols)

#         def add_ratio(name: str, numerator: str, denominator: str) -> None:
#             if has(numerator, denominator):
#                 derived[name] = self._safe_divide(numeric[numerator], numeric[denominator]).astype(np.float32)

#         # Relative category affinity. The campaign category is hidden, so category shares
#         # are often more useful than absolute customer spend.
#         for level in ("5", "6", "7"):
#             add_ratio(f"cus_cat_{level}_rto_share", f"cus_cat_{level}_rto", "rto")
#             add_ratio(f"cus_cat_{level}_atv_rel", f"cus_cat_{level}_atv", "atv")
#             add_ratio(f"cus_cat_{level}_sku_share", f"cus_cat_{level}_n_sku", "n_sku")
#             add_ratio(f"cus_cat_{level}_days_share", f"cus_cat_{level}_n_days", "n_days_life")
#             add_ratio(f"cus_cat_{level}_recency_rel", f"cus_cat_{level}_last_1_days", "n_days_last_visit")

#         add_ratio("cat5_to_cat6_rto", "cus_cat_5_rto", "cus_cat_6_rto")
#         add_ratio("cat6_to_cat7_rto", "cus_cat_6_rto", "cus_cat_7_rto")
#         add_ratio("cat5_to_cat7_rto", "cus_cat_5_rto", "cus_cat_7_rto")
#         add_ratio("cat5_to_cat6_sku", "cus_cat_5_n_sku", "cus_cat_6_n_sku")
#         add_ratio("cat6_to_cat7_sku", "cus_cat_6_n_sku", "cus_cat_7_n_sku")

#         for fmt in ("1", "2", "3"):
#             add_ratio(f"rto_format_{fmt}_share", f"rto_format_{fmt}", "rto")

#         add_ratio("atv_to_mtv", "atv", "mtv")
#         add_ratio("stdtv_to_atv", "stdtv", "atv")
#         add_ratio("p75_to_p25_tv", "p_75_tv", "p_25_tv")
#         add_ratio("mntv_to_mxtv", "mntv", "mxtv")
#         add_ratio("sku_per_trn", "n_sku", "n_trn")
#         add_ratio("cat5_per_trn", "n_cat_5", "n_trn")
#         add_ratio("cat6_per_trn", "n_cat_6", "n_trn")
#         add_ratio("cat7_per_trn", "n_cat_7", "n_trn")
#         add_ratio("redeem_per_trn", "n_redeem", "n_trn")

#         add_ratio("marketing_view_rate", "cus_mark_n_view", "cus_mark_n_offers")
#         add_ratio("marketing_rule_rate", "cus_mark_n_rule", "cus_mark_n_offers")

#         if has("cus_mark_fav_cat_accept_flg_1_month_ago", "cus_mark_fav_cat_accept_flg_2_month_ago"):
#             derived["marketing_accept_recent_sum"] = (
#                 numeric["cus_mark_fav_cat_accept_flg_1_month_ago"]
#                 + numeric["cus_mark_fav_cat_accept_flg_2_month_ago"]
#             ).astype(np.float32)
#             derived["marketing_accept_recent_delta"] = (
#                 numeric["cus_mark_fav_cat_accept_flg_1_month_ago"]
#                 - numeric["cus_mark_fav_cat_accept_flg_2_month_ago"]
#             ).astype(np.float32)

#         if has("cus_cat_5_last_1_days", "cus_cat_5_last_2_days"):
#             derived["cat5_recency_delta_1_2"] = (
#                 numeric["cus_cat_5_last_2_days"] - numeric["cus_cat_5_last_1_days"]
#             ).astype(np.float32)
#         if has("cus_cat_6_last_1_days", "cus_cat_6_last_2_days"):
#             derived["cat6_recency_delta_1_2"] = (
#                 numeric["cus_cat_6_last_2_days"] - numeric["cus_cat_6_last_1_days"]
#             ).astype(np.float32)
#         if has("cus_cat_7_last_1_days", "cus_cat_7_last_2_days"):
#             derived["cat7_recency_delta_1_2"] = (
#                 numeric["cus_cat_7_last_2_days"] - numeric["cus_cat_7_last_1_days"]
#             ).astype(np.float32)

#         return derived.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32, copy=False)

#     def fit(self, train: pd.DataFrame, test: pd.DataFrame) -> FeatureBuilder:
#         reserved = {ID_COL, TREATMENT_COL, TARGET_COL}
#         self.base_feature_columns_ = [c for c in train.columns if c not in reserved]
#         missing = [c for c in self.base_feature_columns_ if c not in test.columns]
#         if missing:
#             raise ValueError(f"test is missing train feature columns: {missing[:20]}")

#         feature_df = train[self.base_feature_columns_].copy()

#         forced_categorical = {COMMUNICATION_COL}
#         self.categorical_columns_ = [
#             c
#             for c in self.base_feature_columns_
#             if c in forced_categorical
#             or pd.api.types.is_object_dtype(feature_df[c])
#             or pd.api.types.is_categorical_dtype(feature_df[c])
#             or pd.api.types.is_bool_dtype(feature_df[c])
#         ]
#         self.numeric_columns_ = [
#             c for c in self.base_feature_columns_ if c not in self.categorical_columns_
#         ]

#         numeric = feature_df[self.numeric_columns_].apply(pd.to_numeric, errors="coerce")
#         numeric = numeric.replace([np.inf, -np.inf], np.nan)
#         self.numeric_medians_ = numeric.median().fillna(0.0)
#         if self.feature_set == "semantic":
#             self.numeric_fill_values_ = self._build_semantic_fill_values(numeric)
#         elif self.feature_set == "event_zero":
#             self.numeric_fill_values_ = self._build_event_zero_fill_values(numeric)
#         else:
#             self.numeric_fill_values_ = self.numeric_medians_

#         self.log_columns_ = []
#         for col in self.numeric_columns_:
#             s = numeric[col]
#             clean = s.dropna()
#             if clean.empty:
#                 continue
#             non_negative = bool((clean >= 0).all())
#             skewed = abs(float(clean.skew())) > 2.0 if len(clean) > 2 else False
#             if non_negative and skewed:
#                 self.log_columns_.append(col)

#         self.category_levels_ = {}
#         for col in self.categorical_columns_:
#             values = feature_df[col].astype("object").where(feature_df[col].notna(), "__MISSING__")
#             levels = values.value_counts(dropna=False).head(self.max_categories).index.tolist()
#             self.category_levels_[col] = levels

#         transformed = self.transform(train)
#         self.feature_columns_ = list(transformed.columns)
#         logger.info(
#             "Features: %d numeric, %d categorical, %d final columns",
#             len(self.numeric_columns_),
#             len(self.categorical_columns_),
#             len(self.feature_columns_),
#         )
#         return self

#     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
#         missing = [c for c in self.base_feature_columns_ if c not in df.columns]
#         if missing:
#             raise ValueError(f"input is missing feature columns: {missing[:20]}")

#         parts: list[pd.DataFrame] = []

#         if self.numeric_columns_:
#             numeric = df[self.numeric_columns_].apply(pd.to_numeric, errors="coerce")
#             numeric = numeric.replace([np.inf, -np.inf], np.nan)
#             missing_flags = numeric.isna().astype(np.float32)
#             missing_flags.columns = [f"{c}__missing" for c in missing_flags.columns]
#             numeric = numeric.fillna(self.numeric_fill_values_).astype(np.float32)
#             parts.extend([numeric, missing_flags])

#             if self.feature_set == "enhanced":
#                 derived_numeric = self._build_derived_numeric(numeric)
#                 if len(derived_numeric.columns):
#                     parts.append(derived_numeric)
#             else:
#                 derived_numeric = pd.DataFrame(index=df.index)

#             if self.log_columns_:
#                 log_df = pd.DataFrame(index=df.index)
#                 for col in self.log_columns_:
#                     log_df[f"{col}__log1p"] = np.log1p(numeric[col].clip(lower=0)).astype(np.float32)
#                 parts.append(log_df)
#         else:
#             numeric = pd.DataFrame(index=df.index)
#             derived_numeric = pd.DataFrame(index=df.index)

#         communication_dummies: pd.DataFrame | None = None
#         for col in self.categorical_columns_:
#             raw = df[col].astype("object").where(df[col].notna(), "__MISSING__")
#             levels = self.category_levels_.get(col, [])
#             cat_df = pd.DataFrame(index=df.index)
#             for level in levels:
#                 safe_level = str(level).replace(" ", "_").replace("/", "_")
#                 cat_df[f"{col}__{safe_level}"] = (raw == level).astype(np.float32)
#             cat_df[f"{col}__OTHER"] = (~raw.isin(levels)).astype(np.float32)
#             parts.append(cat_df)
#             if col == COMMUNICATION_COL:
#                 communication_dummies = cat_df

#         if self.feature_set == "enhanced" and communication_dummies is not None and len(numeric.columns):
#             interaction_source = pd.DataFrame(index=df.index)
#             for col in (
#                 "rto",
#                 "atv",
#                 "n_trn",
#                 "n_days_last_visit",
#                 "cus_cat_5_rto",
#                 "cus_cat_6_rto",
#                 "cus_cat_7_rto",
#                 "cus_mark_n_offers",
#                 "cus_mark_n_view",
#             ):
#                 if col in numeric.columns:
#                     interaction_source[col] = numeric[col]
#             for col in (
#                 "cus_cat_5_rto_share",
#                 "cus_cat_6_rto_share",
#                 "cus_cat_7_rto_share",
#                 "marketing_view_rate",
#             ):
#                 if col in derived_numeric.columns:
#                     interaction_source[col] = derived_numeric[col]
#             if len(interaction_source.columns):
#                 inter = pd.DataFrame(index=df.index)
#                 for comm_col in communication_dummies.columns:
#                     if comm_col.endswith("__OTHER"):
#                         continue
#                     for source_col in interaction_source.columns:
#                         inter[f"{source_col}_x_{comm_col}"] = (
#                             interaction_source[source_col] * communication_dummies[comm_col]
#                         ).astype(np.float32)
#                 parts.append(inter)

#         if not parts:
#             return pd.DataFrame(index=df.index)

#         result = pd.concat(parts, axis=1)
#         result = result.replace([np.inf, -np.inf], 0.0).fillna(0.0)
#         return result.astype(np.float32, copy=False)

#     def fit_transform(
#         self,
#         train: pd.DataFrame,
#         test: pd.DataFrame,
#     ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
#         self.fit(train, test)
#         x_train = self.transform(train)
#         x_test = self.transform(test)
#         x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
#         self.feature_columns_ = list(x_train.columns)
#         return x_train, x_test, self.feature_columns_


# def prepare_datasets(
#     train: pd.DataFrame,
#     test: pd.DataFrame,
#     *,
#     max_categories: int = 40,
#     feature_set: str = "baseline",
# ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], FeatureBuilder]:
#     builder = FeatureBuilder(max_categories=max_categories, feature_set=feature_set)
#     x_train, x_test, feature_cols = builder.fit_transform(train, test)
#     return x_train, x_test, feature_cols, builder
