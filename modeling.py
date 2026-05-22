"""Continuous-outcome uplift models for rec_spend."""

from __future__ import annotations

from abc import ABC, abstractmethod

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import LGBMParams, RANDOM_STATE

ModelName = str


def _parse_model_name(name: str) -> tuple[str, int]:
    if "_s" not in name:
        return name, 0
    base, seed = name.rsplit("_s", 1)
    return (base, int(seed)) if seed.isdigit() else (name, 0)


def _common_params(params: LGBMParams, seed_offset: int) -> dict:
    return {
        "n_estimators": params.n_estimators,
        "learning_rate": params.learning_rate,
        "max_depth": params.max_depth,
        "num_leaves": params.num_leaves,
        "min_child_samples": params.min_child_samples,
        "subsample": params.subsample,
        "colsample_bytree": params.colsample_bytree,
        "reg_alpha": params.reg_alpha,
        "reg_lambda": params.reg_lambda,
        "random_state": RANDOM_STATE + seed_offset,
        "verbose": -1,
        "n_jobs": params.n_jobs,
        "force_col_wise": True,
        "deterministic": True,
    }


def _make_regressor(params: LGBMParams, seed_offset: int = 0) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="regression", **_common_params(params, seed_offset))


def _make_classifier(params: LGBMParams, seed_offset: int = 0) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(objective="binary", **_common_params(params, seed_offset))


def _fit_regressor(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    y: np.ndarray,
    params: LGBMParams,
    *,
    eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    sample_weight: np.ndarray | None = None,
) -> None:
    fit_kwargs: dict = {}
    if eval_set is not None and len(eval_set[0]) > 0:
        fit_kwargs["eval_set"] = [eval_set]
        fit_kwargs["callbacks"] = [lgb.early_stopping(params.early_stopping_rounds, verbose=False)]
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X, y, **fit_kwargs)


def _fit_classifier(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    params: LGBMParams,
    *,
    eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
) -> None:
    fit_kwargs: dict = {}
    if eval_set is not None and len(eval_set[0]) > 0 and len(np.unique(eval_set[1])) > 1:
        fit_kwargs["eval_set"] = [eval_set]
        fit_kwargs["callbacks"] = [lgb.early_stopping(params.early_stopping_rounds, verbose=False)]
    model.fit(X, y, **fit_kwargs)


def _safe_expm1(values: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(values, -20, 20))


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


class BaseUpliftModel(ABC):
    name: str

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        treatment: np.ndarray,
        *,
        eval_set: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None,
    ) -> BaseUpliftModel:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        return pd.Series(dtype=float)


class TwoModelRegressor(BaseUpliftModel):
    """T-learner: separate log-spend regressors for treatment and control."""

    name = "t_learner_log"

    def __init__(self, params: LGBMParams, seed_offset: int = 0) -> None:
        self.params = params
        self.seed_offset = seed_offset
        self.model_t_: lgb.LGBMRegressor | None = None
        self.model_c_: lgb.LGBMRegressor | None = None

    def fit(self, X, y, treatment, *, eval_set=None) -> TwoModelRegressor:
        y_log = np.log1p(np.asarray(y, dtype=float))
        treatment = np.asarray(treatment, dtype=np.int8)
        mask_t = treatment == 1
        mask_c = treatment == 0

        self.model_t_ = _make_regressor(self.params, self.seed_offset + 1)
        self.model_c_ = _make_regressor(self.params, self.seed_offset + 2)

        eval_t = eval_c = None
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            y_val_log = np.log1p(np.asarray(y_val, dtype=float))
            eval_t = (X_val[t_val == 1], y_val_log[t_val == 1])
            eval_c = (X_val[t_val == 0], y_val_log[t_val == 0])

        _fit_regressor(self.model_t_, X[mask_t], y_log[mask_t], self.params, eval_set=eval_t)
        _fit_regressor(self.model_c_, X[mask_c], y_log[mask_c], self.params, eval_set=eval_c)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_t_ is not None and self.model_c_ is not None
        return _safe_expm1(self.model_t_.predict(X)) - _safe_expm1(self.model_c_.predict(X))

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        assert self.model_t_ is not None and self.model_c_ is not None
        imp = (self.model_t_.feature_importances_ + self.model_c_.feature_importances_) / 2
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)


class SoloRegressor(BaseUpliftModel):
    """S-learner: one log-spend model with treatment as a feature."""

    name = "s_learner_log"

    def __init__(self, params: LGBMParams, seed_offset: int = 0) -> None:
        self.params = params
        self.seed_offset = seed_offset
        self.model_: lgb.LGBMRegressor | None = None

    def fit(self, X, y, treatment, *, eval_set=None) -> SoloRegressor:
        X_aug = X.copy()
        X_aug["treatment_flg"] = treatment
        y_log = np.log1p(np.asarray(y, dtype=float))
        self.model_ = _make_regressor(self.params, self.seed_offset + 3)

        eval_reg = None
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            Xv = X_val.copy()
            Xv["treatment_flg"] = t_val
            eval_reg = (Xv, np.log1p(np.asarray(y_val, dtype=float)))

        _fit_regressor(self.model_, X_aug, y_log, self.params, eval_set=eval_reg)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        X1 = X.copy()
        X0 = X.copy()
        X1["treatment_flg"] = 1
        X0["treatment_flg"] = 0
        return _safe_expm1(self.model_.predict(X1)) - _safe_expm1(self.model_.predict(X0))

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        assert self.model_ is not None
        imp = pd.Series(self.model_.feature_importances_, index=feature_names + ["treatment_flg"])
        return imp.drop(index="treatment_flg", errors="ignore").sort_values(ascending=False)


class TransformedOutcomeRegressor(BaseUpliftModel):
    """Transformed outcome learner for randomized experiments."""

    name = "transformed_outcome"

    def __init__(self, params: LGBMParams, seed_offset: int = 0) -> None:
        self.params = params
        self.seed_offset = seed_offset
        self.model_: lgb.LGBMRegressor | None = None
        self.propensity_: float = 0.5

    @staticmethod
    def _target(y: np.ndarray, treatment: np.ndarray, propensity: float) -> np.ndarray:
        p = float(np.clip(propensity, 1e-3, 1 - 1e-3))
        return y * (treatment / p - (1.0 - treatment) / (1.0 - p))

    def fit(self, X, y, treatment, *, eval_set=None) -> TransformedOutcomeRegressor:
        y = np.asarray(y, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        self.propensity_ = float(treatment.mean())
        z = self._target(y, treatment, self.propensity_)
        self.model_ = _make_regressor(self.params, self.seed_offset + 4)

        eval_reg = None
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            z_val = self._target(
                np.asarray(y_val, dtype=float),
                np.asarray(t_val, dtype=float),
                self.propensity_,
            )
            eval_reg = (X_val, z_val)

        _fit_regressor(self.model_, X, z, self.params, eval_set=eval_reg)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict(X)

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        assert self.model_ is not None
        return pd.Series(self.model_.feature_importances_, index=feature_names).sort_values(ascending=False)


class HurdleTwoModelRegressor(BaseUpliftModel):
    """T-learner with purchase probability and positive-spend amount models."""

    name = "hurdle_t_learner"

    def __init__(self, params: LGBMParams, seed_offset: int = 0) -> None:
        self.params = params
        self.seed_offset = seed_offset
        self.clf_t_: lgb.LGBMClassifier | None = None
        self.clf_c_: lgb.LGBMClassifier | None = None
        self.reg_t_: lgb.LGBMRegressor | None = None
        self.reg_c_: lgb.LGBMRegressor | None = None
        self.mean_pos_t_: float = 0.0
        self.mean_pos_c_: float = 0.0

    def fit(self, X, y, treatment, *, eval_set=None) -> HurdleTwoModelRegressor:
        y = np.asarray(y, dtype=float)
        treatment = np.asarray(treatment, dtype=np.int8)
        positive = (y > 0).astype(np.int8)
        mask_t = treatment == 1
        mask_c = treatment == 0

        self.clf_t_ = _make_classifier(self.params, self.seed_offset + 11)
        self.clf_c_ = _make_classifier(self.params, self.seed_offset + 12)
        self.reg_t_ = _make_regressor(self.params, self.seed_offset + 13)
        self.reg_c_ = _make_regressor(self.params, self.seed_offset + 14)

        eval_clf_t = eval_clf_c = eval_reg_t = eval_reg_c = None
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            y_val = np.asarray(y_val, dtype=float)
            pos_val = (y_val > 0).astype(np.int8)
            eval_clf_t = (X_val[t_val == 1], pos_val[t_val == 1])
            eval_clf_c = (X_val[t_val == 0], pos_val[t_val == 0])
            eval_reg_t = (X_val[(t_val == 1) & (y_val > 0)], np.log1p(y_val[(t_val == 1) & (y_val > 0)]))
            eval_reg_c = (X_val[(t_val == 0) & (y_val > 0)], np.log1p(y_val[(t_val == 0) & (y_val > 0)]))

        _fit_classifier(self.clf_t_, X[mask_t], positive[mask_t], self.params, eval_set=eval_clf_t)
        _fit_classifier(self.clf_c_, X[mask_c], positive[mask_c], self.params, eval_set=eval_clf_c)

        pos_t = mask_t & (y > 0)
        pos_c = mask_c & (y > 0)
        self.mean_pos_t_ = float(y[pos_t].mean()) if pos_t.any() else 0.0
        self.mean_pos_c_ = float(y[pos_c].mean()) if pos_c.any() else 0.0

        if pos_t.any():
            _fit_regressor(self.reg_t_, X[pos_t], np.log1p(y[pos_t]), self.params, eval_set=eval_reg_t)
        if pos_c.any():
            _fit_regressor(self.reg_c_, X[pos_c], np.log1p(y[pos_c]), self.params, eval_set=eval_reg_c)
        return self

    def _expected(self, X: pd.DataFrame, group: str) -> np.ndarray:
        if group == "t":
            assert self.clf_t_ is not None and self.reg_t_ is not None
            p = self.clf_t_.predict_proba(X)[:, 1]
            amount = _safe_expm1(self.reg_t_.predict(X)) if hasattr(self.reg_t_, "booster_") else self.mean_pos_t_
        else:
            assert self.clf_c_ is not None and self.reg_c_ is not None
            p = self.clf_c_.predict_proba(X)[:, 1]
            amount = _safe_expm1(self.reg_c_.predict(X)) if hasattr(self.reg_c_, "booster_") else self.mean_pos_c_
        return p * amount

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._expected(X, "t") - self._expected(X, "c")

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        imps = []
        for model in (self.clf_t_, self.clf_c_, self.reg_t_, self.reg_c_):
            if model is not None and hasattr(model, "feature_importances_"):
                imps.append(model.feature_importances_)
        if not imps:
            return pd.Series(dtype=float)
        return pd.Series(np.mean(imps, axis=0), index=feature_names).sort_values(ascending=False)


class RLearner(BaseUpliftModel):
    """R-learner with constant randomized propensity."""

    name = "r_learner"

    def __init__(self, params: LGBMParams, seed_offset: int = 0) -> None:
        self.params = params
        self.seed_offset = seed_offset
        self.mu_: lgb.LGBMRegressor | None = None
        self.tau_: lgb.LGBMRegressor | None = None
        self.propensity_: float = 0.5

    def fit(self, X, y, treatment, *, eval_set=None) -> RLearner:
        y = np.asarray(y, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        self.propensity_ = float(treatment.mean())

        self.mu_ = _make_regressor(self.params, self.seed_offset + 21)
        eval_mu = None
        if eval_set is not None:
            X_val, y_val, _ = eval_set
            eval_mu = (X_val, np.asarray(y_val, dtype=float))
        _fit_regressor(self.mu_, X, y, self.params, eval_set=eval_mu)

        mu_fit = self.mu_.predict(X)
        w = treatment - self.propensity_
        denom = np.where(np.abs(w) < 1e-3, np.sign(w + 1e-9) * 1e-3, w)
        z = (y - mu_fit) / denom
        sample_weight = w ** 2

        self.tau_ = _make_regressor(self.params, self.seed_offset + 22)
        eval_tau = eval_w = None
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            t_val = np.asarray(t_val, dtype=float)
            mu_val = self.mu_.predict(X_val)
            w_val = t_val - self.propensity_
            denom_val = np.where(np.abs(w_val) < 1e-3, np.sign(w_val + 1e-9) * 1e-3, w_val)
            eval_tau = (X_val, (np.asarray(y_val, dtype=float) - mu_val) / denom_val)
            eval_w = w_val ** 2
        _fit_regressor(self.tau_, X, z, self.params, eval_set=eval_tau, sample_weight=sample_weight)
        if eval_w is not None:
            pass
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.tau_ is not None
        return self.tau_.predict(X)

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        assert self.mu_ is not None and self.tau_ is not None
        imp = (self.mu_.feature_importances_ + self.tau_.feature_importances_) / 2
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)


class EnsembleUpliftModel(BaseUpliftModel):
    """Weighted ensemble. Rank mode focuses on stable ordering for uplift@10."""

    name = "ensemble"

    def __init__(
        self,
        models: list[BaseUpliftModel],
        weights: list[float],
        *,
        rank_average: bool = False,
    ) -> None:
        self.models = models
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.sum() <= 0:
            weights_arr = np.ones(len(models), dtype=float)
        self.weights = weights_arr / weights_arr.sum()
        self.rank_average = rank_average
        self.sub_names = [m.name for m in models]

    def fit(self, X, y, treatment, *, eval_set=None) -> EnsembleUpliftModel:
        for model in self.models:
            model.fit(X, y, treatment, eval_set=eval_set)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([model.predict(X) for model in self.models])
        if self.rank_average:
            preds = np.column_stack([_rank_percentile(preds[:, i]) for i in range(preds.shape[1])])
        return preds @ self.weights


def create_model(name: ModelName, params: LGBMParams) -> BaseUpliftModel:
    base_name, seed_offset = _parse_model_name(name)
    if base_name == "t_learner_log":
        model = TwoModelRegressor(params, seed_offset)
    elif base_name == "s_learner_log":
        model = SoloRegressor(params, seed_offset)
    elif base_name == "transformed_outcome":
        model = TransformedOutcomeRegressor(params, seed_offset)
    elif base_name == "hurdle_t_learner":
        model = HurdleTwoModelRegressor(params, seed_offset)
    elif base_name == "r_learner":
        model = RLearner(params, seed_offset)
    else:
        raise ValueError(f"Unknown model: {name}")
    model.name = name
    return model


ALL_MODELS: tuple[ModelName, ...] = (
    "hurdle_t_learner",
    "t_learner_log",
    "s_learner_log",
    "transformed_outcome",
    "r_learner",
    "hurdle_t_learner_s101",
    "t_learner_log_s101",
)
