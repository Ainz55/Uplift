"""Training uplift models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import LGBMParams, RANDOM_STATE

ModelName = str


def _parse_model_name(name: str) -> tuple[str, int]:
    """Return base model name and deterministic seed offset from names like model_s101."""
    if "_s" not in name:
        return name, 0
    base, seed = name.rsplit("_s", 1)
    if seed.isdigit():
        return base, int(seed)
    return name, 0


def _make_lgbm_classifier(params: LGBMParams, seed_offset: int = 0) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        max_depth=params.max_depth,
        num_leaves=params.num_leaves,
        min_child_samples=params.min_child_samples,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        random_state=RANDOM_STATE + seed_offset,
        verbose=-1,
        n_jobs=-1,
        force_col_wise=True,
    )


def _make_lgbm_regressor(params: LGBMParams, seed_offset: int = 0) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        max_depth=params.max_depth,
        num_leaves=params.num_leaves,
        min_child_samples=params.min_child_samples,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        random_state=RANDOM_STATE + seed_offset,
        verbose=-1,
        n_jobs=-1,
        force_col_wise=True,
    )


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


class TwoModelsLearner(BaseUpliftModel):
    """T-learner: separate outcome models for treatment and control."""

    name = "two_models"

    def __init__(self, params: LGBMParams, seed_offset: int = 0) -> None:
        self.params = params
        self.seed_offset = seed_offset
        self.model_t_: lgb.LGBMClassifier | None = None
        self.model_c_: lgb.LGBMClassifier | None = None

    def fit(self, X, y, treatment, *, eval_set=None) -> TwoModelsLearner:
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        treatment = np.asarray(treatment)

        mask_t = treatment == 1
        mask_c = ~mask_t

        self.model_t_ = _make_lgbm_classifier(self.params, self.seed_offset + 1)
        self.model_c_ = _make_lgbm_classifier(self.params, self.seed_offset + 2)

        fit_kwargs_t: dict = {}
        fit_kwargs_c: dict = {}

        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            fit_kwargs_t["eval_set"] = [(X_val[t_val == 1], y_val[t_val == 1])]
            fit_kwargs_c["eval_set"] = [(X_val[t_val == 0], y_val[t_val == 0])]
            rounds = self.params.early_stopping_rounds
            fit_kwargs_t["callbacks"] = [lgb.early_stopping(rounds, verbose=False)]
            fit_kwargs_c["callbacks"] = [lgb.early_stopping(rounds, verbose=False)]

        self.model_t_.fit(X[mask_t], y[mask_t], **fit_kwargs_t)
        self.model_c_.fit(X[mask_c], y[mask_c], **fit_kwargs_c)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_t_ is not None and self.model_c_ is not None
        p1 = self.model_t_.predict_proba(X)[:, 1]
        p0 = self.model_c_.predict_proba(X)[:, 1]
        return p1 - p0

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        assert self.model_t_ and self.model_c_
        imp = (self.model_t_.feature_importances_ + self.model_c_.feature_importances_) / 2
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)


class ClassTransformationLearner(BaseUpliftModel):
    """Class transformation learner."""

    name = "class_transformation"

    def __init__(self, params: LGBMParams, seed_offset: int = 0) -> None:
        self.params = params
        self.seed_offset = seed_offset
        self.model_: lgb.LGBMClassifier | None = None

    @staticmethod
    def _transform_target(y: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        return np.where((treatment == 1) & (y == 1), 1, np.where((treatment == 0) & (y == 0), 1, 0))

    def fit(self, X, y, treatment, *, eval_set=None) -> ClassTransformationLearner:
        y = np.asarray(y)
        treatment = np.asarray(treatment)
        z = self._transform_target(y, treatment)

        self.model_ = _make_lgbm_classifier(self.params, self.seed_offset + 3)
        fit_kwargs: dict = {}
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            z_val = self._transform_target(np.asarray(y_val), np.asarray(t_val))
            fit_kwargs["eval_set"] = [(X_val, z_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.params.early_stopping_rounds, verbose=False)
            ]

        self.model_.fit(X, z, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        return 2.0 * self.model_.predict_proba(X)[:, 1] - 1.0

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        assert self.model_
        return pd.Series(self.model_.feature_importances_, index=feature_names).sort_values(
            ascending=False
        )


class SoloModelLearner(BaseUpliftModel):
    """S-learner: one response model with treatment as a feature."""

    name = "solo_model"

    def __init__(self, params: LGBMParams, seed_offset: int = 0) -> None:
        self.params = params
        self.seed_offset = seed_offset
        self.model_: lgb.LGBMClassifier | None = None

    def fit(self, X, y, treatment, *, eval_set=None) -> SoloModelLearner:
        X_aug = X.copy()
        X_aug["treatment_flg"] = treatment
        self.model_ = _make_lgbm_classifier(self.params, self.seed_offset + 4)

        fit_kwargs: dict = {}
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            Xv = X_val.copy()
            Xv["treatment_flg"] = t_val
            fit_kwargs["eval_set"] = [(Xv, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.params.early_stopping_rounds, verbose=False)
            ]

        self.model_.fit(X_aug, y, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_
        X1 = X.copy()
        X1["treatment_flg"] = 1
        X0 = X.copy()
        X0["treatment_flg"] = 0
        return self.model_.predict_proba(X1)[:, 1] - self.model_.predict_proba(X0)[:, 1]

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        assert self.model_
        imp = pd.Series(self.model_.feature_importances_, index=feature_names + ["treatment_flg"])
        return imp.drop(index="treatment_flg", errors="ignore").sort_values(ascending=False)


class TransformedOutcomeLearner(BaseUpliftModel):
    """Transformed outcome learner for randomized experiments."""

    name = "transformed_outcome"

    def __init__(self, params: LGBMParams, seed_offset: int = 0) -> None:
        self.params = params
        self.seed_offset = seed_offset
        self.model_: lgb.LGBMRegressor | None = None
        self.propensity_: float = 0.5

    @staticmethod
    def _target(y: np.ndarray, treatment: np.ndarray, propensity: float) -> np.ndarray:
        propensity = float(np.clip(propensity, 1e-3, 1 - 1e-3))
        return y * (treatment / propensity - (1 - treatment) / (1 - propensity))

    def fit(self, X, y, treatment, *, eval_set=None) -> TransformedOutcomeLearner:
        y = np.asarray(y, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        self.propensity_ = float(treatment.mean())
        z = self._target(y, treatment, self.propensity_)
        self.model_ = _make_lgbm_regressor(self.params, self.seed_offset + 5)

        fit_kwargs: dict = {}
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            z_val = self._target(np.asarray(y_val, dtype=float), np.asarray(t_val, dtype=float), self.propensity_)
            fit_kwargs["eval_set"] = [(X_val, z_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.params.early_stopping_rounds, verbose=False)
            ]

        self.model_.fit(X, z, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict(X)

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        assert self.model_
        return pd.Series(self.model_.feature_importances_, index=feature_names).sort_values(
            ascending=False
        )


class XLearner(BaseUpliftModel):
    """X-learner with LightGBM outcome and treatment-effect models."""

    name = "x_learner"

    def __init__(self, params: LGBMParams, seed_offset: int = 0) -> None:
        self.params = params
        self.seed_offset = seed_offset
        self.mu_t_: lgb.LGBMClassifier | None = None
        self.mu_c_: lgb.LGBMClassifier | None = None
        self.tau_t_: lgb.LGBMRegressor | None = None
        self.tau_c_: lgb.LGBMRegressor | None = None
        self.propensity_: float = 0.5

    def fit(self, X, y, treatment, *, eval_set=None) -> XLearner:
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        treatment = np.asarray(treatment)
        self.propensity_ = float(treatment.mean())

        mask_t = treatment == 1
        mask_c = ~mask_t

        self.mu_t_ = _make_lgbm_classifier(self.params, self.seed_offset + 11)
        self.mu_c_ = _make_lgbm_classifier(self.params, self.seed_offset + 12)
        fit_kwargs_t: dict = {}
        fit_kwargs_c: dict = {}
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            fit_kwargs_t["eval_set"] = [(X_val[t_val == 1], y_val[t_val == 1])]
            fit_kwargs_c["eval_set"] = [(X_val[t_val == 0], y_val[t_val == 0])]
            rounds = self.params.early_stopping_rounds
            fit_kwargs_t["callbacks"] = [lgb.early_stopping(rounds, verbose=False)]
            fit_kwargs_c["callbacks"] = [lgb.early_stopping(rounds, verbose=False)]

        self.mu_t_.fit(X[mask_t], y[mask_t], **fit_kwargs_t)
        self.mu_c_.fit(X[mask_c], y[mask_c], **fit_kwargs_c)

        d_t = y[mask_t] - self.mu_c_.predict_proba(X[mask_t])[:, 1]
        d_c = self.mu_t_.predict_proba(X[mask_c])[:, 1] - y[mask_c]

        self.tau_t_ = _make_lgbm_regressor(self.params, self.seed_offset + 13)
        self.tau_c_ = _make_lgbm_regressor(self.params, self.seed_offset + 14)

        reg_kwargs_t: dict = {}
        reg_kwargs_c: dict = {}
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            v_t = t_val == 1
            v_c = t_val == 0
            if v_t.any():
                d_val_t = y_val[v_t] - self.mu_c_.predict_proba(X_val[v_t])[:, 1]
                reg_kwargs_t["eval_set"] = [(X_val[v_t], d_val_t)]
                reg_kwargs_t["callbacks"] = [lgb.early_stopping(self.params.early_stopping_rounds, verbose=False)]
            if v_c.any():
                d_val_c = self.mu_t_.predict_proba(X_val[v_c])[:, 1] - y_val[v_c]
                reg_kwargs_c["eval_set"] = [(X_val[v_c], d_val_c)]
                reg_kwargs_c["callbacks"] = [lgb.early_stopping(self.params.early_stopping_rounds, verbose=False)]

        self.tau_t_.fit(X[mask_t], d_t, **reg_kwargs_t)
        self.tau_c_.fit(X[mask_c], d_c, **reg_kwargs_c)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.tau_t_ is not None and self.tau_c_ is not None
        tau_t = self.tau_t_.predict(X)
        tau_c = self.tau_c_.predict(X)
        p = np.clip(self.propensity_, 1e-3, 1 - 1e-3)
        return (1.0 - p) * tau_t + p * tau_c

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        assert self.mu_t_ and self.mu_c_ and self.tau_t_ and self.tau_c_
        imp = (
            self.mu_t_.feature_importances_
            + self.mu_c_.feature_importances_
            + self.tau_t_.feature_importances_
            + self.tau_c_.feature_importances_
        ) / 4
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)


class RLearner(BaseUpliftModel):
    """R-learner: residualized outcome model with constant randomized propensity."""

    name = "r_learner"

    def __init__(self, params: LGBMParams, seed_offset: int = 0) -> None:
        self.params = params
        self.seed_offset = seed_offset
        self.mu_: lgb.LGBMClassifier | None = None
        self.tau_: lgb.LGBMRegressor | None = None
        self.propensity_: float = 0.5

    @staticmethod
    def _pseudo_outcome(
        y: np.ndarray,
        treatment: np.ndarray,
        mu: np.ndarray,
        propensity: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        p = float(np.clip(propensity, 1e-3, 1 - 1e-3))
        w = treatment.astype(float) - p
        z = (y.astype(float) - mu) / np.where(np.abs(w) < 1e-3, np.sign(w) * 1e-3, w)
        sample_weight = w ** 2
        return z, sample_weight

    def fit(self, X, y, treatment, *, eval_set=None) -> RLearner:
        y = np.asarray(y, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        self.propensity_ = float(treatment.mean())

        self.mu_ = _make_lgbm_classifier(self.params, self.seed_offset + 21)
        fit_kwargs: dict = {}
        if eval_set is not None:
            X_val, y_val, _ = eval_set
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.params.early_stopping_rounds, verbose=False)
            ]
        self.mu_.fit(X, y, **fit_kwargs)

        mu_fit = self.mu_.predict_proba(X)[:, 1]
        z_fit, w_fit = self._pseudo_outcome(y, treatment, mu_fit, self.propensity_)

        self.tau_ = _make_lgbm_regressor(self.params, self.seed_offset + 22)
        reg_kwargs: dict = {}
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            mu_val = self.mu_.predict_proba(X_val)[:, 1]
            z_val, w_val = self._pseudo_outcome(
                np.asarray(y_val, dtype=float),
                np.asarray(t_val, dtype=float),
                mu_val,
                self.propensity_,
            )
            reg_kwargs["eval_set"] = [(X_val, z_val)]
            reg_kwargs["eval_sample_weight"] = [w_val]
            reg_kwargs["callbacks"] = [
                lgb.early_stopping(self.params.early_stopping_rounds, verbose=False)
            ]
        self.tau_.fit(X, z_fit, sample_weight=w_fit, **reg_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.tau_ is not None
        return self.tau_.predict(X)

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        assert self.mu_ and self.tau_
        imp = (self.mu_.feature_importances_ + self.tau_.feature_importances_) / 2
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)


class DRLearner(BaseUpliftModel):
    """Doubly robust learner: T-learner nuisance models plus pseudo-outcome regression."""

    name = "dr_learner"

    def __init__(self, params: LGBMParams, seed_offset: int = 0) -> None:
        self.params = params
        self.seed_offset = seed_offset
        self.mu_t_: lgb.LGBMClassifier | None = None
        self.mu_c_: lgb.LGBMClassifier | None = None
        self.tau_: lgb.LGBMRegressor | None = None
        self.propensity_: float = 0.5

    @staticmethod
    def _pseudo_outcome(
        y: np.ndarray,
        treatment: np.ndarray,
        mu_t: np.ndarray,
        mu_c: np.ndarray,
        propensity: float,
    ) -> np.ndarray:
        p = float(np.clip(propensity, 1e-3, 1 - 1e-3))
        t = treatment.astype(float)
        return (
            mu_t
            - mu_c
            + t / p * (y.astype(float) - mu_t)
            - (1.0 - t) / (1.0 - p) * (y.astype(float) - mu_c)
        )

    def fit(self, X, y, treatment, *, eval_set=None) -> DRLearner:
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        treatment = np.asarray(treatment)
        self.propensity_ = float(treatment.mean())

        mask_t = treatment == 1
        mask_c = ~mask_t
        self.mu_t_ = _make_lgbm_classifier(self.params, self.seed_offset + 31)
        self.mu_c_ = _make_lgbm_classifier(self.params, self.seed_offset + 32)

        fit_kwargs_t: dict = {}
        fit_kwargs_c: dict = {}
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            fit_kwargs_t["eval_set"] = [(X_val[t_val == 1], y_val[t_val == 1])]
            fit_kwargs_c["eval_set"] = [(X_val[t_val == 0], y_val[t_val == 0])]
            rounds = self.params.early_stopping_rounds
            fit_kwargs_t["callbacks"] = [lgb.early_stopping(rounds, verbose=False)]
            fit_kwargs_c["callbacks"] = [lgb.early_stopping(rounds, verbose=False)]

        self.mu_t_.fit(X[mask_t], y[mask_t], **fit_kwargs_t)
        self.mu_c_.fit(X[mask_c], y[mask_c], **fit_kwargs_c)

        mu_t_fit = self.mu_t_.predict_proba(X)[:, 1]
        mu_c_fit = self.mu_c_.predict_proba(X)[:, 1]
        z_fit = self._pseudo_outcome(y, treatment.astype(float), mu_t_fit, mu_c_fit, self.propensity_)

        self.tau_ = _make_lgbm_regressor(self.params, self.seed_offset + 33)
        reg_kwargs: dict = {}
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            mu_t_val = self.mu_t_.predict_proba(X_val)[:, 1]
            mu_c_val = self.mu_c_.predict_proba(X_val)[:, 1]
            z_val = self._pseudo_outcome(
                np.asarray(y_val, dtype=float),
                np.asarray(t_val, dtype=float),
                mu_t_val,
                mu_c_val,
                self.propensity_,
            )
            reg_kwargs["eval_set"] = [(X_val, z_val)]
            reg_kwargs["callbacks"] = [
                lgb.early_stopping(self.params.early_stopping_rounds, verbose=False)
            ]

        self.tau_.fit(X, z_fit, **reg_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.tau_ is not None
        return self.tau_.predict(X)

    def feature_importances_(self, feature_names: list[str]) -> pd.Series:
        assert self.mu_t_ and self.mu_c_ and self.tau_
        imp = (
            self.mu_t_.feature_importances_
            + self.mu_c_.feature_importances_
            + self.tau_.feature_importances_
        ) / 3
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)


class EnsembleUpliftModel(BaseUpliftModel):
    """Weighted ensemble. Rank mode optimizes ordering stability for AUUC/Qini submissions."""

    name = "ensemble"

    def __init__(
        self,
        models: list[BaseUpliftModel],
        weights: list[float],
        *,
        rank_average: bool = False,
    ) -> None:
        self.models = models
        w = np.asarray(weights, dtype=float)
        self.weights = w / w.sum()
        self.rank_average = rank_average
        self.sub_names = [m.name for m in models]

    def fit(self, X, y, treatment, *, eval_set=None) -> EnsembleUpliftModel:
        for m in self.models:
            m.fit(X, y, treatment, eval_set=eval_set)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self.models])
        if self.rank_average:
            preds = np.column_stack([_rank_percentile(preds[:, i]) for i in range(preds.shape[1])])
        return preds @ self.weights


def create_model(name: ModelName, params: LGBMParams) -> BaseUpliftModel:
    base_name, seed_offset = _parse_model_name(name)
    if base_name == "two_models":
        model = TwoModelsLearner(params, seed_offset)
    elif base_name == "class_transformation":
        model = ClassTransformationLearner(params, seed_offset)
    elif base_name == "solo_model":
        model = SoloModelLearner(params, seed_offset)
    elif base_name == "transformed_outcome":
        model = TransformedOutcomeLearner(params, seed_offset)
    elif base_name == "x_learner":
        model = XLearner(params, seed_offset)
    elif base_name == "r_learner":
        model = RLearner(params, seed_offset)
    elif base_name == "dr_learner":
        model = DRLearner(params, seed_offset)
    else:
        raise ValueError(f"Unknown model: {name}")
    model.name = name
    return model


ALL_MODELS: tuple[ModelName, ...] = (
    "two_models",
    "class_transformation",
    "solo_model",
    "transformed_outcome",
    "x_learner",
    "r_learner",
    "dr_learner",
    "two_models_s101",
    "class_transformation_s101",
    "class_transformation_s202",
    "transformed_outcome_s101",
    "dr_learner_s101",
)
