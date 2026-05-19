"""Кросс-валидация и выбор лучшей uplift-модели."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import PipelineConfig
from metrics import evaluate_all_metrics, uplift_auc_score
from modeling import (
    ALL_MODELS,
    BaseUpliftModel,
    EnsembleUpliftModel,
    ModelName,
    create_model,
)

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    model_name: str
    fold_metrics: pd.DataFrame
    summary: dict[str, float]
    oof_uplift: np.ndarray = field(repr=False)
    fold_models: list[BaseUpliftModel] = field(default_factory=list, repr=False)


@dataclass
class EvaluationReport:
    best_model_name: str
    best_cv: CVResult
    all_results: dict[str, CVResult]
    oof_uplift: np.ndarray
    y: np.ndarray
    treatment: np.ndarray
    ensemble_weights: dict[str, float]


def stratified_holdout_split(
    indices: np.ndarray,
    treatment: np.ndarray,
    y: np.ndarray | None = None,
    *,
    val_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Разбивает indices на fit / early-stopping с сохранением долей treatment_flg и target.

    Использует train_test_split(..., stratify=treatment/target strata).
    """
    if val_fraction <= 0 or val_fraction >= 1:
        raise ValueError("val_fraction должен быть в (0, 1)")

    t_sub = treatment[indices]
    strata = t_sub if y is None else t_sub.astype(np.int8) * 2 + np.asarray(y)[indices].astype(np.int8)
    n_classes = len(np.unique(strata))
    n_val = max(n_classes, int(round(len(indices) * val_fraction)))

    if n_val >= len(indices) - n_classes:
        raise ValueError(
            f"Слишком мало объектов ({len(indices)}) для holdout при val_fraction={val_fraction}"
        )

    fit_rel, es_rel = train_test_split(
        np.arange(len(indices)),
        test_size=n_val,
        stratify=strata,
        random_state=random_state,
    )
    return indices[fit_rel], indices[es_rel]


def _log_treatment_balance(
    label: str,
    treatment: np.ndarray,
    reference_rate: float | None = None,
) -> None:
    rate = float(treatment.mean())
    msg = f"    {label}: n={len(treatment):,}, treatment_rate={rate:.4f}"
    if reference_rate is not None:
        msg += f" (delta vs full={rate - reference_rate:+.4f})"
    logger.debug(msg)


def _run_cv_single(
    model_name: ModelName,
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    cfg: PipelineConfig,
) -> CVResult:
    skf = StratifiedKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state
    )
    oof = np.zeros(len(y))
    fold_rows: list[dict] = []
    fold_models: list[BaseUpliftModel] = []
    global_treat_rate = float(treatment.mean())

    strata = treatment.astype(np.int8) * 2 + y.astype(np.int8)

    for fold, (tr_idx, oof_val_idx) in enumerate(skf.split(X, strata), start=1):
        # Внутри train-части фолда: отдельный стратифицированный holdout для early stopping
        fit_idx, es_idx = stratified_holdout_split(
            tr_idx,
            treatment,
            y,
            val_fraction=cfg.early_stopping_val_fraction,
            random_state=cfg.random_state + fold,
        )

        model = create_model(model_name, cfg.lgbm)

        X_fit = X.iloc[fit_idx]
        y_fit = y[fit_idx]
        t_fit = treatment[fit_idx]

        X_es = X.iloc[es_idx]
        y_es = y[es_idx]
        t_es = treatment[es_idx]

        X_oof = X.iloc[oof_val_idx]
        y_oof = y[oof_val_idx]
        t_oof = treatment[oof_val_idx]

        if fold == 1 and model_name == ALL_MODELS[0]:
            logger.debug("Проверка баланса treatment_flg (fold 1):")
            _log_treatment_balance("full", treatment)
            _log_treatment_balance("oof_val", t_oof, global_treat_rate)
            _log_treatment_balance("fit", t_fit, global_treat_rate)
            _log_treatment_balance("early_stop", t_es, global_treat_rate)

        model.fit(X_fit, y_fit, t_fit, eval_set=(X_es, y_es, t_es))
        uplift_oof = model.predict(X_oof)
        oof[oof_val_idx] = uplift_oof

        metrics = evaluate_all_metrics(
            y_oof, uplift_oof, t_oof,
            k=cfg.top_k, margin=cfg.margin, cost=cfg.treatment_cost,
            k_grid=cfg.uplift_k_grid,
        )
        metrics["fold"] = fold
        fold_rows.append(metrics)
        fold_models.append(model)

        logger.info(
            "  [%s] fold %d: AUUC=%.4f  Qini=%.4f  uplift@30%%=%.4f  IRR=%.2f",
            model_name, fold,
            metrics["auuc"], metrics["qini"],
            metrics[f"uplift_at_{int(cfg.top_k * 100)}pct"],
            metrics[f"irr_at_{int(cfg.top_k * 100)}pct"],
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = evaluate_all_metrics(
        y,
        oof,
        treatment,
        k=cfg.top_k,
        margin=cfg.margin,
        cost=cfg.treatment_cost,
        k_grid=cfg.uplift_k_grid,
    )
    summary = {k: float(v) for k, v in summary.items()}

    return CVResult(
        model_name=model_name,
        fold_metrics=fold_df,
        summary=summary,
        oof_uplift=oof,
        fold_models=fold_models,
    )


def cross_validate_all_models(
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    cfg: PipelineConfig,
) -> dict[str, CVResult]:
    logger.info(
        "CV: %d folds | OOF-валидация на 1/%d данных | "
        "early stopping на %.0f%% train-части (stratify=treatment+target)",
        cfg.n_folds,
        cfg.n_folds,
        cfg.early_stopping_val_fraction * 100,
    )
    results: dict[str, CVResult] = {}
    for name in ALL_MODELS:
        logger.info("Кросс-валидация модели: %s", name)
        results[name] = _run_cv_single(name, X, y, treatment, cfg)
        logger.info(
            "  >> OOF AUUC=%.4f  Qini=%.4f",
            results[name].summary["auuc"],
            results[name].summary["qini"],
        )
    return results


def build_ensemble_cv(
    results: dict[str, CVResult],
    y: np.ndarray,
    treatment: np.ndarray,
    cfg: PipelineConfig,
    *,
    rank_average: bool = False,
    top_n: int | None = None,
) -> CVResult:
    """Ансамбль с весами пропорционально AUUC на OOF."""
    names = list(results.keys())
    if top_n is not None and top_n < len(names):
        names = sorted(names, key=lambda n: results[n].summary["auuc"], reverse=True)[:top_n]
    auuc_scores = np.array([uplift_auc_score(y, results[n].oof_uplift, treatment) for n in names])
    weights = np.maximum(auuc_scores, 1e-6)
    weights /= weights.sum()

    oof_stack = np.column_stack([results[n].oof_uplift for n in names])
    if rank_average:
        oof_stack = np.column_stack(
            [
                pd.Series(oof_stack[:, i]).rank(method="average", pct=True).to_numpy(dtype=float)
                for i in range(oof_stack.shape[1])
            ]
        )
    oof_ensemble = oof_stack @ weights

    metrics = evaluate_all_metrics(
        y, oof_ensemble, treatment,
        k=cfg.top_k, margin=cfg.margin, cost=cfg.treatment_cost,
        k_grid=cfg.uplift_k_grid,
    )

    summary = {k: float(v) for k, v in metrics.items()}
    summary["weights"] = dict(zip(names, weights.tolist()))

    fold_df = pd.DataFrame([{**metrics, "fold": 0}])

    return CVResult(
        model_name=(
            "rank_top_ensemble" if rank_average and top_n is not None
            else "top_ensemble" if top_n is not None
            else "rank_ensemble" if rank_average
            else "ensemble"
        ),
        fold_metrics=fold_df,
        summary=summary,
        oof_uplift=oof_ensemble,
    )


def select_best_model(
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    cfg: PipelineConfig,
) -> EvaluationReport:
    all_cv = cross_validate_all_models(X, y, treatment, cfg)
    ensemble_cv = build_ensemble_cv(all_cv, y, treatment, cfg)
    rank_ensemble_cv = build_ensemble_cv(all_cv, y, treatment, cfg, rank_average=True)
    top_ensemble_cv = build_ensemble_cv(all_cv, y, treatment, cfg, top_n=4)
    rank_top_ensemble_cv = build_ensemble_cv(all_cv, y, treatment, cfg, rank_average=True, top_n=4)
    all_cv["ensemble"] = ensemble_cv
    all_cv["rank_ensemble"] = rank_ensemble_cv
    all_cv["top_ensemble"] = top_ensemble_cv
    all_cv["rank_top_ensemble"] = rank_top_ensemble_cv

    best_name = max(all_cv, key=lambda n: all_cv[n].summary["auuc"])
    logger.info("Лучшая модель по AUUC: %s (AUUC=%.4f)", best_name, all_cv[best_name].summary["auuc"])

    ensemble_weights: dict[str, float] = all_cv[best_name].summary.get("weights", {})  # type: ignore[assignment]

    return EvaluationReport(
        best_model_name=best_name,
        best_cv=all_cv[best_name],
        all_results=all_cv,
        oof_uplift=all_cv[best_name].oof_uplift,
        y=y,
        treatment=treatment,
        ensemble_weights=ensemble_weights,
    )


def train_final_model(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    cfg: PipelineConfig,
    *,
    sub_weights: dict[str, float] | None = None,
) -> BaseUpliftModel:
    indices = np.arange(len(y))
    fit_idx, es_idx = stratified_holdout_split(
        indices,
        treatment,
        y,
        val_fraction=cfg.early_stopping_val_fraction,
        random_state=cfg.random_state,
    )

    X_fit, y_fit, t_fit = X.iloc[fit_idx], y[fit_idx], treatment[fit_idx]
    X_es, y_es, t_es = X.iloc[es_idx], y[es_idx], treatment[es_idx]

    logger.info(
        "Финальное обучение: fit=%s, early_stop=%s (stratify=treatment+target, rate_fit=%.4f, rate_es=%.4f)",
        f"{len(fit_idx):,}",
        f"{len(es_idx):,}",
        t_fit.mean(),
        t_es.mean(),
    )

    if model_name in {"ensemble", "rank_ensemble", "top_ensemble", "rank_top_ensemble"}:
        assert sub_weights
        models = [create_model(n, cfg.lgbm) for n in sub_weights]
        weights = [sub_weights[n] for n in sub_weights]
        ensemble = EnsembleUpliftModel(
            models,
            weights,
            rank_average=model_name in {"rank_ensemble", "rank_top_ensemble"},
        )
        ensemble.fit(X_fit, y_fit, t_fit, eval_set=(X_es, y_es, t_es))
        return ensemble

    model = create_model(model_name, cfg.lgbm)  # type: ignore[arg-type]
    model.fit(X_fit, y_fit, t_fit, eval_set=(X_es, y_es, t_es))
    return model
