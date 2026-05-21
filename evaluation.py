"""Cross-validation and model selection for continuous uplift."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import PipelineConfig
from metrics import evaluate_all_metrics
from modeling import ALL_MODELS, BaseUpliftModel, EnsembleUpliftModel, ModelName, create_model

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    model_name: str
    fold_metrics: pd.DataFrame
    summary: dict[str, float]
    oof_uplift: np.ndarray = field(repr=False)


@dataclass
class EvaluationReport:
    best_model_name: str
    best_cv: CVResult
    all_results: dict[str, CVResult]
    oof_uplift: np.ndarray
    y: np.ndarray
    treatment: np.ndarray
    ensemble_weights: dict[str, float]


def make_strata(treatment: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stable strata for randomized treatment and zero-heavy continuous outcome."""
    treatment = np.asarray(treatment, dtype=np.int8)
    y = np.asarray(y, dtype=float)
    nonzero = (y > 0).astype(np.int8)
    return treatment * 2 + nonzero


def _valid_strata(strata: np.ndarray, n_splits: int) -> bool:
    _, counts = np.unique(strata, return_counts=True)
    return bool(len(counts) > 1 and counts.min() >= n_splits)


def stratified_holdout_split(
    indices: np.ndarray,
    treatment: np.ndarray,
    y: np.ndarray,
    *,
    val_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if val_fraction <= 0 or val_fraction >= 1:
        raise ValueError("val_fraction must be in (0, 1)")

    strata = make_strata(treatment[indices], y[indices])
    if not _valid_strata(strata, 2):
        strata = treatment[indices]

    fit_rel, es_rel = train_test_split(
        np.arange(len(indices)),
        test_size=val_fraction,
        stratify=strata,
        random_state=random_state,
    )
    return indices[fit_rel], indices[es_rel]


def _metric_key(cfg: PipelineConfig) -> str:
    return f"uplift_at_{int(round(cfg.top_k * 100))}pct_ci_lower"


def _evaluate(
    y: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    cfg: PipelineConfig,
    *,
    seed_offset: int = 0,
) -> dict[str, float]:
    return evaluate_all_metrics(
        y,
        uplift,
        treatment,
        k=cfg.top_k,
        k_grid=cfg.uplift_k_grid,
        bootstrap_iterations=cfg.bootstrap_iterations,
        bootstrap_ci=cfg.bootstrap_ci,
        random_state=cfg.random_state + seed_offset,
    )


def _run_cv_single(
    model_name: ModelName,
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    cfg: PipelineConfig,
) -> CVResult:
    strata = make_strata(treatment, y)
    if not _valid_strata(strata, cfg.n_folds):
        strata = treatment

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_rows: list[dict] = []

    for fold, (tr_idx, oof_idx) in enumerate(skf.split(X, strata), start=1):
        fit_idx, es_idx = stratified_holdout_split(
            tr_idx,
            treatment,
            y,
            val_fraction=cfg.early_stopping_val_fraction,
            random_state=cfg.random_state + fold,
        )

        model = create_model(model_name, cfg.lgbm)
        model.fit(
            X.iloc[fit_idx],
            y[fit_idx],
            treatment[fit_idx],
            eval_set=(X.iloc[es_idx], y[es_idx], treatment[es_idx]),
        )

        uplift_oof = model.predict(X.iloc[oof_idx])
        oof[oof_idx] = uplift_oof.astype(np.float32, copy=False)

        metrics = _evaluate(
            y[oof_idx],
            uplift_oof,
            treatment[oof_idx],
            cfg,
            seed_offset=fold * 100,
        )
        metrics["fold"] = fold
        fold_rows.append(metrics)

        main_pct = int(round(cfg.top_k * 100))
        logger.info(
            "  [%s] fold %d: uplift@%d%%=%.4f  lower80=%.4f  AUUC=%.4f",
            model_name,
            fold,
            main_pct,
            metrics[f"uplift_at_{main_pct}pct"],
            metrics[f"uplift_at_{main_pct}pct_ci_lower"],
            metrics["auuc"],
        )
        del model, uplift_oof
        gc.collect()

    fold_df = pd.DataFrame(fold_rows)
    summary = {k: float(v) for k, v in _evaluate(y, oof, treatment, cfg).items()}

    return CVResult(model_name=model_name, fold_metrics=fold_df, summary=summary, oof_uplift=oof)


def cross_validate_all_models(
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    cfg: PipelineConfig,
) -> dict[str, CVResult]:
    logger.info(
        "CV: %d folds | target=rec_spend | selection=lower %.0f%% CI of uplift@%.0f%%",
        cfg.n_folds,
        cfg.bootstrap_ci * 100,
        cfg.top_k * 100,
    )
    results: dict[str, CVResult] = {}
    model_names = cfg.model_names or ALL_MODELS
    for name in model_names:
        logger.info("Cross-validating model: %s", name)
        results[name] = _run_cv_single(name, X, y, treatment, cfg)
        logger.info(
            "  >> OOF uplift@%.0f%%=%.4f  lower80=%.4f",
            cfg.top_k * 100,
            results[name].summary[f"uplift_at_{int(round(cfg.top_k * 100))}pct"],
            results[name].summary[_metric_key(cfg)],
        )
    return results


def build_ensemble_cv(
    results: dict[str, CVResult],
    y: np.ndarray,
    treatment: np.ndarray,
    cfg: PipelineConfig,
    *,
    candidate_names: list[str] | tuple[str, ...] | None = None,
    rank_average: bool = False,
    top_n: int | None = None,
) -> CVResult:
    names = list(candidate_names) if candidate_names is not None else list(results.keys())
    key = _metric_key(cfg)
    if top_n is not None and top_n < len(names):
        names = sorted(names, key=lambda n: results[n].summary[key], reverse=True)[:top_n]

    raw_scores = np.array([results[n].summary[key] for n in names], dtype=float)
    weights = raw_scores - raw_scores.min() + 1e-6
    if weights.sum() <= 0:
        weights = np.ones(len(names), dtype=float)
    weights = weights / weights.sum()

    oof_stack = np.column_stack([results[n].oof_uplift for n in names])
    if rank_average:
        oof_stack = np.column_stack(
            [pd.Series(oof_stack[:, i]).rank(method="average", pct=True).to_numpy(float) for i in range(oof_stack.shape[1])]
        )
    oof_ensemble = oof_stack @ weights
    metrics = _evaluate(y, oof_ensemble, treatment, cfg)
    summary = {k: float(v) for k, v in metrics.items()}
    summary["weights"] = dict(zip(names, weights.tolist()))  # type: ignore[assignment]

    model_name = (
        "rank_top_ensemble" if rank_average and top_n is not None
        else "top_ensemble" if top_n is not None
        else "rank_ensemble" if rank_average
        else "ensemble"
    )
    return CVResult(
        model_name=model_name,
        fold_metrics=pd.DataFrame([{**metrics, "fold": 0}]),
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
    base_model_names = list(all_cv.keys())
    all_cv["ensemble"] = build_ensemble_cv(
        all_cv, y, treatment, cfg, candidate_names=base_model_names,
    )
    all_cv["rank_ensemble"] = build_ensemble_cv(
        all_cv, y, treatment, cfg, candidate_names=base_model_names, rank_average=True,
    )
    all_cv["top_ensemble"] = build_ensemble_cv(
        all_cv, y, treatment, cfg, candidate_names=base_model_names, top_n=4,
    )
    all_cv["rank_top_ensemble"] = build_ensemble_cv(
        all_cv, y, treatment, cfg, candidate_names=base_model_names, rank_average=True, top_n=4,
    )

    key = _metric_key(cfg)
    best_name = max(all_cv, key=lambda n: all_cv[n].summary[key])
    logger.info(
        "Best model by %s: %s (%.4f)",
        key,
        best_name,
        all_cv[best_name].summary[key],
    )
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

    if model_name in {"ensemble", "rank_ensemble", "top_ensemble", "rank_top_ensemble"}:
        assert sub_weights
        models = [create_model(name, cfg.lgbm) for name in sub_weights]
        weights = [sub_weights[name] for name in sub_weights]
        model = EnsembleUpliftModel(
            models,
            weights,
            rank_average=model_name in {"rank_ensemble", "rank_top_ensemble"},
        )
    else:
        model = create_model(model_name, cfg.lgbm)

    logger.info(
        "Final training: fit=%s, early_stop=%s, treatment_rate_fit=%.4f, treatment_rate_es=%.4f",
        f"{len(fit_idx):,}",
        f"{len(es_idx):,}",
        float(treatment[fit_idx].mean()),
        float(treatment[es_idx].mean()),
    )
    model.fit(
        X.iloc[fit_idx],
        y[fit_idx],
        treatment[fit_idx],
        eval_set=(X.iloc[es_idx], y[es_idx], treatment[es_idx]),
    )
    return model
