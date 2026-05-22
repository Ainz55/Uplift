"""Console reporting for training runs."""

from __future__ import annotations

import logging

import numpy as np

from config import PipelineConfig
from data import DatasetInfo
from evaluation import EvaluationReport

logger = logging.getLogger(__name__)

_SEP = "=" * 72
_SUB = "-" * 72


def print_dataset_info(info: DatasetInfo, n_features: int) -> None:
    logger.info(_SEP)
    logger.info("DATASET")
    logger.info(_SUB)
    logger.info("  Train rows:          %s", f"{info.n_train:,}")
    logger.info("  Test rows:           %s", f"{info.n_test:,}")
    logger.info("  Raw feature columns: %d", info.n_features_raw)
    logger.info("  Model features:      %d", n_features)
    logger.info("  Treatment rate:      %.2f%%", info.treatment_rate * 100)
    logger.info("  rec_spend mean:      %.4f", info.spend_mean)
    logger.info("  rec_spend zeros:     %.2f%%", info.spend_zero_rate * 100)
    logger.info("  Control mean spend:  %.4f", info.spend_control_mean)
    logger.info("  Treatment mean spend:%.4f", info.spend_treatment_mean)
    logger.info("  Average uplift:      %.4f", info.avg_uplift)


def print_models_comparison(report: EvaluationReport, cfg: PipelineConfig) -> None:
    logger.info(_SEP)
    logger.info("MODEL COMPARISON")
    logger.info(_SUB)
    pct = int(round(cfg.top_k * 100))
    key = f"uplift_at_{pct}pct_ci_lower"
    logger.info("  %-24s %12s %12s %12s", "model", f"uplift@{pct}", "lower80", "AUUC")
    logger.info("  " + "-" * 64)
    for name, cv in report.all_results.items():
        s = cv.summary
        logger.info(
            "  %-24s %12.4f %12.4f %12.4f",
            name,
            s.get(f"uplift_at_{pct}pct", float("nan")),
            s.get(key, float("nan")),
            s.get("auuc", float("nan")),
        )

    if report.ensemble_weights:
        logger.info(_SUB)
        logger.info("  Ensemble weights: %s", report.ensemble_weights)


def print_leaderboard_proxy(report: EvaluationReport, cfg: PipelineConfig) -> None:
    pct = int(round(cfg.top_k * 100))
    key = f"uplift_at_{pct}pct_ci_lower"
    raw_key = f"uplift_at_{pct}pct"
    s = report.best_cv.summary
    logger.info(_SEP)
    logger.info("LEADERBOARD_PROXY")
    logger.info(_SUB)
    logger.info("  Best model:      %s", report.best_model_name)
    logger.info("  Feature set:     %s", cfg.feature_set)
    logger.info("  Metric:          lower %.0f%% bootstrap CI for uplift@%d%%", cfg.bootstrap_ci * 100, pct)
    logger.info("  Proxy score:     %.5f", s.get(key, float("nan")))
    logger.info("  Raw uplift@%d%%:  %.5f", pct, s.get(raw_key, float("nan")))
    logger.info("  Target to beat:  20.00123")
    logger.info(
        "  Gap vs target:   %+.5f",
        s.get(key, float("nan")) - 20.00123,
    )


def print_metrics(title: str, metrics: dict[str, float], cfg: PipelineConfig) -> None:
    logger.info(_SEP)
    logger.info(title.upper())
    logger.info(_SUB)
    logger.info("  AUUC:          %.4f", metrics["auuc"])
    logger.info("  Qini AUC:      %.4f", metrics["qini"])
    logger.info("  Avg uplift:    %.4f", metrics["avg_uplift"])
    logger.info(_SUB)
    logger.info("  %-8s %12s %12s", "K", "uplift", "IRR")
    for k in cfg.uplift_k_grid:
        pct = int(round(k * 100))
        logger.info(
            "  %-8s %12.4f %12.2f",
            f"{pct}%",
            metrics.get(f"uplift_at_{pct}pct", float("nan")),
            metrics.get(f"irr_at_{pct}pct", float("nan")),
        )

    main_pct = int(round(cfg.top_k * 100))
    logger.info(_SUB)
    logger.info(
        "  Bootstrap uplift@%d%%: mean=%.4f, lower80=%.4f, upper80=%.4f, std=%.4f",
        main_pct,
        metrics.get(f"uplift_at_{main_pct}pct_bootstrap_mean", float("nan")),
        metrics.get(f"uplift_at_{main_pct}pct_ci_lower", float("nan")),
        metrics.get(f"uplift_at_{main_pct}pct_ci_upper", float("nan")),
        metrics.get(f"uplift_at_{main_pct}pct_bootstrap_std", float("nan")),
    )


def print_score_stats(uplift: np.ndarray, label: str) -> None:
    logger.info(_SUB)
    logger.info("  %s score distribution:", label)
    logger.info("    min:    %.4f", float(np.min(uplift)))
    logger.info("    q25:    %.4f", float(np.quantile(uplift, 0.25)))
    logger.info("    median: %.4f", float(np.median(uplift)))
    logger.info("    mean:   %.4f", float(np.mean(uplift)))
    logger.info("    q75:    %.4f", float(np.quantile(uplift, 0.75)))
    logger.info("    max:    %.4f", float(np.max(uplift)))
    logger.info("    std:    %.4f", float(np.std(uplift)))
    logger.info("    share > 0: %.1f%%", float((uplift > 0).mean() * 100))


def print_submission_stats(uplift: np.ndarray, path: str) -> None:
    logger.info(_SEP)
    logger.info("PREDICTIONS")
    logger.info(_SUB)
    logger.info("  File: %s", path)
    logger.info("  Rows: %s", f"{len(uplift):,}")
    print_score_stats(uplift, "Test uplift")
    logger.info(_SEP)


def print_full_report(
    *,
    info: DatasetInfo,
    n_features: int,
    cfg: PipelineConfig,
    report: EvaluationReport,
    oof_metrics: dict[str, float],
    test_uplift: np.ndarray | None = None,
    predictions_path: str | None = None,
) -> None:
    print_dataset_info(info, n_features)
    print_models_comparison(report, cfg)
    print_leaderboard_proxy(report, cfg)
    print_metrics(f"OOF metrics for {report.best_model_name}", oof_metrics, cfg)
    print_score_stats(report.oof_uplift, "OOF uplift")
    if test_uplift is not None and predictions_path:
        print_submission_stats(test_uplift, predictions_path)
