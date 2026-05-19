"""Форматированный вывод метрик в консоль."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import PipelineConfig
from data import DatasetInfo
from evaluation import EvaluationReport
from metrics import campaign_profitability

logger = logging.getLogger(__name__)

_SEP = "=" * 72
_SUB = "-" * 72


def print_dataset_info(info: DatasetInfo, n_features: int) -> None:
    logger.info(_SEP)
    logger.info("ДАТАСЕТ")
    logger.info(_SUB)
    logger.info("  Train:              %s строк", f"{info.n_train:,}")
    logger.info("  Test:               %s строк", f"{info.n_test:,}")
    logger.info("  Clients:            %s строк", f"{info.n_clients:,}")
    logger.info("  Признаков модели:   %d", n_features)
    logger.info("  Покрытие clients:   train %.1f%% | test %.1f%%",
                info.train_clients_in_clients * 100, info.test_clients_in_clients * 100)
    logger.info(_SUB)
    logger.info("  treatment_flg:      %.2f%% в группе treatment", info.treatment_rate * 100)
    logger.info("  target (overall):   %.2f%%", info.target_rate * 100)
    logger.info("  Конверсия control:  %.2f%%", info.conv_control * 100)
    logger.info("  Конверсия treatment:%.2f%%", info.conv_treatment * 100)
    logger.info("  Средний uplift:     %.2f п.п.", info.avg_uplift * 100)


def print_business_params(cfg: PipelineConfig) -> None:
    logger.info(_SUB)
    logger.info("  Параметры кампании: margin=%.0f руб., cost=%.0f руб., top_k=%.0f%%",
                cfg.margin, cfg.treatment_cost, cfg.top_k * 100)
    logger.info("  CV: %d folds | early_stop=%.0f%% train (stratify=treatment+target)",
                cfg.n_folds, cfg.early_stopping_val_fraction * 100)


def print_models_comparison(report: EvaluationReport, cfg: PipelineConfig) -> None:
    logger.info(_SEP)
    logger.info("СРАВНЕНИЕ МОДЕЛЕЙ (OOF-метрики на всех train-строках)")
    logger.info(_SUB)
    logger.info(
        "  %-22s %8s %8s %10s %8s %12s",
        "Модель", "AUUC", "Qini", "Uplift@30%", "IRR@30%", "ROI@30%",
    )
    logger.info("  " + "-" * 68)

    k30 = int(cfg.top_k * 100)
    for name, cv in report.all_results.items():
        s = cv.summary
        uplift_k = s.get(f"uplift_at_{k30}pct", float("nan"))
        irr_k = s.get(f"irr_at_{k30}pct", float("nan"))
        roi = s.get("roi", float("nan"))
        logger.info(
            "  %-22s %8.4f %8.4f %9.4f %7.2fx %11.1f%%",
            name, s["auuc"], s["qini"], uplift_k, irr_k, roi * 100,
        )

    if report.ensemble_weights:
        logger.info(_SUB)
        logger.info("  Веса ансамбля: %s", report.ensemble_weights)


def print_metrics_full(
    title: str,
    metrics: dict[str, float],
    cfg: PipelineConfig,
) -> None:
    logger.info(_SEP)
    logger.info(title.upper())
    logger.info(_SUB)
    logger.info("  Основные (нормализованные, 0 = случайный, 1 = идеал):")
    logger.info("    AUUC:              %.4f", metrics["auuc"])
    logger.info("    Qini:              %.4f", metrics["qini"])
    logger.info("    Средний uplift:    %.4f (%.2f п.п.)", metrics["avg_uplift"], metrics["avg_uplift"] * 100)

    logger.info(_SUB)
    logger.info("  Uplift@top-K и IRR:")
    logger.info("  %-8s %12s %10s", "K", "Uplift", "IRR")
    for k in cfg.uplift_k_grid:
        pct = int(k * 100)
        u = metrics.get(f"uplift_at_{pct}pct", float("nan"))
        irr = metrics.get(f"irr_at_{pct}pct", float("nan"))
        mark = " <--" if abs(k - cfg.top_k) < 1e-9 else ""
        logger.info("  %-7s%% %12.4f %9.2fx%s", pct, u, irr, mark)

    k_pct = int(cfg.top_k * 100)
    if "net_profit" in metrics:
        logger.info(_SUB)
        logger.info("  Сводка @top-%d%% (основной K):", k_pct)
        logger.info("    Чистая прибыль:    %s руб.", f"{metrics['net_profit']:,.0f}")
        logger.info("    ROI:               %.1f%%", metrics["roi"] * 100)


def print_profitability_grid(
    y: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    cfg: PipelineConfig,
) -> None:
    logger.info(_SUB)
    logger.info("  Рентабельность по доле охвата (margin=%.0f, cost=%.0f):",
                cfg.margin, cfg.treatment_cost)
    logger.info(
        "  %-6s %10s %14s %14s %10s",
        "K", "Uplift", "Прибыль", "Затраты", "ROI",
    )
    for k in cfg.uplift_k_grid:
        p = campaign_profitability(
            y, uplift, treatment,
            k=k, margin_per_conversion=cfg.margin, treatment_cost=cfg.treatment_cost,
        )
        u = p["incremental_conversion_rate"]
        logger.info(
            "  %-5s%% %9.4f %13s %13s %9.1f%%",
            int(k * 100),
            u,
            f"{p['net_profit']:,.0f}",
            f"{p['campaign_cost']:,.0f}",
            p["roi"] * 100,
        )


def print_oof_score_stats(uplift: np.ndarray) -> None:
    logger.info(_SUB)
    logger.info("  Распределение OOF-предсказаний uplift:")
    logger.info("    min:    %8.4f", uplift.min())
    logger.info("    q25:    %8.4f", np.quantile(uplift, 0.25))
    logger.info("    median: %8.4f", np.median(uplift))
    logger.info("    mean:   %8.4f", uplift.mean())
    logger.info("    q75:    %8.4f", np.quantile(uplift, 0.75))
    logger.info("    max:    %8.4f", uplift.max())
    logger.info("    std:    %8.4f", uplift.std())
    logger.info("    доля uplift > 0: %.1f%%", (uplift > 0).mean() * 100)


def print_cv_folds(report: EvaluationReport, model_name: str) -> None:
    if model_name not in report.all_results or model_name == "ensemble":
        return
    fold_df = report.all_results[model_name].fold_metrics
    if "fold" not in fold_df.columns or len(fold_df) <= 1:
        return

    logger.info(_SEP)
    logger.info("CV ПО ФОЛДАМ: %s", model_name)
    logger.info(_SUB)
    cols = ["fold", "auuc", "qini"]
    k30 = "uplift_at_30pct"
    if k30 in fold_df.columns:
        cols.append(k30)
    irr30 = "irr_at_30pct"
    if irr30 in fold_df.columns:
        cols.append(irr30)
    for _, row in fold_df[cols].iterrows():
        parts = [f"  Fold {int(row['fold']):d}:"]
        parts.append(f" AUUC={row['auuc']:.4f}")
        parts.append(f" Qini={row['qini']:.4f}")
        if k30 in cols:
            parts.append(f" Uplift@30%={row[k30]:.4f}")
        if irr30 in cols:
            parts.append(f" IRR={row[irr30]:.2f}x")
        logger.info("".join(parts))

    logger.info(_SUB)
    logger.info(
        "  Среднее +/- std:  AUUC=%.4f +/- %.4f | Qini=%.4f +/- %.4f",
        fold_df["auuc"].mean(), fold_df["auuc"].std(),
        fold_df["qini"].mean(), fold_df["qini"].std(),
    )


def print_submission_stats(uplift: np.ndarray, path: str) -> None:
    logger.info(_SEP)
    logger.info("SUBMISSION")
    logger.info(_SUB)
    logger.info("  Файл:     %s", path)
    logger.info("  Строк:    %s", f"{len(uplift):,}")
    logger.info("  min:      %.4f", uplift.min())
    logger.info("  max:      %.4f", uplift.max())
    logger.info("  mean:     %.4f", uplift.mean())
    logger.info("  median:   %.4f", np.median(uplift))
    logger.info(_SEP)


def print_full_report(
    *,
    info: DatasetInfo,
    n_features: int,
    cfg: PipelineConfig,
    report: EvaluationReport,
    oof_metrics: dict[str, float],
    test_uplift: np.ndarray | None = None,
    submission_path: str | None = None,
) -> None:
    """Единый блок отчёта в консоль."""
    print_dataset_info(info, n_features)
    print_business_params(cfg)
    print_models_comparison(report, cfg)
    print_metrics_full(
        f"OOF-метрики лучшей модели ({report.best_model_name})",
        oof_metrics,
        cfg,
    )
    print_profitability_grid(report.y, report.oof_uplift, report.treatment, cfg)
    print_oof_score_stats(report.oof_uplift)
    print_cv_folds(report, report.best_model_name)
    if test_uplift is not None and submission_path:
        print_submission_stats(test_uplift, submission_path)
