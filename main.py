"""
Задача №2 · Uplift-моделирование для оптимизации маркетинговых кампаний.

Полный пайплайн:
      1. Загрузка и валидация данных (train, test, clients, products)
      2. Feature engineering - создание признаков из временных меток,
         демографии и (опционально) продуктовой истории
      3. Кросс-валидация нескольких uplift-подходов (T-Learner, S-Learner,
         Class Transformation, X-Learner, Causal Forest) и выбор лучшего подхода
      4. Расчёт бизнес-метрик: AUUC, Qini, IRR, ROI, uplift в топ-K%
      5. Обучение финальной модели на всех данных
      6. Генерация отчётов (JSON, CSV, графики) и submission-файла
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from config import PipelineConfig
from console_report import print_full_report
from data import DatasetError, load_and_validate
from evaluation import select_best_model, train_final_model
from features import prepare_datasets
from metrics import evaluate_all_metrics
from visualization import generate_all_plots

# Настройка кодировки stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("uplift")


def save_reports(
    cfg: PipelineConfig,
    eval_report,
    oof_metrics: dict[str, float],
    cv_tables: dict[str, pd.DataFrame],
) -> None:
    """
    Сохранение метрик в JSON и таблиц кросс-валидации в CSV.

    Структура JSON:
      - best_model: название лучшей модели
      - oof_metrics: основные метрики (AUUC, Qini, ROI, ...)
      - models_cv_summary: сводка по каждой модели (среднее/std метрик)
      - ensemble_weights: веса ансамбля (если лучшая модель — ансамбль)

    CSV-файлы создаются для каждой модели по отдельности (например, cv_tlearner.csv).
    """

    cfg.ensure_dirs()

    summary = {
        "best_model": eval_report.best_model_name,
        "oof_metrics": oof_metrics,
        "models_cv_summary": {
            name: {k: float(v) for k, v in cv.summary.items() if k != "weights"}
            for name, cv in eval_report.all_results.items()
        },
    }
    if eval_report.ensemble_weights:
        summary["ensemble_weights"] = eval_report.ensemble_weights

    with open(cfg.output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for name, fold_df in cv_tables.items():
        fold_df.to_csv(cfg.output_dir / f"cv_{name}.csv", index=False)

    logger.info("JSON/CSV отчёты сохранены: %s", cfg.output_dir)


def run(cfg: PipelineConfig) -> None:
    """
    Полный цикл uplift-моделирования, содержит:

    1. Загрузка и валидация CSV (train, test, clients)
    2. Feature engineering: временные, демографические, продуктовые признаки
    3. Кросс-валидация нескольких моделей, выбор лучшей
    4. Расчёт метрик (AUUC, Qini, IRR, ROI, топ-K uplift)
    5. Обучение финальной модели на всех данных
    6. Генерация отчётов, графиков и submission.csv
    """
    cfg.ensure_dirs()
    logger.info("Загрузка данных из %s", cfg.dataset_dir)

    # 1. Загрузка данных
    try:
        train, test, clients, info = load_and_validate(cfg.dataset_dir)
    except DatasetError as e:
        logger.error("Ошибка датасета: %s", e)
        raise SystemExit(1) from e

    # 2. feature engineering
    train_df, test_df, feature_cols = prepare_datasets(train, test, clients)

    X = train_df[feature_cols]
    y = train_df["target"].values.astype(np.int8)
    treatment = train_df["treatment_flg"].values.astype(np.int8)
    X_test = test_df[feature_cols]

    # 3. Кросс-валидация и выбор модели
    logger.info("Кросс-валидация — сравнение моделей...")
    eval_report = select_best_model(X, y, treatment, cfg)

    # 4. Расчёт метрик на oof-предсказаниях
    oof_metrics = evaluate_all_metrics(
        y, eval_report.oof_uplift, treatment,
        k=cfg.top_k, margin=cfg.margin, cost=cfg.treatment_cost,
        k_grid=cfg.uplift_k_grid,
    )

    # 5. Сохранение отчётов
    cv_tables = {
        n: cv.fold_metrics
        for n, cv in eval_report.all_results.items()
        if n not in {"ensemble", "rank_ensemble", "top_ensemble", "rank_top_ensemble"}
    }
    save_reports(cfg, eval_report, oof_metrics, cv_tables)

    # 6. Обучение финальной модели на всех данных
    logger.info("Обучение финальной модели: %s", eval_report.best_model_name)
    weights = (
        eval_report.ensemble_weights
        if eval_report.best_model_name in {"ensemble", "rank_ensemble", "top_ensemble", "rank_top_ensemble"}
        else None
    )
    final_model = train_final_model(
        eval_report.best_model_name, X, y, treatment, cfg, sub_weights=weights,
    )

    # 7. Извлечение важности признаков
    feature_importances = None
    if hasattr(final_model, "feature_importances_"):
        feature_importances = final_model.feature_importances_(feature_cols)
    elif hasattr(final_model, "models"):
        for sub in final_model.models:
            if hasattr(sub, "feature_importances_"):
                feature_importances = sub.feature_importances_(feature_cols)
                break

    # 8. Построение графиков
    logger.info("Построение графиков...")
    generate_all_plots(eval_report, cfg, feature_importances=feature_importances)

    # 9. Предсказание на тестовой выборке
    test_uplift = final_model.predict(X_test)
    submission = pd.DataFrame({"client_id": test_df["client_id"], "uplift": test_uplift})

    sample_path = cfg.dataset_dir / "uplift_sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        if "client_id" in sample.columns and set(sample["client_id"]) == set(submission["client_id"]):
            submission = sample[["client_id"]].merge(submission, on="client_id", how="left", validate="one_to_one")
        else:
            logger.warning("Sample submission найден, но набор client_id не совпадает; сохраняем порядок uplift_test.csv")

    submission.to_csv(cfg.submission_path, index=False)

    # 10. Формирование итогового отчёта в консоль
    print_full_report(
        info=info,
        n_features=len(feature_cols),
        cfg=cfg,
        report=eval_report,
        oof_metrics=oof_metrics,
        test_uplift=test_uplift,
        submission_path=str(cfg.submission_path),
    )

    logger.info("Графики: %s", cfg.reports_dir)
    logger.info("Готово.")


def main() -> None:
    """
    Парсит аргументы командной строки и запускает пайплайн.

    Пример запуска:
        python run.py --dataset ./data --folds 5 --margin 500 --cost 50

    Аргументы:
        --dataset      Папка с uplift_train.csv, uplift_test.csv, clients.csv
        --output-dir   Куда сохранить submission и JSON/CSV отчёты
        --reports-dir  Куда сохранить графики
        --folds        Количество фолдов для кросс-валидации (по умолчанию 5)
        --margin       Маржа с одной конверсии, руб. (для расчёта ROI)
        --cost         Стоимость одного воздействия, руб. (для расчёта ROI)
    """

    parser = argparse.ArgumentParser(
        description="Uplift-моделирование · задача №2 (полное решение)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Папка с uplift_train.csv, uplift_test.csv, clients.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--margin", type=float, default=None, help="Маржа с конверсии, руб.")
    parser.add_argument("--cost", type=float, default=None, help="Стоимость воздействия, руб.")
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.dataset:
        cfg.dataset_dir = args.dataset
    if args.output_dir:
        cfg.output_dir = args.output_dir
        cfg.submission_path = args.output_dir / "submission.csv"
    if args.reports_dir:
        cfg.reports_dir = args.reports_dir
    if args.folds is not None:
        cfg.n_folds = args.folds
    if args.margin is not None:
        cfg.margin = args.margin
    if args.cost is not None:
        cfg.treatment_cost = args.cost

    run(cfg)


if __name__ == "__main__":
    main()
