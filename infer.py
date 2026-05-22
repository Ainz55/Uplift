#!/usr/bin/env python3
"""
Инференс Hurdle TARNet с уже обученной моделью.
Принимает путь к тестовому parquet-файлу (по умолчанию /input/test.parquet).
Сохраняет predictions.csv в /app/output.
"""

import argparse
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("infer")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path("/app/model")
OUTPUT_DIR = Path("/app/output")
DEFAULT_TEST_FILE = "/input/test.parquet"

# Импорт класса модели (должен быть в neural_uplift.py, скопированном в образ)
from neural_uplift import HurdleTARNet


def load_ensemble():
    """Загружает все чекпоинты и пайплайн."""
    ckpt_files = sorted(MODEL_DIR.glob("hurdle_seed*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"Не найдены чекпоинты в {MODEL_DIR}")

    # Пайплайн и параметры из первого чекпоинта
    first = torch.load(ckpt_files[0], map_location="cpu")
    pipeline = first["pipeline"]
    feature_cols = first["feature_cols"]
    hidden_dim = first.get("hidden_dim", 512)
    dropout = first.get("dropout", 0.15)

    # Загружаем все модели в eval режиме
    models = []
    for path in ckpt_files:
        ckpt = torch.load(path, map_location="cpu")
        model = HurdleTARNet(len(feature_cols), hidden_dim=hidden_dim, dropout=dropout)
        model.load_state_dict(ckpt["state_dict"])
        model.to(DEVICE)
        model.eval()
        models.append(model)

    return pipeline, feature_cols, models


def main():
    parser = argparse.ArgumentParser(
        description="Предсказание uplift для тестового файла"
    )
    parser.add_argument(
        "test_file",
        nargs="?",
        default=DEFAULT_TEST_FILE,
        help=f"Путь к parquet-файлу с тестовыми признаками (по умолчанию {DEFAULT_TEST_FILE})",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "predictions.csv"),
        help="Путь для сохранения предсказаний (по умолчанию /app/output/predictions.csv)",
    )
    args = parser.parse_args()

    logger.info("Загрузка модели и пайплайна из %s", MODEL_DIR)
    pipeline, feature_cols, models = load_ensemble()

    logger.info("Чтение тестового файла %s", args.test_file)
    test = pd.read_parquet(args.test_file)
    if "user_id" not in test.columns:
        raise ValueError("Тестовый файл должен содержать колонку user_id")

    # На всякий случай убираем нецелевые колонки, если они вдруг есть
    for col in ["treatment_flg", "rec_spend"]:
        if col in test.columns:
            test = test.drop(columns=[col])

    # Применяем обученный пайплайн
    test_proc = pipeline.transform(test)

    # Финальные признаки
    X_test = test_proc[feature_cols].values.astype(np.float32)
    X_test_t = torch.tensor(X_test).to(DEVICE)

    # Усреднённый uplift по всем seed'ам
    uplifts = []
    for model in models:
        with torch.no_grad():
            uplift = model.predict_uplift(X_test_t)
        uplifts.append(uplift)
    avg_uplift = np.mean(uplifts, axis=0)

    # Сохраняем результат
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "user_id": test["user_id"].values,
        "UPLIFT_SCORE": avg_uplift,
    })
    out_df.to_csv(args.output, index=False)
    logger.info("Предсказания сохранены в %s", args.output)


if __name__ == "__main__":
    main()