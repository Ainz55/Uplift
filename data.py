"""Загрузка и валидация датасета."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TRAIN_FILE = "uplift_train.csv"
TEST_FILE = "uplift_test.csv"
CLIENTS_FILE = "clients.csv"

TRAIN_REQUIRED = {"client_id", "treatment_flg", "target"}
TEST_REQUIRED = {"client_id"}
CLIENTS_REQUIRED = {"client_id", "first_issue_date", "first_redeem_date", "age", "gender"}


@dataclass
class DatasetInfo:
    n_train: int
    n_test: int
    n_clients: int
    treatment_rate: float
    target_rate: float
    conv_control: float
    conv_treatment: float
    avg_uplift: float
    train_clients_in_clients: float
    test_clients_in_clients: float
    duplicate_client_ids_train: int
    duplicate_client_ids_test: int


class DatasetError(ValueError):
    """Ошибка структуры или содержимого датасета."""


def _require_columns(df: pd.DataFrame, required: set[str], context: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise DatasetError(
            f"{context}: отсутствуют колонки {sorted(missing)}. "
            f"Найдены: {list(df.columns)}"
        )


def _check_binary(series: pd.Series, name: str) -> None:
    vals = set(pd.unique(series.dropna()))
    if not vals <= {0, 1}:
        raise DatasetError(f"Колонка '{name}' должна содержать только 0 и 1, получено: {vals}")


def validate_dataset_dir(dataset_dir: Path) -> None:
    """Проверяет наличие файлов и схему до запуска пайплайна."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise DatasetError(f"Папка датасета не найдена: {dataset_dir}")

    for fname in (TRAIN_FILE, TEST_FILE, CLIENTS_FILE):
        path = dataset_dir / fname
        if not path.exists():
            raise DatasetError(
                f"Не найден файл {fname} в {dataset_dir}. "
                f"Ожидаемая структура: {TRAIN_FILE}, {TEST_FILE}, {CLIENTS_FILE}"
            )


def load_and_validate(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DatasetInfo]:
    validate_dataset_dir(dataset_dir)
    dataset_dir = Path(dataset_dir)

    train = pd.read_csv(dataset_dir / TRAIN_FILE)
    test = pd.read_csv(dataset_dir / TEST_FILE)
    clients = pd.read_csv(dataset_dir / CLIENTS_FILE)

    _require_columns(train, TRAIN_REQUIRED, TRAIN_FILE)
    _require_columns(test, TEST_REQUIRED, TEST_FILE)
    _require_columns(clients, CLIENTS_REQUIRED, CLIENTS_FILE)

    _check_binary(train["treatment_flg"], "treatment_flg")
    _check_binary(train["target"], "target")

    dup_train = int(train["client_id"].duplicated().sum())
    dup_test = int(test["client_id"].duplicated().sum())
    if dup_train:
        logger.warning("%s: %d дубликатов client_id", TRAIN_FILE, dup_train)
    if dup_test:
        logger.warning("%s: %d дубликатов client_id", TEST_FILE, dup_test)

    train_cov = train["client_id"].isin(clients["client_id"]).mean()
    test_cov = test["client_id"].isin(clients["client_id"]).mean()
    if train_cov < 1.0:
        logger.warning(
            "Не все client_id из train есть в clients.csv (покрытие %.1f%%)",
            train_cov * 100,
        )
    if test_cov < 1.0:
        logger.warning(
            "Не все client_id из test есть в clients.csv (покрытие %.1f%%)",
            test_cov * 100,
        )

    t_rate = float(train["treatment_flg"].mean())
    if not 0.05 < t_rate < 0.95:
        logger.warning(
            "Сильный дисбаланс treatment_flg (%.1f%%). "
            "Рекомендуется рандомизированный эксперимент ~50/50.",
            t_rate * 100,
        )

    conv_c = float(train.loc[train["treatment_flg"] == 0, "target"].mean())
    conv_t = float(train.loc[train["treatment_flg"] == 1, "target"].mean())

    info = DatasetInfo(
        n_train=len(train),
        n_test=len(test),
        n_clients=len(clients),
        treatment_rate=t_rate,
        target_rate=float(train["target"].mean()),
        conv_control=conv_c,
        conv_treatment=conv_t,
        avg_uplift=conv_t - conv_c,
        train_clients_in_clients=train_cov,
        test_clients_in_clients=test_cov,
        duplicate_client_ids_train=dup_train,
        duplicate_client_ids_test=dup_test,
    )
    return train, test, clients, info
