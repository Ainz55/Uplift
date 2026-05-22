#!/usr/bin/env python3
"""
Hurdle TARNet для uplift моделирования rec_spend.

Предобработка:
  1. Удаление высококоррелированных признаков (жадный отбор на train)
  2. Заполнение пропусков медианой + индикаторы пропусков
  3. One-hot кодирование низкокардинальных числовых признаков
  4. StandardScaler

Модель:
  HurdleTARNet – общий энкодер, отдельные головы для вероятности покупки (логиты)
  и суммы (log1p). Uplift = P(purchase|T=1)*E[spend|T=1] - P(purchase|T=0)*E[spend|T=0]
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data import load_and_validate

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("hurdle_tarnet")

RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BATCH_SIZE = 4096 if DEVICE == "cuda" else 2048

# =============================================================================
# Предобработка
# =============================================================================

class FeaturePipeline:
    def __init__(self, corr_threshold=0.7, missing_threshold=0.4,
                 drop_extreme_missing=0.8, onehot_max_unique=10,
                 ignore_cols=None):
        self.corr_threshold = corr_threshold
        self.missing_threshold = missing_threshold
        self.drop_extreme_missing = drop_extreme_missing
        self.onehot_max_unique = onehot_max_unique
        self.ignore_cols = ignore_cols or ["user_id", "treatment_flg", "rec_spend"]

        # Параметры, вычисляемые при fit
        self.cols_to_drop_corr_ = []
        self.medians_ = {}
        self.indicator_cols_ = []
        self.onehot_encoder_ = None
        self.nonnum_cols_ = []              # нечисловые столбцы для pd.get_dummies
        self.final_feature_cols_ = []
        self.scaler_ = StandardScaler()

    def fit(self, df):
        df = df.copy()

        # 1. Удаление признаков с экстремальной долей пропусков
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_ratio = df[numeric_cols].isnull().mean()
        extreme_cols = missing_ratio[missing_ratio > self.drop_extreme_missing].index.tolist()
        extreme_cols = [c for c in extreme_cols if c not in self.ignore_cols]
        df.drop(columns=extreme_cols, inplace=True, errors="ignore")
        logger.info("Удалено %d столбцов с пропусками > %.1f", len(extreme_cols), self.drop_extreme_missing)

        # 2. Удаление высококоррелированных (жадный алгоритм)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_for_corr = [c for c in numeric_cols if c not in self.ignore_cols]
        if cols_for_corr:
            corr_matrix = df[cols_for_corr].corr()
            high_corr_pairs = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            ).unstack().dropna()
            high_corr_pairs = high_corr_pairs[high_corr_pairs.abs() > self.corr_threshold]
            high_corr_pairs = high_corr_pairs.abs().sort_values(ascending=False)

            self.cols_to_drop_corr_ = []
            for (col1, col2) in high_corr_pairs.index:
                if col1 not in self.cols_to_drop_corr_ and col2 not in self.cols_to_drop_corr_:
                    self.cols_to_drop_corr_.append(col2)
            df.drop(columns=self.cols_to_drop_corr_, inplace=True, errors="ignore")
            logger.info("Жадно удалено %d высококоррелированных признаков", len(self.cols_to_drop_corr_))

        # 3. Пропуски: медиана + индикаторы
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.medians_ = {}
        self.indicator_cols_ = []
        missing_ratio = df[numeric_cols].isnull().mean()

        for col in numeric_cols:
            if col in self.ignore_cols:
                continue
            self.medians_[col] = df[col].median()
            if missing_ratio.get(col, 0) > self.missing_threshold:
                self.indicator_cols_.append(col)
                df[col + "_was_missing"] = df[col].isnull().astype(int)
            df[col].fillna(self.medians_[col], inplace=True)
        logger.info("Индикаторы пропусков добавлены для %d признаков", len(self.indicator_cols_))

        # 4. One-hot для низкокардинальных числовых признаков
        self.onehot_encoder_ = LowCardinalityOneHotEncoder(
            max_unique=self.onehot_max_unique,
            ignore_cols=self.ignore_cols,
        )
        df = self.onehot_encoder_.fit_transform(df)

        # 5. Обработка оставшихся нечисловых столбцов (категориальные, строки)
        self.nonnum_cols_ = [c for c in df.select_dtypes(exclude=[np.number]).columns 
                             if c not in self.ignore_cols]
        if self.nonnum_cols_:
            df = pd.get_dummies(df, columns=self.nonnum_cols_)
            logger.info("Применён one-hot к нечисловым столбцам: %s", self.nonnum_cols_)

        # 6. Фиксируем итоговый список признаков (без ignore_cols)
        self.final_feature_cols_ = [c for c in df.columns if c not in self.ignore_cols]

        # 7. Приводим все признаки к float32 (теперь всё обязано быть числовым)
        for c in self.final_feature_cols_:
            if not pd.api.types.is_numeric_dtype(df[c]):
                # Крайний случай: label encode, если что-то осталось строкой
                df[c] = df[c].astype('category').cat.codes.astype(np.float32)
            else:
                df[c] = df[c].astype(np.float32)

        # 8. Scaler на финальных признаках
        self.scaler_.fit(df[self.final_feature_cols_].values)
        logger.info("Scaler обучен на %d признаках", len(self.final_feature_cols_))

        return self

    # def transform(self, df):
    #     df = df.copy()

    #     # Удаляем столбцы, удалённые при fit
    #     df.drop(columns=[c for c in self.cols_to_drop_corr_ if c in df.columns], errors="ignore", inplace=True)

    #     # Заполнение пропусков + индикаторы
    #     for col, med in self.medians_.items():
    #         if col in df.columns:
    #             if col in self.indicator_cols_:
    #                 df[col + "_was_missing"] = df[col].isnull().astype(int)
    #             df[col].fillna(med, inplace=True)
    #         elif col in self.indicator_cols_:
    #             df[col + "_was_missing"] = 0

    #     # One-hot числовых низкокардинальных
    #     df = self.onehot_encoder_.transform(df)

    #     # One-hot нечисловых столбцов (как на трейне)
    #     for col in self.nonnum_cols_:
    #         if col in df.columns:
    #             dummies = pd.get_dummies(df[col], prefix=col)
    #             df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    #     # Выравнивание под final_feature_cols_
    #     for col in self.final_feature_cols_:
    #         if col not in df.columns:
    #             df[col] = 0.0

    #     keep = self.final_feature_cols_ + [c for c in self.ignore_cols if c in df.columns]
    #     df = df[keep]

    #     # Приведение к float32 и scaler
    #     X = df[self.final_feature_cols_].values.astype(np.float32)
    #     X_scaled = self.scaler_.transform(X)
    #     df[self.final_feature_cols_] = X_scaled

    #     return df
    

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет параметры к новому DataFrame, возвращает копию с признаками и ignore_cols."""
        df = df.copy()

        # Удаляем колонки, удалённые при fit
        df.drop(columns=[c for c in self.cols_to_drop_corr_ if c in df.columns], errors="ignore", inplace=True)

        # Заполнение и индикаторы
        for col, med in self.medians_.items():
            if col in df.columns:
                if col in self.indicator_cols_:
                    df[col + "_was_missing"] = df[col].isnull().astype(int)
                df[col].fillna(med, inplace=True)
            elif col in self.indicator_cols_:
                df[col + "_was_missing"] = 0

        # One-hot
        df = self.onehot_encoder_.transform(df)

        # Выравнивание колонок под final_feature_cols_
        for col in self.final_feature_cols_:
            if col not in df.columns:
                df[col] = 0.0

        # Оставляем только нужные колонки (признаки + при необходимости ignore_cols)
        keep = self.final_feature_cols_ + [c for c in self.ignore_cols if c in df.columns]
        df = df[keep]

        # Приводим признаки к float32 и применяем scaler
        X = df[self.final_feature_cols_].values.astype(np.float32)
        X_scaled = self.scaler_.transform(X)
        df[self.final_feature_cols_] = X_scaled

        return df
    

class LowCardinalityOneHotEncoder:
    """One-hot для float/int с малым числом уникальных."""

    def __init__(self, max_unique: int = 10, ignore_cols: Optional[List[str]] = None):
        self.max_unique = max_unique
        self.ignore_cols = ignore_cols or []
        self.low_card_cols_ = []
        self.dummy_names_: Dict[str, List[str]] = {}

    def fit(self, df: pd.DataFrame) -> LowCardinalityOneHotEncoder:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col in self.ignore_cols:
                continue
            if df[col].nunique(dropna=False) <= self.max_unique:
                self.low_card_cols_.append(col)
                unique_vals = sorted(df[col].dropna().unique())
                self.dummy_names_[col] = [f"{col}_{v}" for v in unique_vals]
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.low_card_cols_:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
                # Добавляем отсутствующие колонки из fit
                for expected_col in self.dummy_names_[col]:
                    if expected_col not in dummies.columns:
                        dummies[expected_col] = 0
                dummies = dummies[self.dummy_names_[col]]
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            else:
                # Если столбца нет, создаём нулевые дамми
                for expected_col in self.dummy_names_[col]:
                    df[expected_col] = 0
        return df


# =============================================================================
# Нейросетевая модель: Hurdle TARNet
# =============================================================================

class HurdleTARNet(nn.Module):
    """TARNet с hurdle-лоссом: отдельные головы для вероятности и суммы."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
        )
        h = hidden_dim // 2

        # Головы вероятности (логиты)
        self.logit_t = nn.Sequential(nn.Linear(h, h // 2), nn.ELU(), nn.Linear(h // 2, 1))
        self.logit_c = nn.Sequential(nn.Linear(h, h // 2), nn.ELU(), nn.Linear(h // 2, 1))

        # Головы суммы (log1p)
        self.amount_t = nn.Sequential(nn.Linear(h, h // 2), nn.ELU(), nn.Linear(h // 2, 1))
        self.amount_c = nn.Sequential(nn.Linear(h, h // 2), nn.ELU(), nn.Linear(h // 2, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.encoder(x)
        return (self.logit_t(z).squeeze(-1),
                self.logit_c(z).squeeze(-1),
                self.amount_t(z).squeeze(-1),
                self.amount_c(z).squeeze(-1))

    @torch.no_grad()
    def predict_uplift(self, x):
        self.eval()
        logit_t, logit_c, amt_t, amt_c = self.forward(x)
        prob_t = torch.sigmoid(logit_t)
        prob_c = torch.sigmoid(logit_c)
        # E[spend] = P(purchase) * E[spend | purchase] ≈ prob * expm1(amount)
        spend_t = prob_t * torch.expm1(amt_t.clamp(-20, 20))
        spend_c = prob_c * torch.expm1(amt_c.clamp(-20, 20))
        return (spend_t - spend_c).cpu().numpy()


def hurdle_loss(pred, y_true, t_true, alpha=1.0, beta=0.5):
    """
    pred: (logit_t, logit_c, amt_t, amt_c)
    y_true: rec_spend (не логарифмированное)
    t_true: treatment flag (int)
    """
    logit_t, logit_c, amt_t, amt_c = pred
    y_bin = (y_true > 0).float()

    # Бинарная кросс-энтропия
    bce_t = F.binary_cross_entropy_with_logits(logit_t, y_bin, reduction='none')
    bce_c = F.binary_cross_entropy_with_logits(logit_c, y_bin, reduction='none')
    loss_prob = (bce_t[t_true == 1].mean() + bce_c[t_true == 0].mean()) / 2.0

    # MSE для log1p(spend) только для покупателей
    mask_t_pos = (t_true == 1) & (y_true > 0)
    mask_c_pos = (t_true == 0) & (y_true > 0)
    loss_amount = 0.0
    if mask_t_pos.any():
        loss_amount += F.mse_loss(amt_t[mask_t_pos], torch.log1p(y_true[mask_t_pos]))
    if mask_c_pos.any():
        loss_amount += F.mse_loss(amt_c[mask_c_pos], torch.log1p(y_true[mask_c_pos]))

    return beta * loss_prob + alpha * loss_amount


# =============================================================================
# Обучение
# =============================================================================

def train_hurdle_tarnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    t_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    t_val: np.ndarray,
    *,
    hidden_dim: int = 512,
    dropout: float = 0.15,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    epochs: int = 120,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = 25,
    alpha: float = 1.0,
    beta: float = 0.5,
    seed: int = RANDOM_STATE,
    seed_idx: int = 1,
    total_seeds: int = 1,
) -> HurdleTARNet:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr_s = X_train.astype(np.float32)
    X_val_s = X_val.astype(np.float32)
    y_tr = y_train.astype(np.float32)
    y_v = y_val.astype(np.float32)
    t_tr = t_train.astype(np.int8)
    t_v = t_val.astype(np.int8)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_s), torch.tensor(y_tr), torch.tensor(t_tr)),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )

    model = HurdleTARNet(X_tr_s.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(ep: int) -> float:
        warmup = 5
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    X_val_t = torch.tensor(X_val_s).to(DEVICE)
    y_val_t = torch.tensor(y_v).to(DEVICE)
    t_val_t = torch.tensor(t_v).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    best_uplift = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_b, y_b, t_b in loader:
            X_b, y_b, t_b = X_b.to(DEVICE), y_b.to(DEVICE), t_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = hurdle_loss(pred, y_b, t_b, alpha=alpha, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            val_loss = hurdle_loss(pred_val, y_val_t, t_val_t, alpha=alpha, beta=beta).item()
            val_uplift = model.predict_uplift(X_val_t)
            u10 = uplift_at_k(y_val, val_uplift, t_val)

        if epoch % 5 == 0 or epoch <= 10:
            logger.info(
                "[seed %d/%d] Epoch %3d | train=%.4f | val_loss=%.4f | uplift@10=%.3f",
                seed_idx, total_seeds, epoch, np.mean(train_losses), val_loss, u10,
            )
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_uplift = u10
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state:
        model.load_state_dict(best_state)
    logger.info("[seed %d/%d] Best val_loss=%.4f uplift@10=%.3f", seed_idx, total_seeds, best_val_loss, best_uplift)
    return model


def uplift_at_k(y: np.ndarray, scores: np.ndarray, t: np.ndarray, k: float = 0.10) -> float:
    n_top = max(1, int(np.ceil(len(scores) * k)))
    order = np.argsort(scores)[::-1]
    y_top = y[order[:n_top]]
    t_top = t[order[:n_top]]
    mask_t = t_top == 1
    mask_c = t_top == 0
    if mask_t.sum() == 0 or mask_c.sum() == 0:
        return 0.0
    return float(y_top[mask_t].mean() - y_top[mask_c].mean())


# =============================================================================
# Главные функции
# =============================================================================

def run_train(cfg: argparse.Namespace) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(cfg.dataset)

    logger.info("Загрузка данных")
    train, test, _ = load_and_validate(dataset_dir)

    # Предобработка
    logger.info("Обучение FeaturePipeline")
    pipeline = FeaturePipeline(
        corr_threshold=cfg.corr_threshold,
        missing_threshold=cfg.missing_threshold,
        drop_extreme_missing=cfg.drop_extreme_missing,
        onehot_max_unique=cfg.onehot_max_unique,
    )
    pipeline.fit(train)

    # Применяем к train и test
    train_proc = pipeline.transform(train)
    test_proc = pipeline.transform(test)

    feature_cols = pipeline.final_feature_cols_
    X = train_proc[feature_cols].values
    y = train_proc["rec_spend"].values.astype(np.float32)
    t = train_proc["treatment_flg"].values.astype(np.int8)

    # Разбиение train/val
    rng = np.random.default_rng(RANDOM_STATE)
    strata = t * 2 + (y > 0).astype(np.int8)
    val_mask = np.zeros(len(y), dtype=bool)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        n_val = max(1, int(len(idx) * 0.15))
        val_mask[rng.choice(idx, size=n_val, replace=False)] = True

    X_tr, X_val = X[~val_mask], X[val_mask]
    y_tr, y_val = y[~val_mask], y[val_mask]
    t_tr, t_val = t[~val_mask], t[val_mask]

    logger.info("Train: %d  Val: %d  Features: %d", len(X_tr), len(X_val), X_tr.shape[1])

    # Обучение ансамбля
    seeds = [RANDOM_STATE + i * 100 for i in range(cfg.n_seeds)]
    all_test_uplifts = []
    for i, seed in enumerate(seeds, start=1):
        logger.info("=== Seed %d/%d (seed=%d) ===", i, cfg.n_seeds, seed)
        model = train_hurdle_tarnet(
            X_tr, y_tr, t_tr, X_val, y_val, t_val,
            hidden_dim=cfg.hidden_dim, dropout=cfg.dropout,
            lr=cfg.lr, weight_decay=cfg.weight_decay,
            epochs=cfg.epochs, batch_size=cfg.batch_size,
            patience=cfg.patience,
            alpha=cfg.alpha, beta=cfg.beta,
            seed=seed, seed_idx=i, total_seeds=cfg.n_seeds,
        )
        # Сохраняем чекпоинт
        ckpt_path = output_dir / f"hurdle_seed{seed}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "hidden_dim": cfg.hidden_dim,
            "dropout": cfg.dropout,
            "pipeline": pipeline,
        }, ckpt_path)

        # Предсказания на тесте
        X_test = test_proc[feature_cols].values.astype(np.float32)
        X_test_t = torch.tensor(X_test).to(DEVICE)
        test_up = model.predict_uplift(X_test_t)
        all_test_uplifts.append(test_up)

    avg_test_uplift = np.mean(all_test_uplifts, axis=0)
    pred_path = output_dir / "predictions_neural.csv"
    pd.DataFrame({
        "user_id": test_proc["user_id"].values,
        "UPLIFT_SCORE": avg_test_uplift,
    }).to_csv(pred_path, index=False)
    logger.info("Финальные предсказания сохранены: %s", pred_path)


def run_infer(cfg: argparse.Namespace) -> None:
    # (для инференса аналогично загружаем пайплайн из чекпоинта и применяем к новым данным)
    raise NotImplementedError("Используйте чекпоинт с сохранённым pipeline для инференса.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "ensemble"], default="train")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output-dir", default="output/hurdle_v1")
    # Параметры предобработки
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument("--missing-threshold", type=float, default=0.4)
    parser.add_argument("--drop-extreme-missing", type=float, default=0.8)
    parser.add_argument("--onehot-max-unique", type=int, default=10)
    # Параметры модели
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=1.0, help="Вес регрессионной части лосса")
    parser.add_argument("--beta", type=float, default=0.5, help="Вес классификационной части лосса")

    cfg = parser.parse_args()

    if cfg.mode == "train":
        run_train(cfg)
    else:
        raise ValueError(f"Mode {cfg.mode} not implemented yet")

if __name__ == "__main__":
    main()