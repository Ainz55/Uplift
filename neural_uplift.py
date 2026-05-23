"""TARNet / DragonNet uplift neural network for rec_spend.

Train locally, save weights, use for inference or ensemble with LightGBM.

Usage:
  # Train and save model + predictions:
  python neural_uplift.py --mode train --output-dir output/neural

  # Inference only (load saved weights):
  python neural_uplift.py --mode infer --model-path output/neural/tarnet.pt --output-dir output/neural

  # Ensemble neural predictions with existing LightGBM predictions:
  python neural_uplift.py --mode ensemble \
      --lgbm-predictions output/semantic_candidate/predictions.csv \
      --neural-predictions output/neural/predictions_neural.csv \
      --output output/neural/predictions_ensemble.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from data import load_and_validate
from features import prepare_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("neural_uplift")

RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Автоматический batch size: больше на GPU (быстрее), меньше на CPU (меньше RAM)
DEFAULT_BATCH_SIZE = 8192 if DEVICE == "cuda" else 2048


def set_full_determinism(seed: int) -> None:
    """Lock all RNG sources so repeated runs give the same uplift@10."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class TARNet(nn.Module):
    """Treatment-Agnostic Representation Network.

    Shared encoder → two separate heads (treatment / control).
    Optionally adds a propensity head (DragonNet-style regularization).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        dragonnet: bool = True,
    ) -> None:
        super().__init__()
        self.dragonnet = dragonnet

        # Shared representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
        )

        head_in = hidden_dim // 2

        # Treatment outcome head
        self.head_t = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ELU(),
            nn.Linear(head_in // 2, 1),
        )
        # Control outcome head
        self.head_c = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ELU(),
            nn.Linear(head_in // 2, 1),
        )
        # Propensity head (DragonNet) — predicts P(T=1|X)
        if dragonnet:
            self.head_prop = nn.Sequential(
                nn.Linear(head_in, head_in // 4),
                nn.ELU(),
                nn.Linear(head_in // 4, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        z = self.encoder(x)
        y_t = self.head_t(z).squeeze(-1)
        y_c = self.head_c(z).squeeze(-1)
        prop = self.head_prop(z).squeeze(-1) if self.dragonnet else None
        return y_t, y_c, prop

    @torch.no_grad()
    def predict_uplift(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        y_t, y_c, _ = self.forward(x)
        # Convert from log1p scale back to original
        uplift = torch.expm1(y_t.clamp(-20, 20)) - torch.expm1(y_c.clamp(-20, 20))
        return uplift.cpu().numpy()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def tarnet_loss(
    y_t_pred: torch.Tensor,
    y_c_pred: torch.Tensor,
    prop_pred: torch.Tensor | None,
    y_true: torch.Tensor,
    treatment: torch.Tensor,
    *,
    prop_weight: float = 1.0,
    nonzero_weight: float = 5.0,
) -> torch.Tensor:
    """MSE loss on log1p(y), separate per group, with non-zero upweighting."""
    mask_t = treatment == 1
    mask_c = treatment == 0

    # Sample weights: upweight non-zero outcomes
    w = torch.ones_like(y_true)
    w[y_true > 0] = nonzero_weight

    loss = torch.tensor(0.0, device=y_true.device)

    if mask_t.any():
        err_t = (y_t_pred[mask_t] - y_true[mask_t]) ** 2 * w[mask_t]
        loss = loss + err_t.mean()

    if mask_c.any():
        err_c = (y_c_pred[mask_c] - y_true[mask_c]) ** 2 * w[mask_c]
        loss = loss + err_c.mean()

    if prop_pred is not None:
        # Propensity BCE loss (DragonNet regularization)
        prop_loss = nn.functional.binary_cross_entropy(
            prop_pred, treatment.float(), reduction="mean"
        )
        loss = loss + prop_weight * prop_loss

    return loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tarnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    t_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    t_val: np.ndarray,
    *,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    dragonnet: bool = False,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    epochs: int = 80,
    batch_size: int = 2048,
    patience: int = 10,
    nonzero_weight: float = 5.0,
    random_state: int = RANDOM_STATE,
) -> TARNet:
    set_full_determinism(random_state)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    y_log_train = np.log1p(np.clip(y_train, 0, None)).astype(np.float32)
    y_log_val = np.log1p(np.clip(y_val, 0, None)).astype(np.float32)

    # Move ALL training data to GPU once — no per-batch CPU→GPU transfer overhead.
    X_train_gpu = torch.from_numpy(X_train_s).to(DEVICE)
    y_train_gpu = torch.from_numpy(y_log_train).to(DEVICE)
    t_train_gpu = torch.from_numpy(t_train.astype(np.int8)).to(DEVICE)
    N = len(X_train_gpu)

    g = torch.Generator(device=DEVICE)
    g.manual_seed(random_state)

    model = TARNet(X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout, dragonnet=dragonnet)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_val_t = torch.tensor(X_val_s).to(DEVICE)
    y_val_t = torch.tensor(y_log_val).to(DEVICE)
    t_val_t = torch.tensor(t_val.astype(np.int8)).to(DEVICE)
    y_val_raw = np.asarray(y_val, dtype=np.float32)
    t_val_np = np.asarray(t_val, dtype=np.int8)

    best_val_loss = float("inf")
    best_uplift_at_best = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        # GPU-resident shuffling — saves ~80% of dataloader overhead.
        perm = torch.randperm(N, generator=g, device=DEVICE)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            X_b = X_train_gpu[idx]
            y_b = y_train_gpu[idx]
            t_b = t_train_gpu[idx]
            optimizer.zero_grad(set_to_none=True)
            y_t_p, y_c_p, prop_p = model(X_b)
            loss = tarnet_loss(y_t_p, y_c_p, prop_p, y_b, t_b, nonzero_weight=nonzero_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Early stop on val MSE (smooth, low-variance criterion).
        # uplift@10 is logged for monitoring but is too noisy for selection.
        model.eval()
        with torch.no_grad():
            y_t_v, y_c_v, prop_v = model(X_val_t)
            val_loss = tarnet_loss(
                y_t_v, y_c_v, prop_v, y_val_t, t_val_t, nonzero_weight=nonzero_weight
            ).item()
        val_uplift = model.predict_uplift(X_val_t)
        val_score = uplift_at_k(y_val_raw, val_uplift, t_val_np, k=0.10)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d | train=%.4f | val_loss=%.4f | val_uplift@10=%.4f | lr=%.2e",
                epoch, float(np.mean(train_losses)), val_loss, val_score, scheduler.get_last_lr()[0],
            )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_uplift_at_best = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    "Early stopping at epoch %d (best val_loss=%.4f, val_uplift@10=%.4f)",
                    epoch, best_val_loss, best_uplift_at_best,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(
            "Loaded best weights (val_loss=%.4f, val_uplift@10=%.4f)",
            best_val_loss, best_uplift_at_best,
        )

    model._scaler = scaler  # type: ignore[attr-defined]
    model._val_uplift = best_uplift_at_best  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# OOF uplift@10 evaluation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _make_val_mask(
    t: np.ndarray, y: np.ndarray, *, val_fraction: float, seed: int,
    strat_mode: str = "quartile",
) -> np.ndarray:
    """Build a stratified val mask.

    strat_mode="quartile" — stratify by treatment × y-quartile (positive-only).
    strat_mode="old"      — coarse stratification by treatment × (y>0). This
                            matches the original 22.23-baseline run exactly.
    """
    rng = np.random.default_rng(seed)
    if strat_mode == "old":
        strata = t.astype(np.int16) * 2 + (y > 0).astype(np.int8)
    elif strat_mode == "quartile":
        pos = y > 0
        if pos.any():
            bins = np.zeros(len(y), dtype=np.int8)
            quartiles = np.quantile(y[pos], [0.25, 0.5, 0.75])
            bins[pos] = 1 + np.digitize(y[pos], quartiles)
        else:
            bins = np.zeros(len(y), dtype=np.int8)
        strata = t.astype(np.int16) * 10 + bins
    else:
        raise ValueError(f"Unknown strat_mode={strat_mode!r}")
    val_mask = np.zeros(len(y), dtype=bool)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        n_val = max(1, int(round(len(idx) * val_fraction)))
        chosen = rng.choice(idx, size=n_val, replace=False)
        val_mask[chosen] = True
    return val_mask


def run_train(cfg: argparse.Namespace) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(cfg.dataset)

    logger.info("Loading data from %s", dataset_dir)
    train, test, _ = load_and_validate(dataset_dir)

    X, X_test, feature_cols, _ = prepare_datasets(
        train, test, max_categories=40, feature_set=cfg.feature_set
    )

    y = train["rec_spend"].to_numpy(dtype=np.float32)
    t = train["treatment_flg"].to_numpy(dtype=np.int8)

    if cfg.honest_val_from:
        # Build val mask on the REAL dataset (so it matches the V4 val exactly),
        # then map those user_ids onto the current (possibly augmented) train.
        logger.info("Using honest val from %s — no leakage with augmented data", cfg.honest_val_from)
        real_train, _, _ = load_and_validate(Path(cfg.honest_val_from))
        real_y = real_train["rec_spend"].to_numpy(dtype=np.float32)
        real_t = real_train["treatment_flg"].to_numpy(dtype=np.int8)
        real_val_mask = _make_val_mask(
            real_t, real_y, val_fraction=cfg.val_fraction, seed=RANDOM_STATE,
            strat_mode=cfg.strat_mode,
        )
        real_val_ids = set(real_train.loc[real_val_mask, "user_id"].astype(np.int64).tolist())
        val_mask = train["user_id"].astype(np.int64).isin(real_val_ids).to_numpy()
        logger.info("Mapped %d real val users → val_mask covers %d of current train",
                    len(real_val_ids), int(val_mask.sum()))
    else:
        val_mask = _make_val_mask(
            t, y, val_fraction=cfg.val_fraction, seed=RANDOM_STATE,
            strat_mode=cfg.strat_mode,
        )

    X_tr, X_val = X.values[~val_mask], X.values[val_mask]
    y_tr, y_val = y[~val_mask], y[val_mask]
    t_tr, t_val = t[~val_mask], t[val_mask]

    logger.info("Train: %d  Val: %d  Features: %d  Device: %s",
                len(X_tr), len(X_val), X_tr.shape[1], DEVICE)
    logger.info("Seeds: %d  Hidden: %d  Dropout: %.2f  DragonNet: %s",
                cfg.n_seeds, cfg.hidden_dim, cfg.dropout, not cfg.no_dragonnet)

    seeds = [RANDOM_STATE + 17 * i for i in range(cfg.n_seeds)]
    test_uplifts: list[np.ndarray] = []
    val_uplifts_avg: list[np.ndarray] = []
    val_scores: list[float] = []
    scalers: list[StandardScaler] = []
    state_dicts: list[dict] = []

    for i, seed in enumerate(seeds, 1):
        logger.info("=" * 60)
        logger.info("Training seed %d/%d (random_state=%d)", i, len(seeds), seed)
        model = train_tarnet(
            X_tr, y_tr, t_tr,
            X_val, y_val, t_val,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            dragonnet=not cfg.no_dragonnet,
            lr=cfg.lr,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            patience=cfg.patience,
            nonzero_weight=cfg.nonzero_weight,
            random_state=seed,
        )
        scaler = model._scaler  # type: ignore[attr-defined]
        scalers.append(scaler)
        state_dicts.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        X_val_s = torch.tensor(scaler.transform(X_val).astype(np.float32)).to(DEVICE)
        val_uplift = model.predict_uplift(X_val_s)
        val_uplifts_avg.append(val_uplift)
        score = uplift_at_k(y_val, val_uplift, t_val)
        val_scores.append(score)
        logger.info("Seed %d val uplift@10: %.4f", seed, score)

        X_test_s = torch.tensor(scaler.transform(X_test.values).astype(np.float32)).to(DEVICE)
        test_uplifts.append(model.predict_uplift(X_test_s))

    # Rank-average across seeds — stable for uplift@10.
    def _rank_pct(arr: np.ndarray) -> np.ndarray:
        return pd.Series(arr).rank(method="average", pct=True).to_numpy(dtype=np.float32)

    val_rank_mean = np.mean([_rank_pct(u) for u in val_uplifts_avg], axis=0)
    val_score_ens = uplift_at_k(y_val, val_rank_mean, t_val)
    logger.info("=" * 60)
    logger.info("Per-seed val uplift@10: %s",
                ", ".join(f"{s:.4f}" for s in val_scores))
    logger.info("Ensemble (rank-avg) val uplift@10: %.4f", val_score_ens)

    # Save last seed's model checkpoint for compatibility (used by --mode infer).
    model_path = output_dir / "tarnet.pt"
    torch.save(
        {
            "state_dict": state_dicts[-1],
            "scaler": scalers[-1],
            "feature_cols": feature_cols,
            "all_state_dicts": state_dicts,
            "all_scalers": scalers,
            "hidden_dim": cfg.hidden_dim,
            "dropout": cfg.dropout,
            "dragonnet": not cfg.no_dragonnet,
            "seeds": seeds,
            "feature_set": cfg.feature_set,
        },
        model_path,
    )
    logger.info("Model saved: %s", model_path)

    test_rank_mean = np.mean([_rank_pct(u) for u in test_uplifts], axis=0)

    pred_path = output_dir / "predictions_neural.csv"
    pd.DataFrame({"user_id": test["user_id"].to_numpy(), "UPLIFT_SCORE": test_rank_mean}).to_csv(
        pred_path, index=False, encoding="utf-8"
    )
    # Also save raw-scale (mean of expm1 uplifts) for ensemble compatibility.
    test_raw_mean = np.mean(test_uplifts, axis=0)
    pd.DataFrame({"user_id": test["user_id"].to_numpy(), "UPLIFT_SCORE": test_raw_mean}).to_csv(
        output_dir / "predictions_neural_raw.csv", index=False, encoding="utf-8"
    )
    logger.info("Neural predictions saved: %s (rank-avg, %d seeds)", pred_path, len(seeds))


def run_infer(cfg: argparse.Namespace) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(cfg.dataset)

    checkpoint = torch.load(cfg.model_path, map_location="cpu", weights_only=False)
    feature_cols: list[str] = checkpoint["feature_cols"]
    feature_set = checkpoint.get("feature_set", "semantic")
    hidden_dim = checkpoint.get("hidden_dim", 256)
    dropout = checkpoint.get("dropout", 0.2)

    # Auto-detect dragonnet from checkpoint keys (old checkpoints lack metadata).
    state_dicts_raw = checkpoint.get("all_state_dicts") or [checkpoint["state_dict"]]
    sample_keys = state_dicts_raw[0].keys()
    has_head_prop = any(k.startswith("head_prop") for k in sample_keys)
    dragonnet = checkpoint.get("dragonnet", has_head_prop)
    if has_head_prop != dragonnet:
        # Reconcile: if keys say it was trained with prop head, respect that.
        dragonnet = has_head_prop

    # Auto-detect hidden_dim from encoder[0] weight shape if missing.
    enc_key = "encoder.0.weight"
    if enc_key in sample_keys and "hidden_dim" not in checkpoint:
        hidden_dim = int(state_dicts_raw[0][enc_key].shape[0])

    # Fit FeatureBuilder on TRAIN (same as during training) — fixes the bug
    # where running on test alone produced different medians/categories.
    train, test, _ = load_and_validate(dataset_dir)
    X_train, X_test, _, _ = prepare_datasets(
        train, test, max_categories=40, feature_set=feature_set
    )
    # Reindex to checkpoint columns (defensive: handles any column-order drift).
    X_test = X_test.reindex(columns=feature_cols, fill_value=0.0)
    X_train = X_train.reindex(columns=feature_cols, fill_value=0.0)

    # Reproduce the same val split used during training so we get a local
    # uplift@10 estimate on data the model never saw.
    y_train = train["rec_spend"].to_numpy(dtype=np.float32)
    t_train = train["treatment_flg"].to_numpy(dtype=np.int8)
    val_mask = _make_val_mask(
        t_train, y_train, val_fraction=cfg.val_fraction, seed=RANDOM_STATE,
        strat_mode=cfg.strat_mode,
    )
    X_val = X_train.values[val_mask]
    y_val = y_train[val_mask]
    t_val = t_train[val_mask]

    state_dicts = state_dicts_raw
    scalers = checkpoint.get("all_scalers") or [checkpoint["scaler"]]

    test_uplifts: list[np.ndarray] = []
    val_uplifts: list[np.ndarray] = []
    val_scores: list[float] = []
    for i, (state_dict, scaler) in enumerate(zip(state_dicts, scalers), 1):
        model = TARNet(len(feature_cols), hidden_dim=hidden_dim, dropout=dropout, dragonnet=dragonnet)
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)

        X_val_s = torch.tensor(scaler.transform(X_val).astype(np.float32)).to(DEVICE)
        val_uplift = model.predict_uplift(X_val_s)
        val_uplifts.append(val_uplift)
        score = uplift_at_k(y_val, val_uplift, t_val, k=0.10)
        val_scores.append(score)
        logger.info("Seed %d/%d val uplift@10: %.4f", i, len(state_dicts), score)

        X_test_s = torch.tensor(scaler.transform(X_test.values).astype(np.float32)).to(DEVICE)
        test_uplifts.append(model.predict_uplift(X_test_s))

    def _rank_pct(arr: np.ndarray) -> np.ndarray:
        return pd.Series(arr).rank(method="average", pct=True).to_numpy(dtype=np.float32)

    if len(test_uplifts) > 1:
        uplift = np.mean([_rank_pct(u) for u in test_uplifts], axis=0)
        val_ens = np.mean([_rank_pct(u) for u in val_uplifts], axis=0)
        val_ens_score = uplift_at_k(y_val, val_ens, t_val, k=0.10)
        logger.info("Per-seed val uplift@10: %s",
                    ", ".join(f"{s:.4f}" for s in val_scores))
        logger.info("Ensemble (rank-avg) val uplift@10: %.4f", val_ens_score)
    else:
        uplift = test_uplifts[0]
        logger.info("Single-seed val uplift@10: %.4f", val_scores[0])

    pred_path = output_dir / "predictions_neural.csv"
    pd.DataFrame({"user_id": test["user_id"].to_numpy(), "UPLIFT_SCORE": uplift}).to_csv(
        pred_path, index=False, encoding="utf-8"
    )
    logger.info("Predictions saved: %s (%d-seed ensemble)", pred_path, len(test_uplifts))


def run_ensemble(cfg: argparse.Namespace) -> None:
    """Rank-average ensemble of LightGBM + neural predictions."""
    lgbm = pd.read_csv(cfg.lgbm_predictions)
    neural = pd.read_csv(cfg.neural_predictions)

    merged = lgbm.merge(neural, on="user_id", suffixes=("_lgbm", "_neural"))

    # Rank-average (more stable than raw average for different scales)
    merged["rank_lgbm"] = merged["UPLIFT_SCORE_lgbm"].rank(pct=True)
    merged["rank_neural"] = merged["UPLIFT_SCORE_neural"].rank(pct=True)
    merged["UPLIFT_SCORE"] = (
        cfg.lgbm_weight * merged["rank_lgbm"] + cfg.neural_weight * merged["rank_neural"]
    )

    out = merged[["user_id", "UPLIFT_SCORE"]]
    out.to_csv(cfg.output, index=False, encoding="utf-8")
    logger.info("Ensemble saved: %s  (lgbm_weight=%.2f  neural_weight=%.2f)", cfg.output, cfg.lgbm_weight, cfg.neural_weight)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "ensemble"], default="train")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output-dir", default="output/neural")
    parser.add_argument("--model-path", default="output/neural/tarnet.pt")
    # Training params
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--nonzero-weight", type=float, default=5.0)
    parser.add_argument("--no-dragonnet", action="store_true", default=True,
                        help="Disabled by default — propensity head is noise in balanced RCT")
    parser.add_argument("--with-dragonnet", dest="no_dragonnet", action="store_false")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Train this many TARNets and rank-average — cuts uplift@10 variance")
    parser.add_argument("--strat-mode", choices=("quartile", "old"), default="quartile",
                        help="'old' = original 22.23-baseline stratification (t*2 + (y>0))")
    parser.add_argument("--feature-set",
                        choices=("baseline", "event_zero", "semantic", "enhanced"),
                        default="semantic",
                        help="Feature preprocessing mode (matches features.py)")
    parser.add_argument("--honest-val-from", default=None,
                        help="Build val_mask on this OTHER dataset (real-only), then map "
                             "user_ids to current dataset. Use when training on augmented data "
                             "to avoid val leakage with synth users.")
    # Ensemble params
    parser.add_argument("--lgbm-predictions", default="output/semantic_candidate/predictions.csv")
    parser.add_argument("--neural-predictions", default="output/neural/predictions_neural.csv")
    parser.add_argument("--output", default="output/neural/predictions_ensemble.csv")
    parser.add_argument("--lgbm-weight", type=float, default=0.6)
    parser.add_argument("--neural-weight", type=float, default=0.4)

    cfg = parser.parse_args()

    if cfg.mode == "train":
        run_train(cfg)
    elif cfg.mode == "infer":
        run_infer(cfg)
    elif cfg.mode == "ensemble":
        run_ensemble(cfg)


if __name__ == "__main__":
    main()
