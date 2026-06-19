# -*- coding: utf-8 -*-
"""
새 버전 학습 - 2단계: dart_v2.db 데이터로 LSTM 파산예측 모델 재학습
실행: python scripts/retrain_lstm.py  (프로젝트 루트에서)

데이터: financials(corp_code별 마지막 SEQ_LEN개 연도) + labels
저장:
  models/saved/lstm.pth          (state_dict + config)
  models/saved/lstm_meta.json    (threshold, metrics, feature_cols 등)
  models/saved/lstm_scaler.npz   (mean, scale)
  models/saved/current_model.json (active_model 갱신)
기존 파일은 models/saved/backup_<timestamp>/ 로 백업 후 교체.
"""

import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                              roc_auc_score, roc_curve, average_precision_score)
from sklearn.model_selection import train_test_split

ROOT       = Path(__file__).resolve().parent.parent
DB_PATH    = ROOT / "db" / "dart_v2.db"
MODELS_DIR = ROOT / "models" / "saved"
STATUS_PATH = MODELS_DIR / "pipeline_status.json"

sys.path.insert(0, str(ROOT / "backend"))
from predictor import LSTMModel  # noqa: E402

SEQ_LEN  = 5
SEED     = 42
EPOCHS   = 80
LR       = 1e-3
PATIENCE = 12
DEVICE   = "cpu"

with open(MODELS_DIR / "lstm_meta.json", "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)["feature_cols"]
NUM_FEATURES = len(FEATURE_COLS)

# financials(dart_v2) 컬럼명 → 모델 feature 이름 매핑 (backend/app.py 와 동일)
DART_COL_MAP = {
    "debt_ratio":              "debt_ratio",
    "current_ratio":           "current_ratio",
    "interest_coverage_proxy": "interest_coverage",
    "equity_ratio":            "equity_ratio",
    "roa":                     "roa",
    "roe":                     "roe",
    "operating_margin":        "op_margin",
    "net_margin":              "net_margin",
    "ocf_to_current_liab":     "cfo_to_debt",
    "z_score":                 "z_score",
}


def write_status(**kw):
    cur = {}
    if STATUS_PATH.exists():
        try:
            cur = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    cur.update(kw)
    cur["updated_at"] = datetime.now().isoformat()
    STATUS_PATH.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, prob, target):
        eps = 1e-7
        prob = prob.clamp(eps, 1 - eps)
        pt = prob * target + (1 - prob) * (1 - target)
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        ce = -torch.log(pt)
        return (at * (1 - pt) ** self.gamma * ce).mean()


# ── 데이터셋 구성 ────────────────────────────────────────────────────
def build_dataset():
    conn = sqlite3.connect(DB_PATH)
    fin    = pd.read_sql("SELECT * FROM financials ORDER BY corp_code, year", conn)
    labels = pd.read_sql("SELECT corp_code, label FROM labels", conn)
    conn.close()

    for dc, mc in DART_COL_MAP.items():
        if dc in fin.columns and mc not in fin.columns:
            fin[mc] = fin[dc]
    for fc in FEATURE_COLS:
        if fc not in fin.columns:
            fin[fc] = 0.0

    label_map = dict(zip(labels["corp_code"], labels["label"]))

    X, y, codes = [], [], []
    for cc, g in fin.groupby("corp_code"):
        if cc not in label_map:
            continue
        g = g.sort_values("year")
        if (g["fs_div"] == "CFS").any():
            g = g[g["fs_div"] == "CFS"]
        feats = g[FEATURE_COLS].fillna(0).values.astype("float32")
        if len(feats) >= SEQ_LEN:
            seq = feats[-SEQ_LEN:]
        else:
            pad = np.zeros((SEQ_LEN - len(feats), NUM_FEATURES), dtype="float32")
            seq = np.vstack([pad, feats])
        X.append(seq)
        y.append(int(label_map[cc]))
        codes.append(cc)

    return np.array(X, dtype="float32"), np.array(y, dtype="float32"), codes


# ── 학습 ─────────────────────────────────────────────────────────────
def main():
    write_status(running=True, step="retrain", message="모델 재학습 데이터 구성 중", error=None)

    X, y, codes = build_dataset()
    print(f"[retrain_lstm] 전체 샘플: {len(X)}  (양성 {int(y.sum())}, 음성 {int((1-y).sum())})")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    mean = X_tr.reshape(-1, NUM_FEATURES).mean(axis=0)
    std  = X_tr.reshape(-1, NUM_FEATURES).std(axis=0)
    std[std == 0] = 1.0

    def scale(arr):
        return np.clip((arr - mean) / std, -5, 5)

    X_tr_s, X_val_s = scale(X_tr), scale(X_val)

    train_x = torch.tensor(X_tr_s, dtype=torch.float32)
    train_y = torch.tensor(y_tr, dtype=torch.float32)
    val_x   = torch.tensor(X_val_s, dtype=torch.float32)
    val_y   = torch.tensor(y_val, dtype=torch.float32)

    torch.manual_seed(SEED)
    model = LSTMModel(NUM_FEATURES).to(DEVICE)
    pos_ratio = float(y_tr.mean())
    criterion = FocalLoss(alpha=1 - pos_ratio, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    best_pr, best_state, no_improve = -1.0, None, 0
    batch_size = 64
    n = len(train_x)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = train_x[idx], train_y[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        total_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x).numpy()
        val_pr = average_precision_score(y_val, val_pred)

        if val_pr > best_pr:
            best_pr, best_state, no_improve = val_pr, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep {epoch:3d}/{EPOCHS}  loss={total_loss:.4f}  val_PR={val_pr:.4f}  best={best_pr:.4f}")
            write_status(message=f"모델 재학습 중 (epoch {epoch}/{EPOCHS}, val_PR={best_pr:.4f})",
                         progress={"epoch": epoch, "epochs": EPOCHS, "val_pr": best_pr})

        if no_improve >= PATIENCE:
            print(f"  조기 종료 ep {epoch}/{EPOCHS}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(val_x).numpy()

    roc_auc = roc_auc_score(y_val, val_pred)
    pr_auc  = average_precision_score(y_val, val_pred)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_val, val_pred)
    f1s = np.where((precisions + recalls) > 0,
                    2 * precisions * recalls / (precisions + recalls + 1e-12), 0)
    best_idx   = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
    threshold  = float(pr_thresholds[best_idx]) if len(pr_thresholds) > 0 else 0.5
    f1         = float(f1s[best_idx])
    fpr, tpr, _ = roc_curve(y_val, val_pred)
    cm = confusion_matrix(y_val, (val_pred >= threshold).astype(int)).tolist()

    print(f"[retrain_lstm] ROC {roc_auc:.4f}  PR {pr_auc:.4f}  F1 {f1:.4f}  threshold {threshold:.4f}")

    # ── 백업 후 저장 ──────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = MODELS_DIR / f"backup_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["lstm.pth", "lstm_meta.json", "lstm_scaler.npz", "current_model.json"]:
        src = MODELS_DIR / fname
        if src.exists():
            shutil.copy2(src, backup_dir / fname)

    torch.save({
        "model_state": model.state_dict(),
        "config": {"feature_cols": FEATURE_COLS, "seq_len": SEQ_LEN, "num_features": NUM_FEATURES},
    }, MODELS_DIR / "lstm.pth")

    np.savez(MODELS_DIR / "lstm_scaler.npz", mean=mean, scale=std)

    meta = {
        "model_name": "lstm",
        "threshold": threshold,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": f1,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "precision_curve": precisions.tolist(),
        "recall_curve": recalls.tolist(),
        "fpr_curve": fpr.tolist(),
        "tpr_curve": tpr.tolist(),
        "confusion_matrix": cm,
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_train": len(X_tr),
        "n_val": len(X_val),
    }
    with open(MODELS_DIR / "lstm_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(MODELS_DIR / "current_model.json", "w", encoding="utf-8") as f:
        json.dump({
            "active_model": "lstm",
            "threshold": threshold,
            "note": f"LSTM (양방향 + 어텐션) - {datetime.now().strftime('%Y-%m-%d %H:%M')} 재학습",
        }, f, ensure_ascii=False, indent=2)

    write_status(message=f"재학습 완료 — ROC {roc_auc:.3f} / PR {pr_auc:.3f} / F1 {f1:.3f}",
                  result={"roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1,
                          "threshold": threshold, "n_samples": len(X), "backup_dir": str(backup_dir)})
    print(f"[retrain_lstm] 백업: {backup_dir}")


if __name__ == "__main__":
    main()
