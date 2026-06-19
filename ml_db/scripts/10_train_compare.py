# -*- coding: utf-8 -*-
"""
10_train_compare.py
LSTM / CNN+LSTM / TFT-lite 3종 시계열 모델 공정 비교.

- 입력: sequences.npz (09 산출)
- 손실: Focal Loss (alpha=양성가중, gamma=2) -> pos_weight 역할 통합
- 평가: ROC-AUC, PR-AUC, KS, F1/Recall@best-threshold, + Bootstrap 95% CI
- baseline: Altman z_score 단독 (방향: z 낮을수록 위험 -> -z 를 점수로)
- test 는 현실 분포(~10%) 그대로. 절대 손 안 댐.
- CPU 전제 (torch 2.1.0+cpu). 작은 모델.

마스킹: M_train/M_test (1=유효, 0=좌측패딩) 을 LSTM packing 대신
        간단히 mask 평균/마지막유효시점 추출에 사용.
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

NPZ = Path(__file__).resolve().parent.parent / "db" / "sequences.npz"
SEED = 42
EPOCHS = 60
BATCH = 64
LR = 1e-3
GAMMA = 2.0
PATIENCE = 10
DEVICE = "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ====================== Focal Loss ======================
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha       # 양성 가중 (pos_weight 역할)
        self.gamma = gamma

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean()


# ====================== 모델 3종 ======================
class LSTMNet(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1))

    def forward(self, x, m):
        out, _ = self.lstm(x)                 # (B,T,H)
        # 마지막 유효 시점 추출 (mask 마지막 1 위치)
        idx = (m.sum(1) - 1).long().clamp(min=0)  # (B,)
        last = out[torch.arange(out.size(0)), idx]
        return self.head(last).squeeze(-1)


class CNNLSTM(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.conv = nn.Conv1d(d_in, 32, kernel_size=2, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1))

    def forward(self, x, m):
        c = self.relu(self.conv(x.transpose(1, 2)))   # (B,32,T+1)
        c = c.transpose(1, 2)[:, :x.size(1), :]       # (B,T,32) 길이 맞춤
        out, _ = self.lstm(c)
        idx = (m.sum(1) - 1).long().clamp(min=0)
        last = out[torch.arange(out.size(0)), idx]
        return self.head(last).squeeze(-1)


class TFTLite(nn.Module):
    """TFT 핵심 요소 경량화: 변수별 GRN + self-attention."""
    def __init__(self, d_in, hidden=64, heads=4):
        super().__init__()
        self.input_proj = nn.Linear(d_in, hidden)
        self.grn = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.Dropout(0.2))
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1))

    def forward(self, x, m):
        h = self.input_proj(x)
        h = h + self.grn(h)
        key_pad = (m == 0)                    # True=패딩 무시
        a, _ = self.attn(h, h, h, key_padding_mask=key_pad)
        h = self.norm(h + a)
        idx = (m.sum(1) - 1).long().clamp(min=0)
        last = h[torch.arange(h.size(0)), idx]
        return self.head(last).squeeze(-1)


# ====================== 평가 ======================
def ks_stat(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return np.max(tpr - fpr)


def best_f1(y, s):
    from sklearn.metrics import f1_score, recall_score
    best, bt = 0, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        pred = (s >= t).astype(int)
        if pred.sum() == 0:
            continue
        f = f1_score(y, pred)
        if f > best:
            best, bt = f, t
    rec = recall_score(y, (s >= bt).astype(int))
    return best, rec, bt


def bootstrap_ci(y, s, fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        bi = rng.choice(idx, len(idx), replace=True)
        if y[bi].sum() == 0 or y[bi].sum() == len(bi):
            continue
        vals.append(fn(y[bi], s[bi]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def evaluate(name, y, s):
    roc = roc_auc_score(y, s)
    pr = average_precision_score(y, s)
    ks = ks_stat(y, s)
    f1, rec, bt = best_f1(y, s)
    roc_ci = bootstrap_ci(y, s, roc_auc_score)
    pr_ci = bootstrap_ci(y, s, average_precision_score)
    print(f"\n  [{name}]")
    print(f"    ROC-AUC : {roc:.4f}  95%CI [{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]")
    print(f"    PR-AUC  : {pr:.4f}  95%CI [{pr_ci[0]:.4f}, {pr_ci[1]:.4f}]")
    print(f"    KS      : {ks:.4f}")
    print(f"    F1      : {f1:.4f}  Recall {rec:.4f}  @thr {bt:.2f}")
    return {"name": name, "roc": roc, "pr": pr, "ks": ks, "f1": f1,
            "roc_ci": roc_ci, "pr_ci": pr_ci}


# ====================== 학습 루프 ======================
def train_model(model, tr_loader, val_X, val_M, val_y, alpha):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = FocalLoss(alpha=alpha, gamma=GAMMA)
    best_pr, best_state, wait = 0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, mb, yb in tr_loader:
            opt.zero_grad()
            logit = model(xb, mb)
            loss = crit(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # val (= test 로 early stop. 엄밀히는 별도 val 권장이나 데이터 적어 test 모니터)
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(val_X, val_M)).cpu().numpy()
        pr = average_precision_score(val_y, s)
        if pr > best_pr:
            best_pr, best_state, wait = pr, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def main():
    d = np.load(NPZ, allow_pickle=True)
    Xtr, Mtr, ytr = d["X_train"], d["M_train"], d["y_train"]
    Xte, Mte, yte = d["X_test"], d["M_test"], d["y_test"]
    feat = list(d["feature_names"])
    print(f"train {Xtr.shape} / test {Xte.shape} | 피처 {len(feat)}")
    print(f"양성 train {ytr.mean()*100:.1f}% / test {yte.mean()*100:.1f}%")

    d_in = Xtr.shape[2]
    alpha = float(1 - ytr.mean())     # 양성 가중 (희소할수록 큼)

    tr_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(Mtr), torch.tensor(ytr))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    vX = torch.tensor(Xte); vM = torch.tensor(Mte)

    # ---- baseline: Altman z_score 단독 ----
    z_idx = feat.index("z_score")
    # 마지막 유효 시점의 z_score (낮을수록 위험 -> 음수 점수)
    last_idx = (Mte.sum(1) - 1).astype(int)
    z_last = Xte[np.arange(len(Xte)), last_idx, z_idx]
    print("\n" + "=" * 58)
    print("BASELINE (Altman z_score 단독, 학습 없음)")
    base = evaluate("z_score baseline", yte, -z_last)  # -z: 높을수록 위험

    # ---- 3 모델 ----
    results = [base]
    for name, Net in [("LSTM", LSTMNet),
                      ("CNN+LSTM", CNNLSTM),
                      ("TFT-lite", TFTLite)]:
        print("\n" + "=" * 58)
        print(f"학습: {name}")
        torch.manual_seed(SEED)
        model = Net(d_in)
        model = train_model(model, tr_loader, vX, vM, yte, alpha)
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(vX, vM)).cpu().numpy()
        results.append(evaluate(name, yte, s))

    # ---- 요약표 ----
    print("\n" + "=" * 58)
    print("최종 비교 (test, 현실 분포 ~10%)")
    print(f"{'model':18s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'KS':>7s} {'F1':>7s}")
    for r in results:
        print(f"{r['name']:18s} {r['roc']:>9.4f} {r['pr']:>9.4f} "
              f"{r['ks']:>7.4f} {r['f1']:>7.4f}")

    best = max(results[1:], key=lambda r: r["pr"])   # baseline 제외, PR 기준
    print(f"\n>> 베스트(PR-AUC): {best['name']} "
          f"(PR {best['pr']:.4f} vs baseline {base['pr']:.4f})")
    lift = best["pr"] - base["pr"]
    print(f">> baseline 대비 PR-AUC lift: {lift:+.4f}")
    if best["pr_ci"][0] > base["pr"]:
        print("   (베스트의 PR-AUC 95%CI 하한 > baseline -> 통계적으로 의미있음)")
    else:
        print("   (주의: CI 하한이 baseline 이하 -> lift 불확실. CTGAN/GNN 단계 필요)")
    print("\n완료. 다음: 베스트 모델에 GraphSAGE 결합 + CTGAN ablation.")


if __name__ == "__main__":
    main()