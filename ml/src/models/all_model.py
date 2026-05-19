"""
3개 모델 일괄 학습 및 저장
실행: python train_all_models.py

LSTM / CNN+LSTM / TFT를 학습하고
saved_models/ 폴더에 .pth + 메타데이터 저장
"""

import os, json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# ── 설정 ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'debt_ratio', 'current_ratio', 'interest_coverage',
    'net_debt_ratio', 'equity_ratio',
    'roa', 'roe', 'op_margin', 'net_margin',
    'cfo_to_debt', 'fcf',
    'revenue_growth', 'op_income_growth', 'asset_growth',
    'interest_cov_yoy', 'debt_ratio_trend', 'cf_volatility',
    'consecutive_loss', 'z_score',
    'sentiment_avg', 'negative_ratio', 'news_count',
    'news_bankruptcy_flag', 'news_lawsuit_flag'
]

SEQ_LEN     = 5
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("saved_models", exist_ok=True)


# ── 데이터셋 ──────────────────────────────────────────────────────────
class BankruptcyDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len=5):
        self.sequences, self.labels = [], []
        df = df.sort_values(['corp_code', 'year']).copy()
        for _, group in df.groupby('corp_code'):
            feats = group[feature_cols].values.astype(np.float32)
            label = int(group['label'].iloc[-1])
            if len(feats) >= seq_len:
                seq = feats[-seq_len:]
            else:
                pad = np.zeros((seq_len - len(feats), len(feature_cols)), dtype=np.float32)
                seq = np.vstack([pad, feats])
            self.sequences.append(seq)
            self.labels.append(label)
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels    = np.array(self.labels,    dtype=np.float32)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])


# ── 모델 클래스 ───────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1), nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.norm(out[:, -1, :])).squeeze(-1)


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, cnn_filters=64, kernel_size=2,
                 hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size, padding='same'),
            nn.BatchNorm1d(cnn_filters), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size, padding='same'),
            nn.BatchNorm1d(cnn_filters), nn.ReLU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(cnn_filters, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1), nn.Sigmoid()
        )
    def forward(self, x):
        cnn_out = self.cnn(x.permute(0, 2, 1))
        lstm_out, _ = self.lstm(cnn_out.permute(0, 2, 1))
        return self.head(self.norm(lstm_out[:, -1, :])).squeeze(-1)


class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(input_size, hidden_size)
        self.fc2  = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None
    def forward(self, x):
        r = x if self.skip is None else self.skip(x)
        h = self.drop(F.elu(self.fc1(x)))
        return self.norm(self.fc2(h) * torch.sigmoid(self.gate(h)) + r)


class VSN(nn.Module):
    def __init__(self, num_vars, hidden_size, dropout=0.1):
        super().__init__()
        self.var_grns   = nn.ModuleList([GRN(1, hidden_size, hidden_size, dropout) for _ in range(num_vars)])
        self.select_grn = GRN(num_vars, hidden_size, num_vars, dropout)
    def forward(self, x):
        B, T, V = x.shape
        embeds = torch.stack([self.var_grns[i](x[:,:,i:i+1]) for i in range(V)], dim=2)
        w      = torch.softmax(self.select_grn(x.reshape(B*T, V)).reshape(B, T, V, 1), dim=2)
        return (embeds * w).sum(dim=2), w.squeeze(-1)


class TFTModel(nn.Module):
    def __init__(self, num_features, hidden_size=32, num_heads=2, dropout=0.2):
        super().__init__()
        self.vsn       = VSN(num_features, hidden_size, dropout)
        self.lstm      = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm_grn  = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.attn      = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.attn_grn  = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn       = GRN(hidden_size, hidden_size*2, hidden_size, dropout)
        self.ffn_norm  = nn.LayerNorm(hidden_size)
        self.head      = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1), nn.Sigmoid()
        )
    def forward(self, x):
        vsn_out, _ = self.vsn(x)
        lstm_out, _ = self.lstm(vsn_out)
        lstm_out = self.lstm_norm(self.lstm_grn(lstm_out) + vsn_out)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(self.attn_grn(attn_out) + lstm_out)
        ffn_out  = self.ffn_norm(self.ffn(attn_out) + attn_out)
        return self.head(ffn_out[:, -1, :]).squeeze(-1)


# ── 학습 함수 ─────────────────────────────────────────────────────────
def train_one_model(model, train_loader, val_loader, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()

    best_val, best_state = float('inf'), None
    print(f"\n[{name}] 학습 시작 | 파라미터: {sum(p.numel() for p in model.parameters()):,}개")

    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                val_loss += criterion(model(X.to(DEVICE)), y.to(DEVICE)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


def evaluate_and_save(model, test_df, features, scaler, name):
    test_ds = BankruptcyDataset(test_df, features, SEQ_LEN)
    loader  = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            all_probs.append(model(X.to(DEVICE)).cpu().numpy())
            all_labels.append(y.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    # 최적 threshold 산출
    p, r, thrs = precision_recall_curve(labels, probs)
    f1_scores  = 2 * p * r / (p + r + 1e-8)
    best_idx   = f1_scores[:-1].argmax()
    best_thr   = float(thrs[best_idx])
    best_f1    = float(f1_scores[best_idx])

    roc_auc = float(roc_auc_score(labels, probs))
    pr_auc  = float(auc(r, p))

    # 모델 저장
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'feature_cols': features,
            'seq_len':      SEQ_LEN,
            'num_features': len(features),
        }
    }, f"saved_models/{name}.pth")

    # 스케일러 저장
    np.savez(f"saved_models/{name}_scaler.npz",
             mean=scaler.mean_, scale=scaler.scale_)

    # 메타데이터 저장
    with open(f"saved_models/{name}_meta.json", "w", encoding='utf-8') as f:
        json.dump({
            'model_name':  name,
            'threshold':   best_thr,
            'roc_auc':     roc_auc,
            'pr_auc':      pr_auc,
            'f1':          best_f1,
            'feature_cols': features,
            'seq_len':     SEQ_LEN,
        }, f, indent=2, ensure_ascii=False)

    print(f"[{name}] ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {best_f1:.4f} | thr: {best_thr:.3f}")
    return {'name': name, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'f1': best_f1, 'threshold': best_thr}


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("3개 모델 일괄 학습 및 저장")
    print("="*60)

    print("\n데이터 로드 중...")
    train = pd.read_csv("data/processed/train_augmented.csv")
    test  = pd.read_csv("data/raw/test.csv")

    features = [c for c in FEATURE_COLS if c in train.columns]
    print(f"사용 피처: {len(features)}개")

    # 전처리
    for col in features:
        m = train[col].median()
        train[col] = train[col].fillna(m)
        test[col]  = test[col].fillna(m)

    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features]  = scaler.transform(test[features])

    for col in features:
        train[col] = train[col].clip(-5, 5)
        test[col]  = test[col].clip(-5, 5)

    # 데이터 로더
    train_ds = BankruptcyDataset(train, features, SEQ_LEN)
    val_size = int(len(train_ds) * 0.2)
    train_size = len(train_ds) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    train_loader = DataLoader(train_subset, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_subset,   BATCH_SIZE, shuffle=False)

    results = []

    # 1. LSTM
    lstm = LSTMModel(len(features)).to(DEVICE)
    lstm = train_one_model(lstm, train_loader, val_loader, "lstm")
    results.append(evaluate_and_save(lstm, test, features, scaler, "lstm"))

    # 2. CNN+LSTM
    cnn_lstm = CNNLSTMModel(len(features)).to(DEVICE)
    cnn_lstm = train_one_model(cnn_lstm, train_loader, val_loader, "cnn_lstm")
    results.append(evaluate_and_save(cnn_lstm, test, features, scaler, "cnn_lstm"))

    # 3. TFT
    tft = TFTModel(len(features)).to(DEVICE)
    tft = train_one_model(tft, train_loader, val_loader, "tft")
    results.append(evaluate_and_save(tft, test, features, scaler, "tft"))

    # 현재 활성 모델 설정 (최고 F1 모델로)
    best  = max(results, key=lambda x: x['f1'])
    with open("saved_models/current_model.json", "w", encoding='utf-8') as f:
        json.dump({
            "active_model": best['name'],
            "threshold":    best['threshold'],
            "note":         "F1 기준 최고 모델"
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("최종 비교")
    print("="*60)
    print(f"{'모델':<12} {'ROC-AUC':>10} {'PR-AUC':>10} {'F1':>10} {'threshold':>12}")
    for r in results:
        print(f"{r['name']:<12} {r['roc_auc']:>10.4f} {r['pr_auc']:>10.4f} {r['f1']:>10.4f} {r['threshold']:>12.3f}")
    print(f"\n활성 모델: {best['name']}")
    print("저장 완료: saved_models/")


if __name__ == "__main__":
    main()