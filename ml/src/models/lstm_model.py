"""
LSTM 기업 파산 예측 모델
실행: python lstm_model.py
필요: torch, pandas, numpy, scikit-learn
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve,
    auc, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

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

SEQ_LEN    = 5
HIDDEN     = 64
N_LAYERS   = 2
DROPOUT    = 0.3
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3
RANDOM_SEED= 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"디바이스: {DEVICE}")


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


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.norm(out[:, -1, :])
        return self.head(out).squeeze(-1)


def load_and_preprocess():
    print("데이터 로드 중...")
    train = pd.read_csv("data/processed/train_augmented.csv")
    test  = pd.read_csv("data/raw/test.csv")
    print(f"학습: {len(train)}행 / {train['corp_code'].nunique()}개 기업")
    print(f"테스트: {len(test)}행 / {test['corp_code'].nunique()}개 기업")

    available = [c for c in FEATURE_COLS if c in train.columns]
    print(f"사용 피처: {len(available)}개")

    for col in available:
        m = train[col].median()
        train[col] = train[col].fillna(m)
        test[col]  = test[col].fillna(m)

    scaler = StandardScaler()
    train[available] = scaler.fit_transform(train[available])
    test[available]  = scaler.transform(test[available])

    for col in available:
        train[col] = train[col].clip(-5, 5)
        test[col]  = test[col].clip(-5, 5)

    return train, test, available


def train_model(train_df, feature_cols):
    dataset    = BankruptcyDataset(train_df, feature_cols, SEQ_LEN)
    val_size   = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)

    model     = LSTMModel(len(feature_cols), HIDDEN, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()

    print(f"\nLSTM 학습 시작 | 파라미터: {sum(p.numel() for p in model.parameters()):,}개")
    print(f"학습: {train_size}개 / 검증: {val_size}개\n")

    best_val_loss, best_state = float('inf'), None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                val_loss += criterion(model(X.to(DEVICE)), y.to(DEVICE)).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    model.load_state_dict(best_state)
    print(f"\n최적 검증 손실: {best_val_loss:.4f}")
    return model


def evaluate(model, test_df, feature_cols):
    dataset = BankruptcyDataset(test_df, feature_cols, SEQ_LEN)
    loader  = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            all_probs.append(model(X.to(DEVICE)).cpu().numpy())
            all_labels.append(y.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(labels, probs)
    p, r, _ = precision_recall_curve(labels, probs)
    pr_auc  = auc(r, p)
    f1      = f1_score(labels, preds)
    cm      = confusion_matrix(labels, preds)

    print("\n" + "="*50)
    print("LSTM 평가 결과")
    print("="*50)
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\n혼동 행렬:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(f"\n분류 리포트:")
    print(classification_report(labels, preds, target_names=['정상', '파산']))

    pd.DataFrame({'true_label': labels, 'pred_prob': probs, 'pred_label': preds})\
      .to_csv("data/predictions/lstm_predictions.csv", index=False)
    print("예측 결과 저장: lstm_predictions.csv")

    return {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'f1': f1}


if __name__ == "__main__":
    print("="*50)
    print("LSTM 파산 예측 모델")
    print("="*50)
    train, test, features = load_and_preprocess()
    model   = train_model(train, features)
    results = evaluate(model, test, features)
    print("\n=== 최종 결과 ===")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"PR-AUC:  {results['pr_auc']:.4f}")
    print(f"F1:      {results['f1']:.4f}")