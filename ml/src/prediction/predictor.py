"""
모델 통합 인터페이스
saved_models/current_model.json에서 활성 모델 자동 로드
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "saved_models"


# ── 모델 클래스 (train_all_models.py와 동일) ─────────────────────────
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
        vsn_out, var_w = self.vsn(x)
        lstm_out, _    = self.lstm(vsn_out)
        lstm_out       = self.lstm_norm(self.lstm_grn(lstm_out) + vsn_out)
        attn_out, _    = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out       = self.attn_norm(self.attn_grn(attn_out) + lstm_out)
        ffn_out        = self.ffn_norm(self.ffn(attn_out) + attn_out)
        return self.head(ffn_out[:, -1, :]).squeeze(-1)


MODEL_REGISTRY = {
    'lstm':     LSTMModel,
    'cnn_lstm': CNNLSTMModel,
    'tft':      TFTModel,
}


class Predictor:
    """모델 통합 예측기"""

    def __init__(self):
        self.model      = None
        self.meta       = None
        self.scaler_mean  = None
        self.scaler_scale = None
        self.active_name  = None
        self.reload()

    def reload(self):
        """현재 활성 모델 로드"""
        with open(f"{MODELS_DIR}/current_model.json", "r", encoding='utf-8') as f:
            current = json.load(f)
        self.active_name = current['active_model']

        # 메타데이터
        with open(f"{MODELS_DIR}/{self.active_name}_meta.json", "r", encoding='utf-8') as f:
            self.meta = json.load(f)

        # 모델
        checkpoint = torch.load(f"{MODELS_DIR}/{self.active_name}.pth",
                                map_location=DEVICE)
        config     = checkpoint['config']
        model_cls  = MODEL_REGISTRY[self.active_name]
        self.model = model_cls(config['num_features']).to(DEVICE)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        # 스케일러
        scaler = np.load(f"{MODELS_DIR}/{self.active_name}_scaler.npz")
        self.scaler_mean  = scaler['mean']
        self.scaler_scale = scaler['scale']

        print(f"[Predictor] 활성 모델: {self.active_name}")
        print(f"            threshold: {self.meta['threshold']:.3f}")
        print(f"            F1: {self.meta['f1']:.4f}")

    def set_active(self, model_name, threshold=None):
        """활성 모델 변경"""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"알 수 없는 모델: {model_name}")

        with open(f"{MODELS_DIR}/{model_name}_meta.json", "r", encoding='utf-8') as f:
            meta = json.load(f)

        thr = threshold if threshold is not None else meta['threshold']
        with open(f"{MODELS_DIR}/current_model.json", "w", encoding='utf-8') as f:
            json.dump({"active_model": model_name, "threshold": thr},
                      f, indent=2, ensure_ascii=False)
        self.reload()

    def predict(self, sequence_array):
        """
        sequence_array: shape (seq_len, num_features) — 정규화 전 원본 값
        반환: dict(probability, label, threshold, model_name)
        """
        # 정규화
        x = (sequence_array - self.scaler_mean) / self.scaler_scale
        x = np.clip(x, -5, 5)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = self.model(x).cpu().item()

        return {
            'probability': prob,
            'label':       int(prob >= self.meta['threshold']),
            'threshold':   self.meta['threshold'],
            'model_name':  self.active_name,
        }

    def list_models(self):
        """저장된 모델 목록"""
        models = []
        for name in MODEL_REGISTRY:
            meta_path = f"{MODELS_DIR}/{name}_meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding='utf-8') as f:
                    meta = json.load(f)
                models.append(meta)
        return models


# ── 비즈니스 룰 (금리/한도 산출) ─────────────────────────────────────
def calc_interest_rate(pd_score, industry_risk="medium", has_collateral=False,
                       base_rate=0.035):
    """
    적정 금리 = 기준금리 + 신용 스프레드
    """
    # PD 구간별 스프레드
    if pd_score < 0.10:
        spread = 0.015
    elif pd_score < 0.30:
        spread = 0.030
    elif pd_score < 0.50:
        spread = 0.050
    else:
        return None  # 거절

    # 업종 가산
    industry_premium = {'low': 0.0, 'medium': 0.003, 'high': 0.005}.get(industry_risk, 0.003)

    # 담보 할인
    collateral_discount = -0.005 if has_collateral else 0.0

    return base_rate + spread + industry_premium + collateral_discount


    return max(recommended, 0)
def calc_loan_limit(operating_cf, total_assets, pd_score,
                     collateral_value=0, loan_period_years=3,
                     interest_rate=0.05, revenue=None):
    """현실적인 한도 산출"""
    # 1. 상환능력 기반 (현재가치 환산)
    annual_repay = operating_cf * 0.5
    if annual_repay > 0 and interest_rate > 0:
        pv_factor = (1 - (1 + interest_rate) ** -loan_period_years) / interest_rate
        repay_capacity = annual_repay * pv_factor
    else:
        repay_capacity = 0

    # 2. 매출 대비 한도 (매출의 3배)
    revenue_limit = revenue * 3 if revenue else float('inf')

    # 3. 자산 대비 한도 (총자산의 50%)
    asset_limit = total_assets * 0.5

    # 4. 신용 기반 한도 (담보 없이 결정)
    credit_limit = min(repay_capacity, revenue_limit, asset_limit)

    # 5. PD 반영
    credit_limit_adjusted = credit_limit * (1 - pd_score)

    # 6. 담보 한도 추가 (추가 한도 개념)
    collateral_limit = collateral_value * 0.7  # LTV 70%

    # 7. 최종 한도 = 신용 한도 + 담보 한도
    recommended = credit_limit_adjusted + collateral_limit

    return max(recommended, 0)


def calc_expected_loss(pd_score, ead, has_collateral=False,
                        collateral_type=None, collateral_value=0):
    """
    Basel III Expected Loss
    LGD를 담보 유형/가치에 따라 조정
    """
    # LGD 기본값 (무담보)
    lgd = 0.60

    # 담보 유형별 LGD 조정
    if has_collateral and collateral_value > 0:
        # 담보가치가 EAD의 100% 이상이면 LGD 낮음
        coverage = min(collateral_value / ead, 1.0)

        if collateral_type == "부동산":
            lgd = 0.60 * (1 - coverage * 0.7)  # 부동산은 회수 가능성 70%
        elif collateral_type == "신용보증서":
            lgd = 0.60 * (1 - coverage * 0.85) # 보증서는 회수 가능성 85%
        elif collateral_type == "기계설비":
            lgd = 0.60 * (1 - coverage * 0.5)  # 기계는 회수 가능성 50%

        lgd = max(lgd, 0.10)  # 최소 10%

    return pd_score * lgd * ead, lgd