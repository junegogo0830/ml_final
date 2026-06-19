# -*- coding: utf-8 -*-
"""
모델 로딩 및 예측기
models/saved/current_model.json에서 활성 모델 자동 로드
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "models/saved"


# ── 모델 정의 ──────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """양방향 LSTM + 어텐션 메커니즘"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        d = hidden_size * 2
        self.attn_w = nn.Linear(d, 1)
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d // 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        w      = torch.softmax(self.attn_w(out), dim=1)
        ctx    = (w * out).sum(dim=1)
        return self.head(self.norm(ctx)).squeeze(-1)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x)).unsqueeze(-1)


class CNNLSTMModel(nn.Module):
    """멀티스케일 CNN + SE 채널 어텐션 + LSTM + 시간 어텐션"""
    def __init__(self, input_size, cnn_filters=128, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn_k2 = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, 2, padding='same'),
            nn.BatchNorm1d(cnn_filters), nn.GELU(), nn.Dropout(dropout),
        )
        self.cnn_k3 = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, 3, padding='same'),
            nn.BatchNorm1d(cnn_filters), nn.GELU(), nn.Dropout(dropout),
        )
        self.merge = nn.Sequential(
            nn.Conv1d(cnn_filters * 2, cnn_filters, 1),
            nn.BatchNorm1d(cnn_filters), nn.GELU(),
        )
        self.se     = SEBlock(cnn_filters)
        self.lstm   = nn.LSTM(
            cnn_filters, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.attn_w = nn.Linear(hidden_size, 1)
        self.norm   = nn.LayerNorm(hidden_size)
        self.head   = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x_t    = x.permute(0, 2, 1)
        merged = self.merge(torch.cat([self.cnn_k2(x_t), self.cnn_k3(x_t)], dim=1))
        merged = self.se(merged)
        out, _ = self.lstm(merged.permute(0, 2, 1))
        w      = torch.softmax(self.attn_w(out), dim=1)
        ctx    = (w * out).sum(dim=1)
        return self.head(self.norm(ctx)).squeeze(-1)


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
        self.var_grns   = nn.ModuleList(
            [GRN(1, hidden_size, hidden_size, dropout) for _ in range(num_vars)]
        )
        self.select_grn = GRN(num_vars, hidden_size, num_vars, dropout)

    def forward(self, x):
        B, T, V = x.shape
        embeds  = torch.stack([self.var_grns[i](x[:, :, i:i+1]) for i in range(V)], dim=2)
        w       = torch.softmax(
            self.select_grn(x.reshape(B * T, V)).reshape(B, T, V, 1), dim=2
        )
        return (embeds * w).sum(dim=2), w.squeeze(-1)


class TFTModel(nn.Module):
    def __init__(self, num_features, hidden_size=32, num_heads=2, dropout=0.2):
        super().__init__()
        self.vsn       = VSN(num_features, hidden_size, dropout)
        self.lstm      = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm_grn  = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.attn      = nn.MultiheadAttention(hidden_size, num_heads,
                                               dropout=dropout, batch_first=True)
        self.attn_grn  = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn       = GRN(hidden_size, hidden_size * 2, hidden_size, dropout)
        self.ffn_norm  = nn.LayerNorm(hidden_size)
        self.head       = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        vsn_out, _  = self.vsn(x)
        lstm_out, _ = self.lstm(vsn_out)
        lstm_out    = self.lstm_norm(self.lstm_grn(lstm_out) + vsn_out)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out    = self.attn_norm(self.attn_grn(attn_out) + lstm_out)
        ffn_out     = self.ffn_norm(self.ffn(attn_out) + attn_out)
        return self.head(ffn_out[:, -1, :]).squeeze(-1)


MODEL_REGISTRY = {
    'lstm':     LSTMModel,
    'cnn_lstm': CNNLSTMModel,
    'tft':      TFTModel,
}


class _ProbOutputWrapper(nn.Module):
    """모델 출력을 (batch, 1) 형태로 맞춰 SHAP GradientExplainer와 호환"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(-1)


# ── Predictor ──────────────────────────────────────────────────────────────

class Predictor:
    def __init__(self):
        self.model        = None
        self.meta         = None
        self.scaler_mean  = None
        self.scaler_scale = None
        self.active_name  = None
        self.reload()

    def reload(self):
        with open(f"{MODELS_DIR}/current_model.json", "r", encoding="utf-8") as f:
            current = json.load(f)
        self.active_name = current["active_model"]

        with open(f"{MODELS_DIR}/{self.active_name}_meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        checkpoint = torch.load(f"{MODELS_DIR}/{self.active_name}.pth",
                                map_location=DEVICE)
        config     = checkpoint["config"]
        model_cls  = MODEL_REGISTRY[self.active_name]
        self.model = model_cls(config["num_features"]).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        scaler            = np.load(f"{MODELS_DIR}/{self.active_name}_scaler.npz")
        self.scaler_mean  = scaler["mean"]
        self.scaler_scale = scaler["scale"]

        print(f"[Predictor] 활성 모델: {self.active_name}")
        print(f"            threshold: {self.meta['threshold']:.3f}")
        print(f"            F1: {self.meta['f1']:.4f}")

    def set_active(self, model_name: str, threshold: float = None):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"알 수 없는 모델: {model_name}")
        with open(f"{MODELS_DIR}/{model_name}_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        thr = threshold if threshold is not None else meta["threshold"]
        with open(f"{MODELS_DIR}/current_model.json", "w", encoding="utf-8") as f:
            json.dump({"active_model": model_name, "threshold": thr}, f,
                      indent=2, ensure_ascii=False)
        self.reload()

    def predict(self, sequence_array: np.ndarray) -> dict:
        """
        sequence_array: (seq_len, num_features)
        반환: dict(probability, label, threshold, model_name)
        """
        x = (sequence_array - self.scaler_mean) / self.scaler_scale
        x = np.clip(x, -5, 5)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = self.model(x).cpu().item()
        return {
            "probability": prob,
            "label":       int(prob >= self.meta["threshold"]),
            "threshold":   self.meta["threshold"],
            "model_name":  self.active_name,
        }

    def explain(self, sequence_array: np.ndarray, background_array: np.ndarray,
                top_n: int = 5) -> list:
        """
        SHAP(GradientExplainer) 기반 피처 기여도 산출.
        sequence_array:    (seq_len, num_features) 원본(미스케일) 값
        background_array:  (n_bg, seq_len, num_features) 원본(미스케일) 값
        반환: [{"feature": str, "shap": float, "value": float}, ...]
              shap > 0  → 파산 확률을 높이는 방향으로 기여
              shap < 0  → 파산 확률을 낮추는 방향으로 기여
              결과는 shap 값 내림차순(위험 기여 큰 순) 정렬
        """
        import shap

        x  = np.clip((sequence_array - self.scaler_mean) / self.scaler_scale, -5, 5)
        bg = np.clip((background_array - self.scaler_mean) / self.scaler_scale, -5, 5)

        x_t  = torch.tensor(x,  dtype=torch.float32).unsqueeze(0).to(DEVICE)
        bg_t = torch.tensor(bg, dtype=torch.float32).to(DEVICE)

        wrapped   = _ProbOutputWrapper(self.model)
        explainer = shap.GradientExplainer(wrapped, bg_t)
        sv        = explainer.shap_values(x_t)          # (1, seq_len, num_features, 1)
        contrib   = np.asarray(sv)[0, :, :, 0].sum(axis=0)  # (num_features,)

        feature_cols = self.meta["feature_cols"]
        results = [
            {"feature": fc, "shap": float(contrib[i]), "value": float(sequence_array[-1, i])}
            for i, fc in enumerate(feature_cols)
        ]
        results.sort(key=lambda r: r["shap"], reverse=True)
        return results[:top_n] if top_n else results

    def list_models(self) -> list:
        models = []
        for name in MODEL_REGISTRY:
            path = f"{MODELS_DIR}/{name}_meta.json"
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    models.append(json.load(f))
        return models


# ── 여신 계산 ──────────────────────────────────────────────────────────────

def calc_interest_rate(pd_score: float, industry_risk: str = "medium",
                       has_collateral: bool = False, base_rate: float = 0.035) -> float | None:
    if pd_score < 0.10:
        spread = 0.015
    elif pd_score < 0.30:
        spread = 0.030
    elif pd_score < 0.50:
        spread = 0.050
    else:
        return None

    industry_premium     = {"low": 0.0, "medium": 0.003, "high": 0.005}.get(industry_risk, 0.003)
    collateral_discount  = -0.005 if has_collateral else 0.0
    return base_rate + spread + industry_premium + collateral_discount


def calc_loan_limit(operating_cf: float, total_assets: float, pd_score: float,
                    collateral_value: float = 0, loan_period_years: int = 3,
                    interest_rate: float = 0.05, revenue: float = None) -> float:
    annual_repay = operating_cf * 0.5
    if annual_repay > 0 and interest_rate > 0:
        pv_factor       = (1 - (1 + interest_rate) ** -loan_period_years) / interest_rate
        repay_capacity  = annual_repay * pv_factor
    else:
        repay_capacity  = 0

    revenue_limit          = revenue * 3 if revenue else float("inf")
    asset_limit            = total_assets * 0.5
    credit_limit           = min(repay_capacity, revenue_limit, asset_limit)
    credit_limit_adjusted  = credit_limit * (1 - pd_score)
    collateral_limit       = collateral_value * 0.7
    return max(credit_limit_adjusted + collateral_limit, 0)


def calc_expected_loss(pd_score: float, ead: float, has_collateral: bool = False,
                       collateral_type: str = None, collateral_value: float = 0) -> tuple:
    lgd = 0.60
    if has_collateral and collateral_value > 0:
        coverage = min(collateral_value / ead, 1.0)
        if collateral_type == "부동산":
            lgd = 0.60 * (1 - coverage * 0.7)
        elif collateral_type == "보증서":
            lgd = 0.60 * (1 - coverage * 0.85)
        elif collateral_type == "유가증권":
            lgd = 0.60 * (1 - coverage * 0.5)
        lgd = max(lgd, 0.10)
    return pd_score * lgd * ead, lgd
