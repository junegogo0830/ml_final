# -*- coding: utf-8 -*-
"""
09_build_sequences.py
financials (corp, year) 평면 -> 기업별 시계열 텐서 + split.

산출물 (npz):
  X_train, M_train, y_train, cc_train   (M = padding mask)
  X_test,  M_test,  y_test,  cc_test
  feature_names, scaler_mean, scaler_std

누수 통제 (전부 여기서):
  1. 양성: label_year 이전 연도만 사용 (label_date 당해/이후 컷)
  2. 음성: pseudo_obs_year 부여(양성 label_year 분포에서 샘플) -> 그 이전만
     -> 양성/음성 관측 시점 분포 동일하게 (시점 지름길 차단)
  3. firm-level split: 같은 기업이 train/test 양쪽에 없음
  4. temporal split: obs_year <= 2022 -> train, >= 2023 -> test
  5. 표준화(mean/std): train 시퀀스 값으로만 fit
  6. 금융업(financial_sector) 제외
  7. is_capital_impaired 를 피처에 포함 (자본잠식 신호 보존)

윈도우: 최대 5시점, 최소 2시점, 좌측(앞) 패딩 + mask
"""

import sqlite3
from pathlib import Path
import numpy as np

# ============================ CONFIG ============================
_DB_DIR = Path(__file__).resolve().parent.parent / "db"
DB = _DB_DIR / "dart_v2.db"
OUT = _DB_DIR / "sequences.npz"

MAX_T = 5          # 최대 시퀀스 길이
MIN_T = 2          # 최소 시점 (이보다 적으면 제외)
TEMPORAL_SPLIT_YEAR = 2023   # >= 이면 test
SEED = 42

# 19개 비율 + 자본잠식 플래그
RATIO_COLS = [
    "debt_ratio", "equity_ratio", "debt_to_assets", "noncurrent_liab_ratio",
    "current_ratio", "quick_ratio", "cash_ratio", "roa", "roe",
    "operating_margin", "net_margin", "gross_margin", "asset_turnover",
    "inventory_turnover", "interest_coverage_proxy", "ocf_to_current_liab",
    "retained_earnings_ratio", "working_capital_ratio", "z_score",
]
FLAG_COLS = ["is_capital_impaired"]
FEATURES = RATIO_COLS + FLAG_COLS    # 총 20 피처


def main():
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # 1) 라벨 로드
    labels = {}        # corp_code -> (label, label_year or None)
    for cc, lab, ldate in cur.execute(
            "SELECT corp_code, label, label_date FROM labels"):
        ly = int(ldate[:4]) if ldate else None
        labels[cc] = (lab, ly)

    pos_years = [ly for (lab, ly) in labels.values() if lab == 1 and ly]
    pos_years = np.array(pos_years)
    print(f"[1] 라벨: 양성 {sum(1 for v in labels.values() if v[0]==1):,} / "
          f"음성 {sum(1 for v in labels.values() if v[0]==0):,}")
    print(f"    양성 label_year 범위 {pos_years.min()}~{pos_years.max()}, "
          f"중앙 {int(np.median(pos_years))}")

    # 2) financials 로드 (금융업 제외)
    col_sql = ", ".join(FEATURES)
    rows = cur.execute(f"""
        SELECT corp_code, year, {col_sql}
        FROM financials
        WHERE data_quality != 'financial_sector'
        ORDER BY corp_code, year
    """).fetchall()

    # corp -> {year: [feat...]}
    from collections import defaultdict
    fin = defaultdict(dict)
    for r in rows:
        cc, yr = r[0], int(r[1])
        vec = [np.nan if v is None else float(v) for v in r[2:]]
        fin[cc][yr] = vec
    print(f"[2] financials 로드: {len(fin):,} 기업 (금융업 제외)")

    # 3) 각 기업 관측연도(obs_year) 결정 + 시퀀스 추출
    #    obs_year = 그 기업 데이터의 "마지막으로 보는 연도"(이 해까지 관측, 라벨은 미래)
    samples = []   # dict(cc, label, obs_year, seq[list of (year,vec)])
    n_drop_short = 0
    n_drop_nohist = 0

    for cc, (lab, ly) in labels.items():
        if cc not in fin:
            continue
        yrs_all = sorted(fin[cc].keys())

        if lab == 1:
            if ly is None:
                continue
            obs_year = ly - 1                      # label_year 이전까지만
        else:
            # 음성: pseudo obs_year 를 양성 분포에서 샘플 -> 그 이전까지만
            pseudo_ly = int(rng.choice(pos_years))
            obs_year = pseudo_ly - 1

        # obs_year 이전(포함) 연도만, 최근 MAX_T 개
        use_yrs = [y for y in yrs_all if y <= obs_year]
        if len(use_yrs) == 0:
            n_drop_nohist += 1
            continue
        use_yrs = use_yrs[-MAX_T:]
        if len(use_yrs) < MIN_T:
            n_drop_short += 1
            continue

        seq = [(y, fin[cc][y]) for y in use_yrs]
        samples.append({"cc": cc, "label": lab,
                        "obs_year": use_yrs[-1], "seq": seq})

    n_pos = sum(1 for s in samples if s["label"] == 1)
    n_neg = sum(1 for s in samples if s["label"] == 0)
    print(f"[3] 시퀀스 생성: {len(samples):,} "
          f"(양성 {n_pos:,} / 음성 {n_neg:,})")
    print(f"    제외: 시점부족 {n_drop_short:,} / 관측이력없음 {n_drop_nohist:,}")

    # 4) temporal split (obs_year 기준) + firm-level (기업 단위라 자동 보장)
    train_s = [s for s in samples if s["obs_year"] < TEMPORAL_SPLIT_YEAR]
    test_s = [s for s in samples if s["obs_year"] >= TEMPORAL_SPLIT_YEAR]
    print(f"[4] split: train {len(train_s):,} "
          f"(양성 {sum(1 for s in train_s if s['label']==1):,}) / "
          f"test {len(test_s):,} "
          f"(양성 {sum(1 for s in test_s if s['label']==1):,})")

    # 5) 표준화: train 시퀀스의 모든 (시점)값으로만 fit (z_score 컬럼 등)
    #    flag 컬럼(is_capital_impaired)은 표준화 제외
    n_ratio = len(RATIO_COLS)
    stacked = []
    for s in train_s:
        for (_, vec) in s["seq"]:
            stacked.append(vec[:n_ratio])
    stacked = np.array(stacked, dtype=float)
    # winsorize + mean/std (nan 무시)
    lo = np.nanpercentile(stacked, 1, axis=0)
    hi = np.nanpercentile(stacked, 99, axis=0)
    stacked = np.clip(stacked, lo, hi)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    std[std == 0] = 1.0
    print(f"[5] 표준화 fit (train only): {stacked.shape[0]:,} 시점-행")

    def to_tensor(sample_list):
        N = len(sample_list)
        D = len(FEATURES)
        X = np.zeros((N, MAX_T, D), dtype=np.float32)
        M = np.zeros((N, MAX_T), dtype=np.float32)   # 1=유효, 0=패딩
        y = np.zeros(N, dtype=np.float32)
        ccs = []
        for i, s in enumerate(sample_list):
            seq = s["seq"]
            t = len(seq)
            offset = MAX_T - t          # 좌측 패딩
            for k, (_, vec) in enumerate(seq):
                v = np.array(vec, dtype=float)
                # ratio 부분만 winsor+표준화
                rr = np.clip(v[:n_ratio], lo, hi)
                rr = (rr - mean) / std
                rr = np.nan_to_num(rr, nan=0.0)      # 결측 -> 0 (표준화 후 평균)
                ff = np.nan_to_num(v[n_ratio:], nan=0.0)
                X[i, offset + k, :] = np.concatenate([rr, ff])
                M[i, offset + k] = 1.0
            y[i] = s["label"]
            ccs.append(s["cc"])
        return X, M, y, np.array(ccs)

    Xtr, Mtr, ytr, cctr = to_tensor(train_s)
    Xte, Mte, yte, ccte = to_tensor(test_s)

    # firm-level 누수 최종 검증
    overlap = set(cctr) & set(ccte)
    assert len(overlap) == 0, f"firm 누수! 겹치는 기업 {len(overlap)}"
    print(f"[6] 텐서: train {Xtr.shape} / test {Xte.shape} | firm 겹침 0 확인")

    np.savez_compressed(
        OUT,
        X_train=Xtr, M_train=Mtr, y_train=ytr, cc_train=cctr,
        X_test=Xte, M_test=Mte, y_test=yte, cc_test=ccte,
        feature_names=np.array(FEATURES),
        scaler_mean=mean, scaler_std=std, winsor_lo=lo, winsor_hi=hi)
    print(f"[7] 저장: {OUT}")
    print(f"    양성 비율 train {ytr.mean()*100:.1f}% / test {yte.mean()*100:.1f}%")
    print("\n완료. 다음: 10_train_compare.py (LSTM/CNN+LSTM/TFT + z_score baseline)")

    con.close()


if __name__ == "__main__":
    main()