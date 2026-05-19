# fix_paths.py
# 사용법: ml 폴더(프로젝트 루트)에서 실행
#   미리보기: python fix_paths.py --dry-run
#   실제 수정: python fix_paths.py

import os
import re
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

# 데이터/DB 경로 매핑 (옛 → 새)
PATH_MAPPING = {
    "train_original.csv":       "data/raw/train_original.csv",
    "test.csv":                 "data/raw/test.csv",
    "train_augmented.csv":      "data/processed/train_augmented.csv",
    "financials_with_news.csv": "data/processed/financials_with_news.csv",
    "lstm_predictions.csv":     "data/predictions/lstm_predictions.csv",
    "cnn_lstm_predictions.csv": "data/predictions/cnn_lstm_predictions.csv",
    "tft_predictions.csv":      "data/predictions/tft_predictions.csv",
    "bankruptcy_prediction.db": "db/bankruptcy_prediction.db",
}

# import 매핑 (실제 폴더 구조 기준)
IMPORT_MAPPING = {
    "collect_2024_2025":    "src.data_collection.collect_2024_2025",
    "collect_companies":    "src.data_collection.collect_companies",
    "collect_financial":    "src.data_collection.collect_financial",
    "crawl_news":           "src.data_collection.crawl_news",
    "calc_finalcials":      "src.data_collection.calc_finalcials",
    "Setup_db":             "src.database.Setup_db",
    "start_db":             "src.database.start_db",
    "build_relationship":   "src.database.build_relationship",
    "Insert_dataquery":     "src.database.Insert_dataquery",
    "lstm":                 "src.models.lstm",
    "lstm_model":           "src.models.lstm_model",
    "lstm_cnn_model":       "src.models.lstm_cnn_model",
    "tft_model":            "src.models.tft_model",
    "all_model":            "src.models.all_model",
    "kobert_sentiment":     "src.models.kobert_sentiment",
    "split_augment":        "src.models.split_augment",
    "predictor":            "src.prediction.predictor",
    "Dash_board":           "src.dashboard.Dash_board",
}

# 제외할 경로
EXCLUDE_DIRS = {"__pycache__", "saved_models", "data", "db", ".git", ".venv", "venv"}
EXCLUDE_FILES = {"fix_paths.py", "organize.ps1"}

def fix_file(filepath):
    """단일 .py 파일의 경로/import 수정"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            original = f.read()
    except UnicodeDecodeError:
        # cp949로 재시도
        with open(filepath, "r", encoding="cp949") as f:
            original = f.read()
    
    content = original
    changes = []
    
    # 1. 문자열 경로 수정 (따옴표로 감싸진 것만)
    for old, new in PATH_MAPPING.items():
        for quote in ['"', "'"]:
            pat_old = f'{quote}{old}{quote}'
            pat_new = f'{quote}{new}{quote}'
            if pat_old in content:
                count = content.count(pat_old)
                content = content.replace(pat_old, pat_new)
                changes.append(f"  경로 x{count}: {old} → {new}")
            
            # ./X.csv 형태도 처리
            pat_old2 = f'{quote}./{old}{quote}'
            pat_new2 = f'{quote}{new}{quote}'
            if pat_old2 in content:
                count = content.count(pat_old2)
                content = content.replace(pat_old2, pat_new2)
                changes.append(f"  경로 x{count}: ./{old} → {new}")
    
    # 2. import 문 수정 (긴 이름부터 처리해서 부분 매칭 방지)
    sorted_imports = sorted(IMPORT_MAPPING.items(), key=lambda x: -len(x[0]))
    
    for old, new in sorted_imports:
        # from X import Y
        pattern1 = rf'(^|\n)(\s*)from\s+{re.escape(old)}\s+import\b'
        matches = re.findall(pattern1, content)
        if matches:
            content = re.sub(pattern1, rf'\1\2from {new} import', content)
            changes.append(f"  import x{len(matches)}: from {old} → from {new}")
        
        # import X (단독)
        pattern2 = rf'(^|\n)(\s*)import\s+{re.escape(old)}(\s|$)'
        matches = re.findall(pattern2, content)
        if matches:
            content = re.sub(pattern2, rf'\1\2import {new}\3', content)
            changes.append(f"  import x{len(matches)}: import {old} → import {new}")
    
    if changes:
        print(f"\n[{filepath}]")
        for c in changes:
            print(c)
        
        if not DRY_RUN:
            # 백업
            backup = str(filepath) + ".bak"
            with open(backup, "w", encoding="utf-8") as f:
                f.write(original)
            # 저장
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  → 저장 완료 (백업: {backup})")
        else:
            print("  [DryRun - 실제 수정 안 함]")
        return True
    return False

def main():
    if DRY_RUN:
        print("=== DRY RUN 모드 (실제 수정 안 함) ===\n")
    else:
        print("=== 실제 수정 모드 ===")
        print("각 파일의 백업이 .bak로 저장됩니다.\n")
    
    # 모든 .py 파일 스캔 (제외 폴더 제외)
    py_files = []
    for f in Path(".").rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in f.parts):
            continue
        if f.name in EXCLUDE_FILES:
            continue
        if f.suffix == ".bak":
            continue
        py_files.append(f)
    
    print(f"검사할 파일: {len(py_files)}개")
    for f in py_files:
        print(f"  - {f}")
    print()
    
    modified_count = 0
    for f in py_files:
        if fix_file(f):
            modified_count += 1
    
    print(f"\n{'='*50}")
    print(f"수정된 파일: {modified_count}개 / 전체 {len(py_files)}개")
    if DRY_RUN:
        print("실제 수정을 원하면: python fix_paths.py")

if __name__ == "__main__":
    main()