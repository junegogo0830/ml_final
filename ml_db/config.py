"""
ml_db 전역 설정.
DART API 키는 환경변수 DART_API_KEY로 받는 게 안전. 코드에 박지 마.

set DART_API_KEY=xxxxxxxx     (Windows cmd)
$env:DART_API_KEY="xxxxxxxx"  (PowerShell)
"""

import os
from pathlib import Path

# ===== 경로 =====
ROOT = Path(__file__).resolve().parent
DB_DIR = ROOT / "db"
DB_PATH = DB_DIR / "dart_v2.db"
LOG_DIR = ROOT / "logs"
CACHE_DIR = ROOT / "cache"  # corp_code.zip 등 임시 파일

DB_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ===== DART API =====
DART_API_KEY = os.environ.get("DART_API_KEY", "")
if not DART_API_KEY:
    raise RuntimeError(
        "환경변수 DART_API_KEY가 설정되지 않았습니다.\n"
        "  set DART_API_KEY=발급받은_40자리_키       (Windows cmd)\n"
        "  $env:DART_API_KEY = \"발급받은_40자리_키\"  (PowerShell)"
    )

DART_BASE = "https://opendart.fss.or.kr/api"

# ===== 수집 범위 =====
YEAR_FROM = 2015
YEAR_TO = 2025

# ===== Rate Limit =====
# DART 공식: 분당 1000회, 일 20,000회 (개인키)
# 안전 마진 두고 분당 600회 = 100ms 간격
REQUEST_INTERVAL_SEC = 0.10
DAILY_LIMIT = 20000

# ===== 재시도 =====
MAX_RETRIES = 3
RETRY_WAIT_SEC = 5

# ===== HTTP =====
REQUEST_TIMEOUT = 30  # 초
USER_AGENT = "ml_db-bankruptcy-research/1.0"
