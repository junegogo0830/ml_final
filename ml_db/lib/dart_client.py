"""
DART OpenAPI 호출 래퍼.

- Rate limit 자동 준수 (요청 간격 보장)
- 에러 자동 분류: 정상(000) / 데이터없음(013) / rate limit(020) / 기타
- 재시도 로직 (지수 백오프)
- 일일 호출 카운터
"""

import time
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any

import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ===== 로깅 =====
def _setup_logger():
    log_file = config.LOG_DIR / f"dart_{date.today().isoformat()}.log"
    logger = logging.getLogger("dart")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = _setup_logger()


# ===== 호출 카운터 =====
class CallCounter:
    """일일 호출 횟수 추적. 파일에 저장해서 재시작 시에도 유지."""

    def __init__(self):
        self.counter_file = config.LOG_DIR / f"calls_{date.today().isoformat()}.json"
        self.load()

    def load(self):
        if self.counter_file.exists():
            with open(self.counter_file, "r") as f:
                data = json.load(f)
            self.count = data.get("count", 0)
        else:
            self.count = 0

    def save(self):
        with open(self.counter_file, "w") as f:
            json.dump({"count": self.count, "updated": datetime.now().isoformat()}, f)

    def increment(self):
        self.count += 1
        if self.count % 100 == 0:
            self.save()
            log.info(f"오늘 호출 횟수: {self.count} / {config.DAILY_LIMIT}")
        if self.count >= config.DAILY_LIMIT:
            log.error(f"일일 한도 도달: {self.count}")
            raise DartQuotaExceeded(self.count)


counter = CallCounter()


# ===== 메인 호출 함수 =====
class DartAPIError(Exception):
    """DART API 응답 status가 정상(000)이 아닌 경우."""

    def __init__(self, status, message):
        self.status = status
        self.message = message
        super().__init__(f"[{status}] {message}")


class DartNoData(DartAPIError):
    """status 013: 조회된 데이터 없음. 정상적인 빈 응답."""

    pass


class DartQuotaExceeded(DartAPIError):
    """일일 호출 한도 초과."""

    def __init__(self, count: int = 0):
        super().__init__("QUOTA", f"일일 한도 도달: {count}회")
        self.count = count


_last_request_time = 0.0


def _wait_for_rate_limit():
    """요청 간격 보장."""
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time
    if elapsed < config.REQUEST_INTERVAL_SEC:
        time.sleep(config.REQUEST_INTERVAL_SEC - elapsed)
    _last_request_time = time.monotonic()


def call(endpoint: str, params: dict, response_type: str = "json") -> dict | bytes:
    """
    DART API 단일 호출.

    endpoint: 'list', 'fnlttSinglAcntAll', 'dfOcr', 'crpRrhsNcsr', 'corpCode' 등
    params: crtfc_key는 자동 추가됨
    response_type: 'json' (기본) | 'xml' | 'binary' (corp_code zip용)

    return: dict (JSON 응답의 list/result 부분) | bytes (binary)
    raise: DartNoData (013), DartAPIError (기타 에러)
    """
    url = f"{config.DART_BASE}/{endpoint}.{ 'xml' if response_type == 'xml' else 'json' }"
    if response_type == "binary":
        # corp_code 같은 zip 다운로드는 .xml 엔드포인트 사용
        url = f"{config.DART_BASE}/{endpoint}.xml"

    full_params = {"crtfc_key": config.DART_API_KEY, **params}

    last_err = None
    for attempt in range(config.MAX_RETRIES):
        _wait_for_rate_limit()
        counter.increment()
        try:
            resp = requests.get(
                url,
                params=full_params,
                timeout=config.REQUEST_TIMEOUT,
                headers={"User-Agent": config.USER_AGENT},
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            last_err = e
            log.warning(f"네트워크 에러 (attempt {attempt+1}/{config.MAX_RETRIES}): {e}")
            time.sleep(config.RETRY_WAIT_SEC * (2 ** attempt))
            continue

        if response_type == "binary":
            return resp.content

        try:
            data = resp.json()
        except json.JSONDecodeError:
            last_err = "JSON 파싱 실패"
            log.warning(f"JSON 파싱 실패 (attempt {attempt+1}): {resp.text[:200]}")
            time.sleep(config.RETRY_WAIT_SEC)
            continue

        status = data.get("status")
        message = data.get("message", "")

        if status == "000":
            return data

        if status == "013":
            raise DartNoData(status, message)

        if status == "020":
            log.error(f"Rate limit 초과 (status 020): {message}")
            log.error("60초 대기 후 재시도...")
            time.sleep(60)
            last_err = DartAPIError(status, message)
            continue

        if status in ("010", "011", "012", "101"):
            # 인증/접근 에러: 재시도 무의미
            raise DartAPIError(status, message)

        if status == "800":
            log.warning("DART 시스템 점검 중. 5분 대기...")
            time.sleep(300)
            last_err = DartAPIError(status, message)
            continue

        # 기타 에러: 재시도
        last_err = DartAPIError(status, message)
        log.warning(f"DART 에러 (attempt {attempt+1}): {last_err}")
        time.sleep(config.RETRY_WAIT_SEC * (2 ** attempt))

    raise last_err if isinstance(last_err, Exception) else DartAPIError("???", str(last_err))


# ===== 헬퍼 =====
def get_list_rows(data: dict) -> list:
    """API 응답에서 list 부분만 안전하게 꺼냄."""
    if not isinstance(data, dict):
        return []
    return data.get("list", []) or []
