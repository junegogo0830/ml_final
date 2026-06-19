# -*- coding: utf-8 -*-
"""
새 버전 학습 파이프라인 (DART 재무제표 수집 → 모델 재학습)
실행: python scripts/refresh_pipeline.py  (프로젝트 루트에서)

backend/app.py의 "버전 관리" 탭 "새 버전 학습" 버튼이 백그라운드로 호출.
진행 상황은 models/saved/pipeline_status.json 에 기록되며
각 단계 스크립트(dart_fetch_latest.py, retrain_lstm.py)가 세부 진행률을 갱신한다.
"""

import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT        = Path(__file__).resolve().parent
PROJ_ROOT   = ROOT.parent
STATUS_PATH = PROJ_ROOT / "models" / "saved" / "pipeline_status.json"


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


def run_step(script_name, step_name, message):
    write_status(running=True, step=step_name, message=message, error=None)
    proc = subprocess.run([sys.executable, str(ROOT / script_name)], cwd=str(PROJ_ROOT))
    if proc.returncode != 0:
        write_status(running=False, step="error",
                      message=f"{step_name} 단계 실패 (exit code {proc.returncode})")
        sys.exit(proc.returncode)


def main():
    write_status(running=True, step="fetch", message="DART API 재무제표 수집 시작",
                  started_at=datetime.now().isoformat(), error=None, result=None)
    run_step("dart_fetch_latest.py", "fetch", "DART API에서 최신 재무제표 수집 중...")
    run_step("retrain_lstm.py", "retrain", "모델 재학습 중...")
    write_status(running=False, step="done", message="새 버전 학습이 완료되었습니다.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_status(running=False, step="error", message=traceback.format_exc()[:2000])
        raise
