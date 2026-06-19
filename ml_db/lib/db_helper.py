"""SQLite 헬퍼. 연결 관리 + 진행 로그."""

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


@contextmanager
def connect(db_path: Path = None):
    """with connect() as conn: ... 형태로 사용."""
    path = db_path or config.DB_PATH
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def log_step(step: str, target: str, status: str, error: str = None):
    """collection_log_v2에 진행상태 기록."""
    with connect() as conn:
        conn.execute(
            """INSERT INTO collection_log_v2
               (step, target, status, started_at, completed_at, error)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                step,
                target,
                status,
                datetime.now().isoformat(),
                datetime.now().isoformat() if status in ("done", "failed") else None,
                error,
            ),
        )


def is_done(step: str, target: str) -> bool:
    """이미 수집 완료된 작업인지 체크 (재시작 지원)."""
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM collection_log_v2 WHERE step=? AND target=? AND status='done' LIMIT 1",
            (step, target),
        ).fetchone()
        return row is not None


def get_pending_targets(step: str, all_targets: list) -> list:
    """완료된 target 제외하고 남은 것만 반환."""
    with connect() as conn:
        done = set(
            r[0]
            for r in conn.execute(
                "SELECT target FROM collection_log_v2 WHERE step=? AND status='done'",
                (step,),
            )
        )
    return [t for t in all_targets if t not in done]
