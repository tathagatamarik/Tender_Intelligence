"""Optional SQLite persistence for human-in-the-loop approval requests and saved scenarios.

Nothing here executes a business action. Records only capture that a user submitted a
recommendation for approval by the accountable function.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "demo_store.sqlite"


def _conn(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE IF NOT EXISTS approval_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, module TEXT, item_ref TEXT,
        summary TEXT, approver_role TEXT, status TEXT, payload TEXT)""")
    return con


def submit_for_approval(module: str, item_ref: str, summary: str, approver_role: str, payload: dict | None = None,
                        path: Path = DB_PATH) -> int:
    with _conn(path) as con:
        cur = con.execute(
            "INSERT INTO approval_requests (created_at, module, item_ref, summary, approver_role, status, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), module, item_ref, summary[:500], approver_role,
             "PENDING APPROVAL", json.dumps(payload or {}, default=str)[:5000]))
        return int(cur.lastrowid)


def list_requests(path: Path = DB_PATH) -> pd.DataFrame:
    try:
        with _conn(path) as con:
            return pd.read_sql_query("SELECT id, created_at, module, item_ref, summary, approver_role, status "
                                     "FROM approval_requests ORDER BY id DESC", con)
    except sqlite3.Error:
        return pd.DataFrame(columns=["id", "created_at", "module", "item_ref", "summary", "approver_role", "status"])


def clear_requests(path: Path = DB_PATH) -> None:
    with _conn(path) as con:
        con.execute("DELETE FROM approval_requests")
