"""Tiny SQLite layer. One file, WAL mode, no ORM."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from . import config

_lock = threading.Lock()
_DB_PATH = config.DATA_DIR / "mlhelp.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  rows INTEGER NOT NULL,
  cols INTEGER NOT NULL,
  bytes INTEGER NOT NULL,
  sample INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL,
  profile TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  dataset_id TEXT,
  params TEXT NOT NULL,
  status TEXT NOT NULL,
  progress TEXT,
  result TEXT,
  error TEXT,
  created REAL NOT NULL,
  started REAL,
  finished REAL
);
CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  name TEXT NOT NULL,
  task TEXT NOT NULL,
  target TEXT,
  features TEXT NOT NULL,
  metrics TEXT,
  created REAL NOT NULL
);
"""


def init() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(_DB_PATH, timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        with _lock:
            yield con
            con.commit()
    finally:
        con.close()


def row_to_dict(row: sqlite3.Row | None, json_fields: tuple[str, ...] = ()) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for f in json_fields:
        if d.get(f):
            try:
                d[f] = json.loads(d[f])
            except (TypeError, ValueError):
                pass
    return d


def now() -> float:
    return time.time()
