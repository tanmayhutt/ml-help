"""In-memory sliding-window rate limiter keyed by client IP. No external service."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from . import config

_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check(request: Request, bucket: str, per_minute: int) -> None:
    key = f"{bucket}:{client_ip(request)}"
    now = time.time()
    with _lock:
        d = _hits[key]
        while d and d[0] < now - 60:
            d.popleft()
        if len(d) >= per_minute:
            raise HTTPException(429, f"Too many {bucket} requests. Limit is {per_minute} per minute.")
        d.append(now)
        if len(_hits) > 5000:
            for k in [k for k, v in _hits.items() if not v or v[-1] < now - 120]:
                del _hits[k]


def require_token(request: Request) -> None:
    if config.ACCESS_TOKEN and request.headers.get("x-access-token") != config.ACCESS_TOKEN:
        raise HTTPException(401, "Access token required.")
