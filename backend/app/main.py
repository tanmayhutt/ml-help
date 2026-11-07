from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, db, storage
from .api import router
from .runner import get_runner


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    storage.ensure_dirs()
    get_runner()
    yield


app = FastAPI(title="ML Help", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'; object-src 'none'"
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    else:
        # Pages and static files: browsers and the CDN revalidate on every load (ETag), so a deploy
        # never leaves users with a new page and old scripts or styles.
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    return JSONResponse({"detail": f"Server error: {type(exc).__name__}"}, status_code=500)


STATIC = config.STATIC_DIR


def _asset_version(folder: Path) -> str:
    """Short hash of every static file, so each deploy gets a new asset path and no CDN or browser
    can keep serving the previous scripts or styles."""
    h = hashlib.sha256()
    for f in sorted(folder.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(folder)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


if STATIC.exists():
    ASSET_VERSION = _asset_version(STATIC / "static")
    ASSET_PREFIX = f"/a/{ASSET_VERSION}"
    INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8").replace("/static/", ASSET_PREFIX + "/")
    app.mount(ASSET_PREFIX, StaticFiles(directory=STATIC / "static"), name="assets")
    app.mount("/static", StaticFiles(directory=STATIC / "static"), name="static")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        if path.startswith(("api/", "static/", "a/")):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        return HTMLResponse(INDEX_HTML)
