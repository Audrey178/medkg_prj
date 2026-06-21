"""
FastAPI application for ChronoMedKG KG-RAG QA inference.

Start:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Auth: include  X-API-Key: <QA_API_KEY>  header on every request.
"""

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before importing pipeline
_env = PROJECT_ROOT / ".env"
if _env.exists():
    for _line in open(_env):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _v.strip():
                os.environ.setdefault(_k.strip(), _v.strip())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.qa_inference import QAPipeline
from agents.qa_inference.utils.config import get_config
from api.routes.qa import RateLimiter, router as qa_router
from api.routes.admin import router as admin_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("api")

app = FastAPI(
    title="ChronoMedKG QA API",
    description="KG-RAG inference over ChronoMedKG biomedical knowledge graph.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa_router, prefix="", tags=["QA"])
app.include_router(admin_router, prefix="", tags=["Admin"])


@app.on_event("startup")
async def startup():
    cfg = get_config()
    app.state.pipeline = QAPipeline(config=cfg)
    app.state.rate_limiter = RateLimiter(
        max_per_hour=cfg.get("rate_limit", {}).get("max_requests_per_hour", 500)
    )
    app.state.stats = {
        "total_requests": 0,
        "total_tokens": 0,
        "total_latency_ms": 0.0,
        "kg_hits": 0,
    }
    logger.info("QAPipeline ready")
