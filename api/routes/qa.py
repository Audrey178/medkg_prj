import asyncio
import logging
from collections import deque
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..middleware.auth import verify_api_key
from ..schemas.models import (
    BatchRequest, BatchResponse, BatchResultItem,
    QueryRequest, QueryResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_pipeline(request: Request):
    return request.app.state.pipeline


def _get_rate_limiter(request: Request):
    return request.app.state.rate_limiter


def _get_stats(request: Request):
    return request.app.state.stats


class RateLimiter:
    def __init__(self, max_per_hour: int):
        self._max = max_per_hour
        self._timestamps: deque = deque()

    def check_and_record(self) -> bool:
        import time
        now = time.time()
        cutoff = now - 3600
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max:
            return False
        self._timestamps.append(now)
        return True

    @property
    def count_last_hour(self) -> int:
        import time
        cutoff = time.time() - 3600
        return sum(1 for t in self._timestamps if t >= cutoff)


@router.get("/health")
async def health(request: Request):
    pipeline = _get_pipeline(request)
    neo4j_ok = False
    faiss_ok = False
    try:
        from agents.qa_inference.nodes.entity_node import _neo4j_driver, _faiss_cache
        neo4j_ok = _neo4j_driver is not None
        faiss_ok = _faiss_cache is not None
    except Exception:
        pass
    return {
        "status": "ok",
        "neo4j": "connected" if neo4j_ok else "not_initialized",
        "faiss": "loaded" if faiss_ok else "not_loaded",
        "pipeline": "ready" if pipeline is not None else "error",
    }


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    request: Request,
    _key: Annotated[str, Depends(verify_api_key)],
):
    limiter: RateLimiter = _get_rate_limiter(request)
    if not limiter.check_and_record():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
        )

    pipeline = _get_pipeline(request)
    stats = _get_stats(request)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pipeline.run(
            req.query,
            benchmark_type=req.benchmark_type,
            mode=req.mode,
            options=req.options,
        ),
    )

    stats["total_requests"] += 1
    stats["total_tokens"] += result.get("tokens_used", 0)
    stats["total_latency_ms"] += result.get("latency_ms", 0)
    if result.get("kg_coverage"):
        stats["kg_hits"] += 1

    logger.info(
        "query | lang=%s | type=%s | kg=%s | latency=%.0fms | tokens=%d",
        result.get("lang_detected", "?"),
        result.get("question_type", "?"),
        result.get("kg_coverage", False),
        result.get("latency_ms", 0),
        result.get("tokens_used", 0),
    )

    return QueryResponse(**result)


@router.post("/batch", response_model=BatchResponse)
async def batch(
    req: BatchRequest,
    request: Request,
    _key: Annotated[str, Depends(verify_api_key)],
):
    limiter: RateLimiter = _get_rate_limiter(request)
    pipeline = _get_pipeline(request)
    stats = _get_stats(request)

    results: list[BatchResultItem] = []
    kg_hits = 0

    for item in req.queries:
        if not limiter.check_and_record():
            results.append(BatchResultItem(id=item.id, result=None, error="rate_limit_exceeded"))
            continue
        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda i=item: pipeline.run(
                    i.query,
                    benchmark_type=i.benchmark_type,
                    mode=i.mode,
                    options=i.options,
                ),
            )
            stats["total_requests"] += 1
            stats["total_tokens"] += raw.get("tokens_used", 0)
            stats["total_latency_ms"] += raw.get("latency_ms", 0)
            if raw.get("kg_coverage"):
                stats["kg_hits"] += 1
                kg_hits += 1
            results.append(BatchResultItem(id=item.id, result=QueryResponse(**raw)))
        except Exception as exc:
            logger.warning("Batch item %s failed: %s", item.id, exc)
            results.append(BatchResultItem(id=item.id, result=None, error=str(exc)))

    success = sum(1 for r in results if r.result is not None)
    return BatchResponse(
        results=results,
        summary={
            "total": len(req.queries),
            "success": success,
            "failed": len(req.queries) - success,
            "kg_hit_rate": kg_hits / len(req.queries) if req.queries else 0,
        },
    )
