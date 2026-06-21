from typing import Annotated
from fastapi import APIRouter, Depends, Request

from ..middleware.auth import verify_api_key

router = APIRouter()


@router.get("/stats")
async def stats(
    request: Request,
    _key: Annotated[str, Depends(verify_api_key)],
):
    s = request.app.state.stats
    limiter = request.app.state.rate_limiter
    total = s["total_requests"]
    return {
        "requests_last_hour": limiter.count_last_hour,
        "total_requests": total,
        "avg_latency_ms": round(s["total_latency_ms"] / total, 1) if total else 0,
        "avg_tokens": round(s["total_tokens"] / total, 1) if total else 0,
        "kg_hit_rate": round(s["kg_hits"] / total, 3) if total else 0,
    }
