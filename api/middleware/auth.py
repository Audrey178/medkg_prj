import os
from fastapi import Header, HTTPException, status


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    expected = os.environ.get("OPENAI_API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY not configured on server.",
        )
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return x_api_key
