from typing import Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    benchmark_type: str = Field("bioasq", pattern="^(bioasq|medqa|pubmedqa)$")
    mode: str = Field("kg_rag", pattern="^(kg_rag|llm_only|kg_only)$")
    options: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: Optional[Any]
    question_type: str
    sources: list[str]
    kg_coverage: bool
    matched_entities: list[str]
    lang_detected: str
    latency_ms: float
    tokens_used: int
    error: Optional[str]


class BatchItem(BaseModel):
    id: str
    query: str
    benchmark_type: str = "bioasq"
    mode: str = "kg_rag"
    options: dict = Field(default_factory=dict)


class BatchRequest(BaseModel):
    queries: list[BatchItem]


class BatchResultItem(BaseModel):
    id: str
    result: Optional[QueryResponse]
    error: Optional[str] = None


class BatchResponse(BaseModel):
    results: list[BatchResultItem]
    summary: dict
