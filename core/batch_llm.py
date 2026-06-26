"""
Batch LLM Client for ChronoMedKG (v2)
=========================================
Submits extraction prompts to OpenAI, Anthropic, and Gemini batch APIs.
50% cost reduction + no rate limits vs real-time calls.

v2 improvements over v1:
- Auto-chunking at provider limits (50K OpenAI, 10K Anthropic)
- Cost tracking per batch
- Retry with exponential backoff on submission failures
- Collision-safe results keyed by {provider: {custom_id: result}}
- Checkpoint-friendly: batch_ids now {provider: [batch_id, ...]} for multi-batch
- Progress callback during polling
- Robust error result parsing (captures error details from failed requests)
 
Usage:
    client = BatchLLMClient()
    batch_ids = client.submit(prompts, disease_id="MONDO_1234")
    statuses = client.poll(batch_ids)
    results = client.retrieve(batch_ids)
    print(client.cost_summary)
"""

from __future__ import annotations

import json
import uuid
import logging
import hashlib
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider limits
# ---------------------------------------------------------------------------
OPENAI_MAX_REQUESTS_PER_BATCH = 1_500   # 1500 × ~1K tokens = 1.5M, fits under 2M enqueued limit
ANTHROPIC_MAX_REQUESTS_PER_BATCH = 10_000

# ---------------------------------------------------------------------------
# Cost per 1K tokens (input/output) — batch pricing (50% of real-time)
# ---------------------------------------------------------------------------
COST_PER_1K = {
    "gpt-4o-mini": {"input": 0.000075, "output": 0.0003},       # $0.075/$0.30 per 1M
    "claude-3-haiku-20240307": {"input": 0.000125, "output": 0.000625},  # $0.125/$0.625 per 1M
}


@dataclass
class BatchResult:
    """Result from a single batch request."""
    custom_id: str
    provider: str
    model: str
    triples: list[dict]
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


@dataclass
class BatchCostTracker:
    """Tracks costs across batch submissions."""
    total_input_tokens: dict[str, int] = field(default_factory=lambda: {"openai": 0, "anthropic": 0})
    total_output_tokens: dict[str, int] = field(default_factory=lambda: {"openai": 0, "anthropic": 0})
    total_requests: dict[str, int] = field(default_factory=lambda: {"openai": 0, "anthropic": 0})
    total_batches: dict[str, int] = field(default_factory=lambda: {"openai": 0, "anthropic": 0})

    def add(self, provider: str, input_tokens: int, output_tokens: int):
        self.total_input_tokens[provider] = self.total_input_tokens.get(provider, 0) + input_tokens
        self.total_output_tokens[provider] = self.total_output_tokens.get(provider, 0) + output_tokens
        self.total_requests[provider] = self.total_requests.get(provider, 0) + 1

    @property
    def estimated_cost(self) -> dict[str, float]:
        costs = {}
        for provider, model in [("openai", "gpt-4o-mini"), ("anthropic", "claude-3-haiku-20240307")]:
            if provider in self.total_input_tokens:
                rates = COST_PER_1K.get(model, {"input": 0, "output": 0})
                inp = self.total_input_tokens.get(provider, 0) / 1000 * rates["input"]
                out = self.total_output_tokens.get(provider, 0) / 1000 * rates["output"]
                costs[provider] = round(inp + out, 4)
        costs["total"] = round(sum(costs.values()), 4)
        return costs

    def summary(self) -> str:
        costs = self.estimated_cost
        lines = ["=== Batch Cost Summary ==="]
        for provider in ("openai", "anthropic"):
            reqs = self.total_requests.get(provider, 0)
            batches = self.total_batches.get(provider, 0)
            inp = self.total_input_tokens.get(provider, 0)
            out = self.total_output_tokens.get(provider, 0)
            cost = costs.get(provider, 0)
            if reqs > 0:
                lines.append(
                    f"  {provider}: {reqs:,} requests in {batches} batches, "
                    f"{inp:,} in / {out:,} out tokens, ~${cost:.4f}"
                )
        lines.append(f"  TOTAL: ~${costs.get('total', 0):.4f}")
        return "\n".join(lines)


class BatchLLMClient:
    """Submit and retrieve batch LLM extractions via provider batch APIs.

    v2: auto-chunking, cost tracking, retry, collision-safe results.
    """

    def __init__(self):
        self._openai_client = None
        self._anthropic_client = None
        self.cost_tracker = BatchCostTracker()
        self._init_clients()

    def _init_clients(self):
        """Initialize API clients for batch submission."""
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            try:
                import openai
                self._openai_client = openai.OpenAI(api_key=openai_key)
                logger.info("OpenAI batch client initialized")
            except ImportError:
                logger.warning("openai package not installed")

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                logger.info("Anthropic batch client initialized")
            except ImportError:
                logger.warning("anthropic package not installed")

    @property
    def available_providers(self) -> list[str]:
        providers = []
        if self._openai_client:
            providers.append("openai")
        if self._anthropic_client:
            providers.append("anthropic")
        return providers

    @property
    def cost_summary(self) -> str:
        return self.cost_tracker.summary()

    # -----------------------------------------------------------------------
    # Submit
    # -----------------------------------------------------------------------

    def submit(
        self,
        prompts: list[str],
        disease_id: str,
        doc_ids: list[str] | None = None,
        max_retries: int = 3,
        providers: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """
        Submit prompts to batch APIs with auto-chunking.

        Args:
            prompts: List of extraction prompts (one per document)
            disease_id: Disease/chunk identifier for custom_id encoding
            doc_ids: Optional document identifiers for tracking
            max_retries: Max retry attempts per provider on failure
            providers: Which providers to submit to. Default: all available.
                       e.g. ["anthropic"] to skip OpenAI.

        Returns:
            Dict of {provider: [batch_id, ...]}  — list because large prompt
            sets may be split into multiple batches per provider.
        """
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(prompts))]

        batch_ids: dict[str, list[str]] = {}
        use_providers = providers or self.available_providers

        # Submit Anthropic FIRST (reliable), then OpenAI (has token enqueue limits)
        if self._anthropic_client and "anthropic" in use_providers:
            bids = self._submit_with_retry(
                self._submit_anthropic_chunk, prompts, disease_id, doc_ids,
                chunk_size=ANTHROPIC_MAX_REQUESTS_PER_BATCH,
                provider="anthropic",
                max_retries=max_retries,
            )
            if bids:
                batch_ids["anthropic"] = bids
                self.cost_tracker.total_batches["anthropic"] = (
                    self.cost_tracker.total_batches.get("anthropic", 0) + len(bids)
                )
                logger.info("Anthropic: %d batch(es) submitted for %d prompts", len(bids), len(prompts))

        if self._openai_client and "openai" in use_providers:
            bids = self._submit_with_retry(
                self._submit_openai_chunk, prompts, disease_id, doc_ids,
                chunk_size=OPENAI_MAX_REQUESTS_PER_BATCH,
                provider="openai",
                max_retries=max_retries,
            )
            if bids:
                batch_ids["openai"] = bids
                self.cost_tracker.total_batches["openai"] = (
                    self.cost_tracker.total_batches.get("openai", 0) + len(bids)
                )
                logger.info("OpenAI: %d batch(es) submitted for %d prompts", len(bids), len(prompts))

        return batch_ids

    def _submit_with_retry(
        self,
        submit_fn: Callable,
        prompts: list[str],
        disease_id: str,
        doc_ids: list[str],
        chunk_size: int,
        provider: str,
        max_retries: int = 3,
    ) -> list[str]:
        """Submit with auto-chunking and retry with exponential backoff."""
        batch_ids = []

        # Split into chunks if needed
        chunks = []
        for i in range(0, len(prompts), chunk_size):
            chunks.append((
                prompts[i:i + chunk_size],
                doc_ids[i:i + chunk_size],
                i,  # offset for chunk tracking
            ))

        if len(chunks) > 1:
            logger.info("%s: splitting %d prompts into %d batches (limit %d/batch)",
                        provider, len(prompts), len(chunks), chunk_size)

        for chunk_idx, (chunk_prompts, chunk_doc_ids, offset) in enumerate(chunks):
            chunk_disease_id = disease_id if len(chunks) == 1 else f"{disease_id}__part{chunk_idx}"

            # Stagger submissions to avoid enqueued token limits
            if chunk_idx > 0 and provider == "openai":
                logger.info("%s: waiting 90s between batch submissions to respect token enqueue limit...", provider)
                time.sleep(90)

            # For OpenAI, wait until enqueued tokens are below threshold before submitting
            if provider == "openai" and self._openai_client:
                self._wait_for_openai_token_capacity(
                    estimated_tokens=len(chunk_prompts) * 1000,  # ~1K tokens/prompt estimate
                    max_wait=600,
                )

            for attempt in range(1, max_retries + 1):
                try:
                    bid = submit_fn(chunk_prompts, chunk_disease_id, chunk_doc_ids)
                    if bid:
                        batch_ids.append(bid)
                        logger.info("%s: batch %d/%d submitted (%s, %d prompts)",
                                    provider, chunk_idx + 1, len(chunks), bid[:20], len(chunk_prompts))
                        break
                except Exception as e:
                    err_str = str(e)
                    # If we hit the enqueued token limit, wait longer and retry
                    if "enqueued token limit" in err_str.lower() or "enqueued_token" in err_str.lower():
                        wait = 180  # 3 minutes — let existing batches drain
                        logger.warning(
                            "%s: enqueued token limit hit, waiting %ds for batches to drain...",
                            provider, wait,
                        )
                        time.sleep(wait)
                    else:
                        wait = min(2 ** attempt * 5, 120)  # 10s, 20s, 40s... max 120s
                        logger.warning(
                            "%s batch submission attempt %d/%d failed: %s. Retrying in %ds...",
                            provider, attempt, max_retries, e, wait,
                        )
                        if attempt < max_retries:
                            time.sleep(wait)
                        else:
                            logger.error("%s batch submission FAILED after %d attempts", provider, max_retries)

        return batch_ids

    def _wait_for_openai_token_capacity(
        self,
        estimated_tokens: int,
        max_enqueued: int = 1_800_000,  # Stay under 2M limit with headroom
        max_wait: int = 600,
        poll_interval: int = 60,
    ):
        """Wait until OpenAI enqueued token count is low enough for a new batch.

        Checks all in-progress batches and sums their estimated enqueued tokens.
        If the total + estimated_tokens exceeds max_enqueued, waits for batches
        to complete before returning.
        """
        start = time.monotonic()
        while time.monotonic() - start < max_wait:
            try:
                # List recent batches and sum tokens from those still in progress
                enqueued_tokens = 0
                in_progress_count = 0
                batches = self._openai_client.batches.list(limit=20)
                for batch in batches.data:
                    if batch.status in ("validating", "in_progress", "finalizing"):
                        in_progress_count += 1
                        # Use metadata prompt_count * ~1K tokens as estimate
                        meta = batch.metadata or {}
                        prompt_count = int(meta.get("prompt_count", 0))
                        enqueued_tokens += prompt_count * 1000  # ~1K tokens per prompt

                available = max_enqueued - enqueued_tokens
                if available >= estimated_tokens:
                    if in_progress_count > 0:
                        logger.info(
                            "OpenAI token capacity OK: ~%dK enqueued, ~%dK available, "
                            "need ~%dK (%d batches in progress)",
                            enqueued_tokens // 1000, available // 1000,
                            estimated_tokens // 1000, in_progress_count,
                        )
                    return

                logger.info(
                    "OpenAI enqueued tokens ~%dK (limit %dK), need ~%dK. "
                    "Waiting %ds for %d batches to drain...",
                    enqueued_tokens // 1000, max_enqueued // 1000,
                    estimated_tokens // 1000, poll_interval, in_progress_count,
                )
                time.sleep(poll_interval)

            except Exception as e:
                logger.warning("Failed to check OpenAI batch capacity: %s. Proceeding anyway.", e)
                return

        logger.warning(
            "Timed out waiting for OpenAI token capacity after %ds. Submitting anyway.", max_wait
        )

    @staticmethod
    def _safe_id(s: str, max_len: int = 50) -> str:
        """Sanitize string for custom_id: alphanumeric, underscore, hyphen only.
        Max 64 chars per Anthropic spec; we use 50 per component to stay safe."""
        clean = re.sub(r'[^a-zA-Z0-9_-]', '_', s)
        return clean[:max_len]

    @staticmethod
    def _make_custom_id(disease_id: str, doc_id: str, suffix: str) -> str:
        """Build a custom_id guaranteed ≤64 chars total.
        Format: {disease}_{doc}_{suffix}  (suffix e.g. 'gpt4omini', 'claudehaiku')
        """
        clean_d = re.sub(r'[^a-zA-Z0-9_-]', '_', disease_id)[:20]
        doc_hash = hashlib.md5(doc_id.encode()).hexdigest()[:12]

        return (
            f"{clean_d}__{doc_hash}__{suffix}"
        )

    def _submit_openai_chunk(self, prompts: list[str], disease_id: str, doc_ids: list[str]) -> str | None:
        """Submit a single chunk to OpenAI Batch API."""
        lines = []
        for prompt, doc_id in zip(prompts, doc_ids):
            cid = self._make_custom_id(disease_id, doc_id, "gpt4omini")
            request = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 4096,
                    "response_format": {"type": "json_object"},
                },
            }
            lines.append(json.dumps(request))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(lines))
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                file_obj = self._openai_client.files.create(file=f, purpose="batch")

            batch = self._openai_client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"disease_id": disease_id, "prompt_count": str(len(prompts))},
            )
            return batch.id
        finally:
            os.unlink(temp_path)

    def _submit_anthropic_chunk(self, prompts: list[str], disease_id: str, doc_ids: list[str]) -> str | None:
        """Submit a single chunk to Anthropic Message Batches API."""
        requests = []
        for prompt, doc_id in zip(prompts, doc_ids):
            cid = self._make_custom_id(disease_id, doc_id, "claudehaiku")
            requests.append({
                "custom_id": cid,
                "params": {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 4096,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}],
                },
            })

        batch = self._anthropic_client.beta.messages.batches.create(requests=requests)
        return batch.id

    # -----------------------------------------------------------------------
    # Poll
    # -----------------------------------------------------------------------

    def poll(
        self,
        batch_ids: dict[str, list[str] | str],
        poll_interval: int = 30,
        max_wait: int = 7200,
        on_progress: Callable[[dict], None] | None = None,
    ) -> dict[str, dict[str, str]]:
        """
        Poll all batch APIs until completion.

        Args:
            batch_ids: {provider: batch_id} or {provider: [batch_id, ...]}
            poll_interval: Seconds between polls
            max_wait: Max seconds to wait
            on_progress: Optional callback with current statuses

        Returns:
            {provider: {batch_id: status}} where status is "completed", "failed", or "timeout"
        """
        # Normalize to {provider: [batch_id, ...]}
        normalized = {}
        for provider, bids in batch_ids.items():
            if isinstance(bids, str):
                normalized[provider] = [bids]
            else:
                normalized[provider] = list(bids)

        # Track status per individual batch
        statuses: dict[str, dict[str, str]] = {}
        for provider, bids in normalized.items():
            statuses[provider] = {bid: "pending" for bid in bids}

        start = time.monotonic()

        while time.monotonic() - start < max_wait:
            all_done = True

            for provider, bid_statuses in statuses.items():
                for bid, status in bid_statuses.items():
                    if status in ("completed", "failed"):
                        continue

                    try:
                        if provider == "openai":
                            new_status = self._poll_openai(bid)
                        elif provider == "anthropic":
                            new_status = self._poll_anthropic(bid)
                        else:
                            new_status = "unknown"

                        statuses[provider][bid] = new_status
                        if new_status not in ("completed", "failed"):
                            all_done = False
                    except Exception as e:
                        logger.warning("Poll error for %s batch %s: %s", provider, bid, e)
                        all_done = False

            if on_progress:
                on_progress(statuses)

            if all_done:
                break

            elapsed = int(time.monotonic() - start)
            pending = []
            for p, bs in statuses.items():
                for bid, s in bs.items():
                    if s not in ("completed", "failed"):
                        pending.append(f"{p}:{bid[:12]}")
            logger.info("Batch poll: %ds elapsed, pending: %s", elapsed, pending)
            time.sleep(poll_interval)

        # Mark remaining as timeout
        for provider in statuses:
            for bid in statuses[provider]:
                if statuses[provider][bid] not in ("completed", "failed"):
                    statuses[provider][bid] = "timeout"

        return statuses

    def _poll_openai(self, batch_id: str) -> str:
        batch = self._openai_client.batches.retrieve(batch_id)
        if batch.status == "completed":
            return "completed"
        elif batch.status in ("failed", "expired", "cancelled"):
            return "failed"
        return "pending"

    def _poll_anthropic(self, batch_id: str) -> str:
        batch = self._anthropic_client.beta.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return "completed"
        return "pending"

    # -----------------------------------------------------------------------
    # Retrieve
    # -----------------------------------------------------------------------

    def retrieve(
        self,
        batch_ids: dict[str, list[str] | str],
    ) -> dict[str, dict]:
        """
        Retrieve results from completed batches.

        Accepts both v1 format {provider: batch_id} and v2 format {provider: [batch_id, ...]}.

        Returns:
            Dict of {custom_id: {"model": model_name, "triples": [raw_dicts]}}
            (flat dict for backward compatibility with run_batch_17k.py)
        """
        # Normalize
        normalized = {}
        for provider, bids in batch_ids.items():
            if isinstance(bids, str):
                normalized[provider] = [bids]
            else:
                normalized[provider] = list(bids)

        results = {}

        if "openai" in normalized:
            for bid in normalized["openai"]:
                try:
                    openai_results = self._retrieve_openai(bid)
                    results.update(openai_results)
                except Exception as e:
                    logger.error("OpenAI batch retrieval failed for %s: %s", bid, e)

        if "anthropic" in normalized:
            for bid in normalized["anthropic"]:
                try:
                    anthropic_results = self._retrieve_anthropic(bid)
                    results.update(anthropic_results)
                except Exception as e:
                    logger.error("Anthropic batch retrieval failed for %s: %s", bid, e)

        return results

    def _retrieve_openai(self, batch_id: str) -> dict[str, dict]:
        """Retrieve and parse OpenAI batch results with token tracking."""
        batch = self._openai_client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            logger.warning("OpenAI batch %s has no output file", batch_id)
            return {}

        content = self._openai_client.files.content(batch.output_file_id)
        results = {}

        for line in content.text.strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                custom_id = entry["custom_id"]
                response = entry.get("response", {})
                body = response.get("body", {})

                # Track token usage
                usage = body.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                self.cost_tracker.add("openai", input_tokens, output_tokens)

                # Extract text from chat completion response
                choices = body.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    triples = self._parse_json_response(text)
                else:
                    triples = []

                results[custom_id] = {
                    "model": "gpt-4o-mini",
                    "triples": triples,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            except Exception as e:
                logger.debug("Failed to parse OpenAI batch result: %s", e)

        # Also check error file
        if batch.error_file_id:
            try:
                err_content = self._openai_client.files.content(batch.error_file_id)
                error_count = 0
                for line in err_content.text.strip().split("\n"):
                    if line.strip():
                        error_count += 1
                        try:
                            entry = json.loads(line)
                            cid = entry.get("custom_id", "unknown")
                            err = entry.get("error", {})
                            logger.debug("OpenAI batch error for %s: %s", cid, err)
                        except json.JSONDecodeError:
                            pass
                if error_count > 0:
                    logger.warning("OpenAI batch %s had %d errors", batch_id, error_count)
            except Exception:
                pass

        return results

    def _retrieve_anthropic(self, batch_id: str) -> dict[str, dict]:
        """Retrieve and parse Anthropic batch results with token tracking."""
        results = {}

        for result in self._anthropic_client.beta.messages.batches.results(batch_id):
            try:
                custom_id = result.custom_id
                if result.result.type == "succeeded":
                    text = result.result.message.content[0].text
                    triples = self._parse_json_response(text)

                    # Track token usage
                    usage = result.result.message.usage
                    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
                    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
                    self.cost_tracker.add("anthropic", input_tokens, output_tokens)
                else:
                    triples = []
                    input_tokens = 0
                    output_tokens = 0
                    error_type = getattr(result.result, "type", "unknown")
                    logger.debug("Anthropic batch result %s: %s", custom_id, error_type)

                results[custom_id] = {
                    "model": "claude-haiku",
                    "triples": triples,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            except Exception as e:
                logger.debug("Failed to parse Anthropic batch result: %s", e)

        return results

    # -----------------------------------------------------------------------
    # JSON parsing
    # -----------------------------------------------------------------------

    def _parse_json_response(self, text: str) -> list[dict]:
        """Parse JSON response, handling various formats. Same logic as LLMClient."""
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for key in ("triples", "results", "relationships", "data",
                            "extracted_relationships", "extracted_triples",
                            "extractions", "output", "entities"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                list_vals = [(k, v) for k, v in parsed.items() if isinstance(v, list)]
                if len(list_vals) == 1:
                    return list_vals[0][1]
                if "subject" in parsed and "object" in parsed:
                    return [parsed]
                return []
            return []
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return []
