"""
Base Agent
==========
Abstract base class for all ChronoMedKG pipeline agents.
Provides retry logic, logging, and metrics collection.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime

from core.models import AgentResult


class BaseAgent(ABC):
    """Abstract base for pipeline agents."""

    def __init__(self, config: dict, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.retry_count = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay_seconds", 5)

    @abstractmethod
    async def run(self, input_data: dict) -> AgentResult:
        """Execute the agent's primary task."""
        ...

    async def run_with_retry(self, input_data: dict) -> AgentResult:
        """Execute with exponential backoff retry."""
        last_error = None
        for attempt in range(self.retry_count):
            try:
                start = time.monotonic()
                result = await self.run(input_data)
                elapsed = time.monotonic() - start
                result.metrics["elapsed_seconds"] = round(elapsed, 2)
                return result
            except Exception as e:
                last_error = e
                self.logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt + 1, self.retry_count,
                    input_data.get("disease_id", "unknown"), e,
                )
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

        return AgentResult(
            agent_name=self.__class__.__name__,
            disease_id=input_data.get("disease_id", "unknown"),
            status="failed",
            data={},
            metrics={},
            errors=[f"All {self.retry_count} attempts failed. Last error: {last_error}"],
            timestamp=datetime.utcnow(),
        )
