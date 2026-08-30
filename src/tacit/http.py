"""Shared HTTP client: cache-first, budget-guarded, backoff on 429."""
from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urlencode

import requests

from .budget import DailyBudget
from .cache import ResponseCache

log = logging.getLogger(__name__)


class ApiClient:
    """Every outbound call passes through cache -> budget -> network, in that order.

    dry_run reports what would be requested without spending budget or touching the
    network, which is how a harvest plan gets reviewed before it costs anything.
    """

    def __init__(
        self,
        namespace: str,
        budget: DailyBudget,
        *,
        min_interval: float = 0.0,
        dry_run: bool = False,
        timeout: int = 60,
    ):
        self.cache = ResponseCache(namespace)
        self.budget = budget
        self.min_interval = min_interval
        self.dry_run = dry_run
        self.timeout = timeout
        self.session = requests.Session()
        self._last_call = 0.0
        self.planned: list[str] = []
        self.hits = 0
        self.misses = 0

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        delta = time.monotonic() - self._last_call
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last_call = time.monotonic()

    def get(self, base: str, params: dict, *, secret_params: dict | None = None) -> Any:
        """GET with caching. secret_params are sent but excluded from the cache key."""
        url = f"{base}?{urlencode(params)}"
        cached = self.cache.get(url)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        if self.dry_run:
            self.planned.append(url)
            return None

        self.budget.spend(1)  # Reserve before the call: the API charges us either way.
        self._throttle()

        full = dict(params)
        full.update(secret_params or {})
        delay = 2.0
        for attempt in range(5):
            try:
                resp = self.session.get(base, params=full, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("request error (%s), retrying in %.0fs", exc, delay)
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 429:
                # Do not re-spend budget: the reservation above already covers this call.
                wait = float(resp.headers.get("Retry-After", delay))
                log.warning("429 rate limited, sleeping %.0fs", wait)
                time.sleep(wait)
                delay *= 2
                continue

            if resp.status_code >= 500:
                log.warning("server %s, retrying in %.0fs", resp.status_code, delay)
                time.sleep(delay)
                delay *= 2
                continue

            resp.raise_for_status()
            payload = resp.json()
            self.cache.put(url, payload)
            return payload

        raise RuntimeError(f"giving up on {base} after 5 attempts")
