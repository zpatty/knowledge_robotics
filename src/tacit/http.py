"""Shared HTTP client: cache-first, budget-guarded, backoff on 429."""
from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urlencode

import requests

from .budget import DailyBudget
from .cache import ResponseCache, scrub_url

log = logging.getLogger(__name__)


class ApiError(RuntimeError):
    """An HTTP error with the credential scrubbed out of the message."""


class QuotaExhausted(ApiError):
    """The provider's daily allowance is spent, not merely a momentary rate limit.

    Distinguished from an ordinary 429 by the length of Retry-After: a "slow down"
    is seconds, a spent daily quota is hours. Retrying will not help before the
    quota resets, so this is raised rather than slept through.
    """


class AccountInactive(ApiError):
    """The API account is not activated — distinct from a wrong or expired key.

    IEEE fronts its API with Mashery, which answers an unactivated developer
    account with 403 ERR_403_DEVELOPER_INACTIVE for *any* key, valid or not. That
    is a registration problem, not a credential problem, and no amount of retrying
    or key-rotation fixes it — so it is raised as its own type rather than being
    lost in a generic 403.
    """


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
        max_retry_wait: float = 120.0,
    ):
        self.cache = ResponseCache(namespace)
        self.budget = budget
        self.min_interval = min_interval
        self.dry_run = dry_run
        self.timeout = timeout
        self.max_retry_wait = max_retry_wait
        self.session = requests.Session()
        # Provider-reported budget remaining, when the API sends it (OpenAlex does).
        # Authoritative where present: our local estimate is only a reservation.
        self.provider_remaining: int | None = None
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

    @staticmethod
    def canonical_url(base: str, params: dict) -> str:
        """Order-independent URL, used as the cache key.

        Callers build params as keyword arguments, so two spellings of the *same*
        query can differ only in dict order. Keying the cache on the unsorted URL
        would make those miss each other and pay for the identical request twice —
        which, at 200 IEEE calls a day, is a real cost rather than a tidiness point.
        Sorting makes the key canonical. The outbound request is unaffected: query
        parameter order is not significant to either API.
        """
        return f"{base}?{urlencode(sorted(params.items()))}"

    def get(
        self,
        base: str,
        params: dict,
        *,
        secret_params: dict | None = None,
        cost: int = 1,
    ) -> Any:
        """GET with caching. secret_params are sent but excluded from the cache key.

        `cost` is how many budget units this call consumes. IEEE meters whole calls
        (cost=1); OpenAlex meters *credits* that scale with page size, so a caller
        that pages 200 at a time spends ten times what the call count suggests.
        """
        url = self.canonical_url(base, params)
        cached = self.cache.get(url)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        if self.dry_run:
            self.planned.append(url)
            return None

        self.budget.spend(cost)  # Reserve before the call: the API charges us either way.
        self._throttle()

        full = dict(params)
        full.update(secret_params or {})
        delay = 2.0
        for attempt in range(5):
            try:
                resp = self.session.get(base, params=full, timeout=self.timeout)
            except requests.RequestException as exc:
                # scrub: requests puts the full URL, credential included, in str(exc).
                log.warning(
                    "request error (%s), retrying in %.0fs", scrub_url(str(exc)), delay
                )
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 403:
                detail = resp.headers.get("x-error-detail-header", "")
                code = resp.headers.get("x-mashery-error-code", "")
                if "INACTIVE" in code.upper() or "inactive" in detail.lower():
                    raise AccountInactive(
                        f"{code or '403'}: {detail or 'account inactive'}. "
                        "The API account itself is not activated, so no key will "
                        "work — this is not a bad-credential error. Activate the "
                        "developer account at the provider's portal."
                    )
                raise ApiError(f"403 Forbidden for {scrub_url(url)}: {detail or code}")

            if resp.status_code == 429:
                # Do not re-spend budget: the reservation above already covers this call.
                wait = float(resp.headers.get("Retry-After", delay))
                if wait > self.max_retry_wait:
                    # A 429 can mean "slow down" (seconds) or "your daily quota is
                    # gone" (hours). OpenAlex signals the latter with a Retry-After
                    # of up to a full day; sleeping on it would hang the harvest
                    # overnight with no output. Fail loudly and let the operator
                    # decide to wait, top up, or resume tomorrow from cache.
                    raise QuotaExhausted(
                        f"429 with Retry-After={wait:.0f}s "
                        f"({wait / 3600:.1f}h), beyond max_retry_wait="
                        f"{self.max_retry_wait:.0f}s. The daily quota is most likely "
                        f"spent. Response: {resp.text[:200]}"
                    )
                log.warning("429 rate limited, sleeping %.0fs", wait)
                time.sleep(wait)
                delay *= 2
                continue

            if resp.status_code >= 500:
                log.warning("server %s, retrying in %.0fs", resp.status_code, delay)
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code >= 400:
                # Not raise_for_status(): its message embeds the full URL, key included.
                raise ApiError(
                    f"HTTP {resp.status_code} for {scrub_url(url)}: "
                    f"{resp.text[:200]}"
                )
            remaining = resp.headers.get("x-ratelimit-remaining")
            if remaining is not None:
                try:
                    self.provider_remaining = int(remaining)
                except ValueError:
                    pass
            payload = resp.json()
            self.cache.put(url, payload)
            return payload

        raise RuntimeError(f"giving up on {base} after 5 attempts")
