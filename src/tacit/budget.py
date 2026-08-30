"""Persistent per-day API call budget.

The IEEE Xplore key allows 200 calls per day. A bug that burns the budget costs a
day of work, so every call goes through a guard that (a) persists its count across
process restarts, and (b) refuses rather than exceeds. Reserve-then-commit so a
crashed request still counts: the API charged us for it either way.
"""
from __future__ import annotations

import datetime as _dt
import json
import threading
from pathlib import Path

from .config import DATA


class BudgetExceeded(RuntimeError):
    pass


class DailyBudget:
    def __init__(self, name: str, limit: int, path: Path | None = None):
        self.name = name
        self.limit = limit
        self.path = path or DATA / f"budget_{name}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> str:
        # IEEE and OpenAlex both reset on UTC midnight.
        return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {}

    def used(self) -> int:
        return int(self._read().get(self._today(), 0))

    def remaining(self) -> int:
        return max(0, self.limit - self.used())

    def spend(self, n: int = 1) -> int:
        """Reserve n calls. Raises BudgetExceeded rather than going over."""
        with self._lock:
            state = self._read()
            today = self._today()
            used = int(state.get(today, 0))
            if used + n > self.limit:
                raise BudgetExceeded(
                    f"{self.name}: {used}/{self.limit} calls used today; "
                    f"{n} more would exceed. Resets at UTC midnight."
                )
            state[today] = used + n
            # Keep only the last 30 days.
            for key in sorted(state)[:-30]:
                state.pop(key, None)
            self.path.write_text(json.dumps(state, indent=2, sort_keys=True))
            return self.limit - state[today]
