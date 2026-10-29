"""In-memory query history so traces can be fetched after a request."""

from __future__ import annotations

import threading
from collections import OrderedDict

from ...models.query import PipelineResult


class QueryHistory:
    def __init__(self, limit: int = 200) -> None:
        self._limit = limit
        self._lock = threading.RLock()
        self._items: OrderedDict[str, PipelineResult] = OrderedDict()

    def add(self, result: PipelineResult) -> None:
        with self._lock:
            self._items[result.query_id] = result
            self._items.move_to_end(result.query_id)
            while len(self._items) > self._limit:
                self._items.popitem(last=False)

    def get(self, query_id: str) -> PipelineResult | None:
        with self._lock:
            return self._items.get(query_id)

    def recent(self, limit: int = 20) -> list[PipelineResult]:
        with self._lock:
            values = list(self._items.values())
        return list(reversed(values))[:limit]

    def delete(self, query_id: str) -> bool:
        with self._lock:
            if query_id not in self._items:
                return False
            del self._items[query_id]
            return True


history = QueryHistory()
