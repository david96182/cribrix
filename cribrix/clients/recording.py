"""Record / replay store for model responses.

Why this exists
---------------
The evaluation harness is only credible if it measures the *real* models. But
live calls cost money, are rate limited and are not reproducible, so CI cannot
depend on them. The standard answer — used throughout the TypeSafe cookbooks —
is a response cache:

* ``record``  call the live API and persist every response to a JSON file.
* ``replay``  answer from the file only; a miss is an error, never a network call.

The file is committed, so anyone can reproduce the published numbers offline,
byte for byte, and CI can gate merges on real-model behaviour.

Keys are a SHA-256 over the canonical JSON of the request, so any change to a
prompt, rubric, chunk or claim produces a miss rather than silently reusing a
stale answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

RecordingMode = Literal["record", "replay"]

DEFAULT_RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "recordings"


class RecordingMiss(LookupError):
    """Raised in replay mode when a request was never recorded."""


def request_key(payload: Any) -> str:
    """Stable hash of a JSON-serialisable request."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """A JSON file mapping request hashes to recorded responses.

    Writes are atomic (temp file + rename) and batched, so an interrupted
    recording run keeps everything captured so far and a re-run only pays for
    the misses. That matters on rate-limited free tiers.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        mode: RecordingMode,
        meta: dict[str, Any] | None = None,
        flush_every: int = 20,
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        self._flush_every = flush_every
        self._dirty = 0
        self.meta: dict[str, Any] = {}
        self._entries: dict[str, Any] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.meta = dict(data.get("meta", {}))
            self._entries = dict(data.get("entries", {}))
        elif mode == "replay":
            raise RecordingMiss(
                f"no recording at {self.path}. Record one first with `make eval-record`."
            )
        if meta and mode == "record":
            self.meta.update(meta)

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> Any | None:
        """Return the recorded response for `key`, or None."""
        return self._entries.get(key)

    def put(self, key: str, value: Any) -> None:
        """Store a response. Only legal in record mode."""
        if self.mode != "record":
            raise RuntimeError("ResponseCache.put() called in replay mode")
        self._entries[key] = value
        self._dirty += 1
        if self._dirty >= self._flush_every:
            self.flush()

    def update_meta(self, **values: Any) -> None:
        """Merge descriptive metadata (model versions, dates) into the file."""
        if self.mode == "record":
            self.meta.update(values)
            self._dirty += 1

    def flush(self) -> None:
        """Atomically persist the store to disk."""
        if self.mode != "record" or not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            {"meta": self.meta, "entries": self._entries},
            indent=1,
            sort_keys=True,
            ensure_ascii=False,
        )
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".rec-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body + "\n")
        os.replace(tmp, self.path)
        self._dirty = 0
