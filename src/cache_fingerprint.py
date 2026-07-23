"""Small source-code fingerprints used to invalidate research caches safely.

Cache keys already include the data path, date boundaries, and user parameters.
They also need to change when the implementation itself changes. Otherwise a
modified Alpha formula, preprocessing rule, selector, or model can silently reuse
an output produced by older code while appearing to be a fresh experiment.

The helper hashes only a short explicit list of source files. This costs far less
than rebuilding a feature matrix and avoids relying exclusively on a manually
incremented cache-version string.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_source_fingerprint(relative_paths: Iterable[str]) -> dict[str, str]:
    """Return deterministic SHA256 hashes for cache-relevant source files."""

    fingerprints: dict[str, str] = {}
    for relative_path in sorted(set(str(path) for path in relative_paths)):
        source_path = PROJECT_ROOT / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Cache fingerprint source is missing: {source_path}")
        fingerprints[relative_path] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return fingerprints
