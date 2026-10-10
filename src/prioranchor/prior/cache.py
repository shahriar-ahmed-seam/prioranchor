"""Content-addressable storage and fingerprinting for synthetic class priors."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PriorManifest(BaseModel):
    """Metadata manifest tracking cached prior distributions."""
    fingerprint: str
    base_model_path: str
    class_noun: str
    num_samples: int
    resolution: int
    seed: int
    created_at: float
    entries: List[Dict[str, Any]] = Field(default_factory=list)


def compute_prior_fingerprint(
    base_model_path: str,
    class_noun: str,
    resolution: int,
    num_samples: int,
    seed: int,
    templates: Optional[List[str]] = None,
) -> str:
    """Compute a deterministic SHA-256 fingerprint for a prior generation configuration."""
    hasher = hashlib.sha256()
    hasher.update(base_model_path.strip().encode("utf-8"))
    hasher.update(class_noun.strip().lower().encode("utf-8"))
    hasher.update(str(resolution).encode("utf-8"))
    hasher.update(str(num_samples).encode("utf-8"))
    hasher.update(str(seed).encode("utf-8"))
    if templates:
        for t in sorted(templates):
            hasher.update(t.encode("utf-8"))
    return hasher.hexdigest()[:16]


class PriorCache:
    """Manages content-addressable local disk cache for class-generic priors."""

    def __init__(self, cache_root: str = "prior_cache"):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, fingerprint: str) -> Path:
        """Return the directory corresponding to a specific configuration fingerprint."""
        target = self.cache_root / fingerprint
        target.mkdir(parents=True, exist_ok=True)
        return target

    def has_valid_cache(self, fingerprint: str, expected_count: int) -> bool:
        """Check if cached priors exist and contain the required sample count."""
        dir_path = self.cache_root / fingerprint
        manifest_path = dir_path / "prior_manifest.json"
        if not manifest_path.exists():
            return False

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            manifest = PriorManifest.model_validate(data)
            return len(manifest.entries) >= expected_count
        except Exception:
            return False

    def save_manifest(self, manifest: PriorManifest) -> Path:
        """Write manifest index to cache directory."""
        dir_path = self.get_cache_path(manifest.fingerprint)
        manifest_path = dir_path / "prior_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        return manifest_path

    def load_manifest(self, fingerprint: str) -> Optional[PriorManifest]:
        """Read and validate cached manifest index."""
        dir_path = self.cache_root / fingerprint
        manifest_path = dir_path / "prior_manifest.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PriorManifest.model_validate(data)

    def clean_orphaned(self) -> int:
        """Remove empty cache subdirectories."""
        removed = 0
        for entry in self.cache_root.iterdir():
            if entry.is_dir() and not (entry / "prior_manifest.json").exists():
                for sub in entry.iterdir():
                    if sub.is_file():
                        os.remove(sub)
                entry.rmdir()
                removed += 1
        return removed
