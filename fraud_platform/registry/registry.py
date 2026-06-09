"""Local versioned model registry.

Filesystem layout:

    registry_store/
    ├── index.json                      # {model_type: {"champion": version_int}}
    └── <model_type>/
        └── v<N>/
            ├── model.pkl | model.pt     # the model + its preprocessing pipeline
            └── metadata.json            # version, timestamp, model_type, metrics, data_hash

Every register() writes a new immutable version (never overwrites). The "champion" pointer
in index.json marks which version serving should load. promote() flips the champion only if
a candidate beats the current champion on the chosen metric — that gate lives in retrain.py.

This is deliberately simple and dependency-free so it runs anywhere and is easy to read.
In production you'd use MLflow (or similar): experiment tracking, a backing artifact store
(S3/GCS), model staging/transitions, and lineage. See the README for that note.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fraud_platform.config import REGISTRY_DIR
from fraud_platform.models import ARTIFACT_EXT, load_model


class ModelRegistry:
    def __init__(self, root: str | Path = REGISTRY_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists():
            self._write_index({})

    # ---- index helpers ----
    def _read_index(self) -> dict:
        with open(self.index_path) as f:
            return json.load(f)

    def _write_index(self, idx: dict) -> None:
        with open(self.index_path, "w") as f:
            json.dump(idx, f, indent=2)

    # ---- versioning ----
    def _versions(self, model_type: str) -> list[int]:
        d = self.root / model_type
        if not d.exists():
            return []
        return sorted(int(p.name[1:]) for p in d.glob("v*") if p.name[1:].isdigit())

    def next_version(self, model_type: str) -> int:
        vs = self._versions(model_type)
        return (max(vs) + 1) if vs else 1

    def register(self, model, metrics: dict, data_hash: str = "",
                 make_champion: bool = False) -> int:
        """Persist a new immutable version of `model`. Returns the version number."""
        model_type = model.name
        version = self.next_version(model_type)
        vdir = self.root / model_type / f"v{version}"
        vdir.mkdir(parents=True, exist_ok=True)

        ext = ARTIFACT_EXT.get(model_type, "pkl")
        artifact = vdir / f"model.{ext}"
        model.save(str(artifact))

        metadata = {
            "model_type": model_type,
            "version": version,
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "data_hash": data_hash,
            "artifact": artifact.name,
        }
        with open(vdir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        idx = self._read_index()
        entry = idx.get(model_type, {})
        if make_champion or "champion" not in entry:
            entry["champion"] = version
        idx[model_type] = entry
        self._write_index(idx)
        return version

    def promote(self, model_type: str, version: int) -> None:
        """Set the champion pointer for a model type to a specific version."""
        if version not in self._versions(model_type):
            raise ValueError(f"{model_type} v{version} does not exist")
        idx = self._read_index()
        idx.setdefault(model_type, {})["champion"] = version
        self._write_index(idx)

    def champion_version(self, model_type: str) -> int | None:
        return self._read_index().get(model_type, {}).get("champion")

    def get_metadata(self, model_type: str, version: int) -> dict:
        with open(self.root / model_type / f"v{version}" / "metadata.json") as f:
            return json.load(f)

    def champion_metadata(self, model_type: str) -> dict | None:
        v = self.champion_version(model_type)
        return self.get_metadata(model_type, v) if v else None

    def load_champion(self, model_type: str):
        """Load the current champion model object for a type, or None if none registered."""
        v = self.champion_version(model_type)
        if v is None:
            return None
        meta = self.get_metadata(model_type, v)
        artifact = self.root / model_type / f"v{v}" / meta["artifact"]
        return load_model(model_type, str(artifact))

    def list_versions(self, model_type: str) -> list[dict]:
        return [self.get_metadata(model_type, v) for v in self._versions(model_type)]
