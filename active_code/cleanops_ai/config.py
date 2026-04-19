from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    active_code: Path
    artifacts: Path
    experiments: Path
    datasets: Path
    models: Path
    manifests: Path
    reports: Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = ProjectPaths(
    root=PROJECT_ROOT,
    active_code=PROJECT_ROOT / "active_code",
    artifacts=PROJECT_ROOT / "artifacts",
    experiments=PROJECT_ROOT / "experiments",
    datasets=PROJECT_ROOT / "datasets",
    models=PROJECT_ROOT / "artifacts" / "models",
    manifests=PROJECT_ROOT / "artifacts" / "manifests",
    reports=PROJECT_ROOT / "artifacts" / "reports",
)


def _resolve_env_path(env_name: str, default: Path) -> Path:
    raw_value = os.getenv(env_name)
    if not raw_value:
        return default

    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return PATHS.root / candidate


PPE_MODEL_PATH = _resolve_env_path(
    "PPE_MODEL_PATH",
    PATHS.models / "ppe_detector.pt",
)
PPE_BASE_MODEL_PATH = _resolve_env_path(
    "PPE_BASE_MODEL_PATH",
    PATHS.models / "ppe_detector.pt",
)
PPE_DATA_MANIFEST_PATH = _resolve_env_path(
    "PPE_DATA_MANIFEST_PATH",
    PATHS.manifests / "ppe_dataset.yaml",
)
TRAINING_REPORTS_DIR = _resolve_env_path(
    "PPE_TRAINING_REPORTS_DIR",
    PATHS.reports / "training_runs",
)
