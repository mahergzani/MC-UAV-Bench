"""
Resolve MC-UAV-Bench image paths against a local UAV3D v1.0 copy.

The released JSON stores image paths relative to the UAV3D v1.0 root
(e.g. 'sweeps/town07_row4/CAMERA_FRONT_id_0/<ts>.png'). The root is taken
from --image-root, else the UAV3D_ROOT environment variable.

Missing images raise FileNotFoundError: evaluation never silently falls back
to text-only input.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

ENV_VAR = "UAV3D_ROOT"


def get_image_root(image_root: Optional[str] = None) -> Path:
    root = image_root or os.environ.get(ENV_VAR)
    if not root:
        raise ValueError(
            f"No image root given. Pass --image-root /path/to/UAV3D/v1.0 "
            f"or set {ENV_VAR}."
        )
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {root}")
    return root


def resolve_image_path(path: str, root: Path) -> str:
    p = Path(path)
    full = p if p.is_absolute() else root / p
    if not full.is_file():
        raise FileNotFoundError(
            f"Camera image not found: {full}\n"
            f"Check that --image-root / {ENV_VAR} points at the UAV3D v1.0 root "
            f"(the folder containing 'sweeps/')."
        )
    return str(full)


def resolve_image_paths(image_paths: Dict[str, str], root: Path) -> Dict[str, str]:
    return {cam: resolve_image_path(p, root) for cam, p in image_paths.items()}


def resolve_samples(samples: List[dict], image_root: Optional[str] = None) -> List[dict]:
    """Rewrite every sample's image_paths to absolute local paths, in place.
    Fails on the first missing image, before any model is loaded."""
    root = get_image_root(image_root)
    for s in samples:
        s["image_paths"] = resolve_image_paths(s.get("image_paths", {}), root)
    return samples
