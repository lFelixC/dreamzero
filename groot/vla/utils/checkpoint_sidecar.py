from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from safetensors import safe_open


FULL_CHECKPOINT_REQUIRED_PREFIXES = (
    "action_head.text_encoder.",
    "action_head.image_encoder.",
    "action_head.vae.",
    "action_head.model.",
)

_COMPONENT_DEFAULT_FILENAMES = {
    "text_encoder_pretrained_path": ("models_t5_umt5-xxl-enc-bf16.pth",),
    "image_encoder_pretrained_path": ("models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",),
    "vae_pretrained_path": ("Wan2.1_VAE.pth", "Wan2.2_VAE.pth"),
}

_NESTED_COMPONENT_CFG_FIELDS = {
    "text_encoder_pretrained_path": ("text_encoder_cfg", "text_encoder_pretrained_path"),
    "image_encoder_pretrained_path": ("image_encoder_cfg", "image_encoder_pretrained_path"),
    "vae_pretrained_path": ("vae_cfg", "vae_pretrained_path"),
}


def _mapping_like(value: Any) -> bool:
    return hasattr(value, "keys") and hasattr(value, "__getitem__") and hasattr(value, "__setitem__")


def _get(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if _mapping_like(container):
        try:
            return container[key]
        except Exception:
            return default
    return getattr(container, key, default)


def _set(container: Any, key: str, value: Any) -> None:
    if _mapping_like(container):
        container[key] = value
    else:
        setattr(container, key, value)


def _inner_action_head_cfg(action_head_cfg: Any) -> Any:
    inner = _get(action_head_cfg, "config")
    return inner if inner is not None else action_head_cfg


def _candidate_component_filenames(action_head_cfg: Any) -> dict[str, list[str]]:
    inner = _inner_action_head_cfg(action_head_cfg)
    out: dict[str, list[str]] = {}
    for field, defaults in _COMPONENT_DEFAULT_FILENAMES.items():
        candidates: list[str] = []
        configured = _get(inner, field)
        if configured:
            candidates.append(Path(str(configured)).name)
        nested_cfg_name, nested_field = _NESTED_COMPONENT_CFG_FIELDS[field]
        nested_cfg = _get(inner, nested_cfg_name)
        nested_configured = _get(nested_cfg, nested_field)
        if nested_configured:
            nested_name = Path(str(nested_configured)).name
            if nested_name not in candidates:
                candidates.append(nested_name)
        for default_name in defaults:
            if default_name not in candidates:
                candidates.append(default_name)
        out[field] = candidates
    return out


def find_local_sidecar_overrides(action_head_cfg: Any, checkpoint_dir: str | os.PathLike[str]) -> dict[str, str]:
    checkpoint_dir = Path(checkpoint_dir)
    overrides: dict[str, str] = {}
    for field, candidates in _candidate_component_filenames(action_head_cfg).items():
        for candidate in candidates:
            candidate_path = checkpoint_dir / candidate
            if candidate_path.exists():
                overrides[field] = str(candidate_path)
                break
    return overrides


def apply_local_sidecar_overrides(action_head_cfg: Any, checkpoint_dir: str | os.PathLike[str]) -> dict[str, str]:
    overrides = find_local_sidecar_overrides(action_head_cfg, checkpoint_dir)
    inner = _inner_action_head_cfg(action_head_cfg)
    for field, path in overrides.items():
        if _get(inner, field) is not None:
            _set(inner, field, path)
        nested_cfg_name, nested_field = _NESTED_COMPONENT_CFG_FIELDS[field]
        nested_cfg = _get(inner, nested_cfg_name)
        if nested_cfg is not None:
            _set(nested_cfg, nested_field, path)
    return overrides


def checkpoint_has_required_prefixes(
    checkpoint_dir: str | os.PathLike[str],
    required_prefixes: tuple[str, ...] = FULL_CHECKPOINT_REQUIRED_PREFIXES,
) -> bool:
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    single_path = checkpoint_dir / "model.safetensors"

    keys: list[str]
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        keys = list(payload.get("weight_map", {}).keys())
    elif single_path.exists():
        with safe_open(str(single_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
    else:
        return False

    return all(any(key.startswith(prefix) for key in keys) for prefix in required_prefixes)


def prepare_action_head_cfg_for_checkpoint(
    action_head_cfg: Any,
    checkpoint_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    overrides = apply_local_sidecar_overrides(action_head_cfg, checkpoint_dir)
    is_self_contained_full = checkpoint_has_required_prefixes(checkpoint_dir)
    if is_self_contained_full:
        inner = _inner_action_head_cfg(action_head_cfg)
        _set(inner, "skip_component_loading", True)
    return {
        "local_sidecars": overrides,
        "self_contained_full": is_self_contained_full,
    }


def _same_file_or_link(src: Path, dst: Path) -> bool:
    try:
        return dst.exists() and src.samefile(dst)
    except FileNotFoundError:
        return False
    except OSError:
        return False


def link_or_copy_file(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> str:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if _same_file_or_link(src_path, dst_path):
        return "exists"

    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()

    try:
        os.link(src_path, dst_path)
        return "hardlink"
    except OSError:
        shutil.copy2(src_path, dst_path)
        return "copy"


def materialize_component_sidecars(model: Any, output_dir: str | os.PathLike[str]) -> dict[str, str]:
    action_head = getattr(model, "action_head", None)
    if action_head is None:
        return {}

    output_dir = Path(output_dir)

    # Full checkpoints already contain text/image/VAE weights inside model.safetensors
    # shards. In that case, writing sidecar files is redundant and wastes inode / disk.
    if checkpoint_has_required_prefixes(output_dir):
        return {}

    component_sources = {
        "text_encoder_pretrained_path": getattr(getattr(action_head, "text_encoder", None), "text_encoder_pretrained_path", None),
        "image_encoder_pretrained_path": getattr(getattr(action_head, "image_encoder", None), "image_encoder_pretrained_path", None),
        "vae_pretrained_path": getattr(getattr(action_head, "vae", None), "vae_pretrained_path", None),
    }

    sidecars: dict[str, str] = {}
    for field, source in component_sources.items():
        if not source:
            continue
        source_path = Path(str(source))
        if not source_path.exists():
            continue
        target_path = output_dir / source_path.name
        link_or_copy_file(source_path, target_path)
        sidecars[field] = str(target_path)
    return sidecars
