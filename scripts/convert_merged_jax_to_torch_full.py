#!/usr/bin/env python
"""Build a Torch full checkpoint from a merged JAX fine-tuned checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch_effective_conv import jax_effective_paths_to_torch, materialize_effective_convs


def _add_torch_repo(path: Path) -> None:
    path = path.expanduser().resolve()
    for entry in (path / "src", path):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in tree.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, name))
        else:
            out[name] = value
    return out


def _load_jax_flat(checkpoint: Path) -> dict[str, Any]:
    import orbax.checkpoint as ocp

    params, state = ocp.StandardCheckpointer().restore(str(checkpoint))
    flat = _flatten(params)
    if state:
        flat.update(_flatten(state))
    return flat


def _load_torch_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    return payload


def _load_checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint.expanduser().resolve().parent / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open() as handle:
        return json.load(handle)


def _discover_custom_head(
    flat_jax: dict[str, Any], head_id: str
) -> tuple[list[int], int, int]:
    prefix = f"head/{head_id}/"
    resolutions: list[int] = []
    n_tracks: int | None = None
    n_org: int | None = None
    for key, value in flat_jax.items():
        if not key.startswith(prefix) or not key.endswith("/multi_organism_linear/w"):
            continue
        parts = key.split("/")
        res = int(parts[2].removeprefix("resolution_"))
        arr = np.asarray(value)
        resolutions.append(res)
        n_org = int(arr.shape[0])
        n_tracks = int(arr.shape[2])
    if not resolutions or n_tracks is None or n_org is None:
        raise KeyError(f"No JAX custom head weights found under {prefix!r}")
    return sorted(resolutions), n_tracks, n_org


def _custom_head_state(
    flat_jax: dict[str, Any],
    *,
    head_id: str,
    resolutions: list[int],
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for res in resolutions:
        base = f"head/{head_id}/resolution_{res}"
        weight = np.asarray(flat_jax[f"{base}/multi_organism_linear/w"])
        bias = np.asarray(flat_jax[f"{base}/multi_organism_linear/b"])
        scale = np.asarray(flat_jax[f"{base}/learnt_scale"])
        state[f"heads.{head_id}.convs.{res}.weight"] = torch.from_numpy(
            weight.transpose(0, 2, 1).copy()
        )
        state[f"heads.{head_id}.convs.{res}.bias"] = torch.from_numpy(bias.copy())
        state[f"heads.{head_id}.residual_scales.{res}"] = torch.from_numpy(scale.copy())
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-checkpoint", type=Path, required=True)
    parser.add_argument("--base-torch-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--torch-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "alphagenome-pytorch",
    )
    parser.add_argument("--head-id", default="humanbraindev_atac")
    parser.add_argument("--assay-type", default="atac")
    parser.add_argument("--bigwig-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _add_torch_repo(args.torch_repo)

    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
    from alphagenome_pytorch.extensions.finetuning.transfer import add_head, remove_all_heads
    from alphagenome_pytorch.jax_compat.transforms import apply_transform
    from alphagenome_pytorch.jax_compat.weight_mapping import map_pytorch_to_jax

    jax_checkpoint = args.jax_checkpoint.expanduser().resolve()
    flat_jax = _load_jax_flat(jax_checkpoint)
    checkpoint_config = _load_checkpoint_config(jax_checkpoint)
    effective_jax_paths = tuple(
        str(path)
        for path in (
            checkpoint_config.get("backbone_effective_conv_paths")
            or checkpoint_config.get("effective_conv_paths")
            or ()
        )
    )
    effective_torch_modules = jax_effective_paths_to_torch(effective_jax_paths)
    resolutions, n_tracks, n_org = _discover_custom_head(flat_jax, args.head_id)
    custom_state = _custom_head_state(flat_jax, head_id=args.head_id, resolutions=resolutions)

    model = AlphaGenome(num_organisms=2)
    model = remove_all_heads(model)
    add_head(
        model,
        args.head_id,
        create_finetuning_head(
            args.assay_type,
            n_tracks=n_tracks,
            resolutions=tuple(resolutions),
            num_organisms=n_org,
        ),
    )

    base_state = _load_torch_state(args.base_torch_weights.expanduser().resolve())
    model.load_state_dict(base_state, strict=False)
    merged_state = model.state_dict()
    converted_backbone = 0
    for key, value in list(merged_state.items()):
        jax_key = map_pytorch_to_jax(key)
        if jax_key is None or jax_key not in flat_jax:
            continue
        converted = apply_transform(key, np.asarray(flat_jax[jax_key]), tuple(value.shape))
        merged_state[key] = torch.from_numpy(converted.copy()).to(dtype=value.dtype)
        converted_backbone += 1
    for key, value in custom_state.items():
        if key not in merged_state:
            raise KeyError(f"Torch custom head key not present: {key}")
        if tuple(merged_state[key].shape) != tuple(value.shape):
            raise ValueError(f"Shape mismatch for {key}: {tuple(value.shape)} vs {tuple(merged_state[key].shape)}")
        merged_state[key] = value.to(dtype=merged_state[key].dtype)
    model.load_state_dict(merged_state, strict=True)
    materialize_effective_convs(model, effective_torch_modules)

    track_names: list[str]
    if args.bigwig_dir is not None:
        track_names = [path.stem for path in sorted(args.bigwig_dir.expanduser().glob("*.bw"))]
    else:
        track_names = [f"track_{idx}" for idx in range(n_tracks)]
    if len(track_names) != n_tracks:
        raise ValueError(f"Expected {n_tracks} track names, found {len(track_names)}")

    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "head_id": args.head_id,
            "assay_type": args.assay_type,
            "modality": args.head_id,
            "resolutions": tuple(resolutions),
            "num_organisms": n_org,
            "track_names": track_names,
            "source_jax_checkpoint": str(jax_checkpoint),
            "source_base_torch_weights": str(args.base_torch_weights.expanduser().resolve()),
            "backbone_effective_conv_paths": effective_jax_paths,
            "torch_effective_conv_modules": effective_torch_modules,
            "converted_backbone_leaves": converted_backbone,
        },
        args.output.expanduser().resolve(),
    )
    print(
        f"Saved Torch full checkpoint to {args.output} "
        f"(head={args.head_id}, tracks={n_tracks}, resolutions={resolutions}, organisms={n_org}, "
        f"converted_backbone_leaves={converted_backbone}, effective_convs={len(effective_torch_modules)})"
    )


if __name__ == "__main__":
    main()
