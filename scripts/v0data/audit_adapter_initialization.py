#!/usr/bin/env python
"""Verify identical shared initialization for LoRA and LoRA plus LoCon."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft import (
    BackboneLoConConfig,
    BackboneLoRAConfig,
    create_model_with_heads,
    lora,
    parameter_utils,
)
from alphagenome_ft.finetune import (
    load_targets_config,
    prepare_head_specs,
    register_predefined_heads,
    validate_head_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-config", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence-length", type=int, default=131072)
    return parser.parse_args()


def array_digests(params, selected_paths: set[str]) -> dict[str, str]:
    digests = {}
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    for key_path, value in path_leaves:
        path = parameter_utils._keypath_to_str(key_path)
        if path not in selected_paths:
            continue
        array = np.asarray(value)
        digest = hashlib.sha256()
        digest.update(path.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
        digests[path] = digest.hexdigest()
    if set(digests) != selected_paths:
        raise ValueError(
            f"Could not resolve parameter paths {sorted(selected_paths - set(digests))}."
        )
    return digests


def snapshot(model) -> dict[str, object]:
    head_paths = set(parameter_utils.get_head_parameter_paths(model._params))
    adapter_paths = set(lora.get_lora_parameter_paths(model._params))
    shared_lora_paths = {
        path for path in adapter_paths if path.rsplit("/", 1)[-1] in {"lora_a", "lora_b"}
    }
    locon_up_paths = {path for path in adapter_paths if path.rsplit("/", 1)[-1] == "locon_up_w"}
    locon_up = array_digests(model._params, locon_up_paths) if locon_up_paths else {}
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(model._params)
    nonzero_locon_up = []
    for key_path, value in path_leaves:
        path = parameter_utils._keypath_to_str(key_path)
        if path in locon_up_paths and np.any(np.asarray(value) != 0):
            nonzero_locon_up.append(path)
    return {
        "head": array_digests(model._params, head_paths),
        "shared_lora": array_digests(model._params, shared_lora_paths),
        "locon_up": locon_up,
        "nonzero_locon_up": nonzero_locon_up,
    }


def build_snapshot(args: argparse.Namespace, heads: list[str], *, with_locon: bool):
    lora_config = BackboneLoRAConfig(
        rank=16,
        alpha=16.0,
        base_param_dtype="float32",
        lora_param_dtype="float32",
        activation_dtype="bfloat16",
        base_compute_dtype="bfloat16",
        lora_compute_dtype="bfloat16",
    )
    locon_config = (
        BackboneLoConConfig(
            rank=4,
            alpha=1.0,
            param_dtype="float32",
            compute_dtype="bfloat16",
        )
        if with_locon
        else None
    )
    model = create_model_with_heads(
        "all_folds",
        heads=heads,
        checkpoint_path=args.checkpoint_path,
        detach_backbone=False,
        init_seq_len=args.sequence_length,
        backbone_lora_config=lora_config,
        backbone_locon_config=locon_config,
        runtime_backbone_param_dtype="float32",
    )
    result = snapshot(model)
    del model
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    target_config = load_targets_config(args.targets_config.expanduser().resolve())
    specs = prepare_head_specs(target_config, organism="HOMO_SAPIENS")
    validate_head_specs(specs)
    register_predefined_heads(specs)
    heads = [spec.head_id for spec in specs]

    lora_snapshot = build_snapshot(args, heads, with_locon=False)
    combo_snapshot = build_snapshot(args, heads, with_locon=True)
    head_equal = lora_snapshot["head"] == combo_snapshot["head"]
    shared_lora_equal = lora_snapshot["shared_lora"] == combo_snapshot["shared_lora"]
    locon_zero = not combo_snapshot["nonzero_locon_up"]
    report = {
        "targets_config": str(args.targets_config.expanduser().resolve()),
        "head_leaves": len(lora_snapshot["head"]),
        "shared_lora_leaves": len(lora_snapshot["shared_lora"]),
        "locon_up_leaves": len(combo_snapshot["locon_up"]),
        "head_equal": head_equal,
        "shared_lora_equal": shared_lora_equal,
        "locon_up_zero": locon_zero,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not (head_equal and shared_lora_equal and locon_zero):
        raise RuntimeError("Adapter initialization parity audit failed.")


if __name__ == "__main__":
    main()
