#!/usr/bin/env python
"""Export a finetuned adapter checkpoint as a merged no-adapter JAX checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp

from alphagenome_ft import load_checkpoint
from alphagenome_ft.finetune import (
    load_targets_config,
    prepare_head_specs,
    register_predefined_heads,
    validate_head_specs,
)
from alphagenome_ft.lora import ADAPTER_LEAF_NAMES
from scripts.run_humanbraindev_finetune import build_targets_config, discover_bigwigs


DEFAULT_SOURCE = Path(
    "outputs/precision_adapter_compare/"
    "20260627_232826_full_valdiffpearson_wandb/"
    "jax_default_lora_locon/best"
)
DEFAULT_BIGWIG_DIR = Path(
    "/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _key_name(key: Any) -> str:
    return str(getattr(key, "key", key))


def _remove_adapter_leaves(tree):
    if not isinstance(tree, dict):
        return tree
    out = {}
    for key, value in tree.items():
        if str(key) in ADAPTER_LEAF_NAMES:
            continue
        out[key] = _remove_adapter_leaves(value)
    return out


def _fold_lora_and_locon(params, *, lora_alpha: float, lora_rank: int, locon_alpha: float, locon_rank: int):
    """Fold adapter leaves into base leaves where representable.

    Dense LoRA is exact: ``w += A @ B * alpha/rank``.

    LoCon over AlphaGenome StandardizedConv1D is approximate because the base
    module standardizes raw conv weights at runtime; an arbitrary LoCon conv
    kernel is not generally representable by only changing the raw standardized
    conv weight. We still fold the composed conv kernel into ``w`` so the
    verification step can measure the approximation error.
    """
    stats = {
        "folded_lora": 0,
        "folded_locon_approx": 0,
        "dropped_adapter_leaves": 0,
    }

    def visit(node):
        if not isinstance(node, dict):
            return node
        updated = {key: visit(value) for key, value in node.items()}
        keys = {str(key): key for key in updated}

        if {"w", "lora_a", "lora_b"}.issubset(keys):
            w_key = keys["w"]
            a_key = keys["lora_a"]
            b_key = keys["lora_b"]
            w = updated[w_key]
            a = updated[a_key].astype(jnp.float32)
            b = updated[b_key].astype(jnp.float32)
            delta = (a @ b) * (float(lora_alpha) / float(lora_rank))
            updated[w_key] = (w.astype(jnp.float32) + delta).astype(w.dtype)
            stats["folded_lora"] += 1

        if {"w", "locon_down_w", "locon_up_w"}.issubset(keys):
            w_key = keys["w"]
            down_key = keys["locon_down_w"]
            up_key = keys["locon_up_w"]
            w = updated[w_key]
            down = updated[down_key].astype(jnp.float32)
            up = updated[up_key].astype(jnp.float32)
            if down.ndim == 3 and up.ndim == 3 and up.shape[0] == 1:
                delta = jnp.einsum("kir,rO->kiO", down, up[0])
                delta = delta * (float(locon_alpha) / float(locon_rank))
                if delta.shape == w.shape:
                    updated[w_key] = (w.astype(jnp.float32) + delta).astype(w.dtype)
                    stats["folded_locon_approx"] += 1

        for leaf in ADAPTER_LEAF_NAMES:
            if leaf in keys:
                del updated[keys[leaf]]
                stats["dropped_adapter_leaves"] += 1
        return updated

    return visit(params), stats


def _count_adapter_leaves(params) -> int:
    count = 0

    def visit(path, value):
        nonlocal count
        if path and _key_name(path[-1]) in ADAPTER_LEAF_NAMES:
            count += 1

    jax.tree_util.tree_map_with_path(visit, params)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("outputs/quant_ablation/merged_jax_default_lora_locon"),
    )
    parser.add_argument("--base-checkpoint-path", type=Path, default=None)
    parser.add_argument("--model-version", default="all_folds")
    parser.add_argument("--window-size", type=int, default=131072)
    parser.add_argument("--bigwig-dir", type=Path, default=DEFAULT_BIGWIG_DIR)
    parser.add_argument("--head-id", default="humanbraindev_atac")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_checkpoint.expanduser().resolve()
    output = args.output_checkpoint.expanduser().resolve()
    config = _load_json(source / "config.json")
    lora_cfg = dict(config.get("backbone_lora_config") or {})
    locon_cfg = dict(config.get("backbone_locon_config") or {})
    if not lora_cfg:
        raise ValueError(f"Source checkpoint has no backbone_lora_config: {source}")

    bigwigs = discover_bigwigs(args.bigwig_dir.expanduser().resolve())
    targets_config = load_targets_config(build_targets_config(bigwigs, args.head_id))
    head_specs = prepare_head_specs(targets_config, organism="HOMO_SAPIENS")
    validate_head_specs(head_specs)
    register_predefined_heads(head_specs)

    model = load_checkpoint(
        source,
        base_model_version=args.model_version,
        base_checkpoint_path=(
            args.base_checkpoint_path.expanduser().resolve()
            if args.base_checkpoint_path is not None
            else None
        ),
        init_seq_len=args.window_size,
    )
    before = _count_adapter_leaves(model._params)
    folded_params, fold_stats = _fold_lora_and_locon(
        model._params,
        lora_alpha=float(lora_cfg.get("alpha", 1.0)),
        lora_rank=int(lora_cfg.get("rank", 1)),
        locon_alpha=float(locon_cfg.get("alpha", 1.0)),
        locon_rank=int(locon_cfg.get("rank", 1)),
    )
    model._params = _remove_adapter_leaves(folded_params)
    model._state = _remove_adapter_leaves(model._state)
    model._backbone_lora_config = None
    model._backbone_locon_config = None
    after = _count_adapter_leaves(model._params)

    output.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(output, save_full_model=True, save_lora_adapters=False)
    metadata = {
        "source_checkpoint": str(source),
        "output_checkpoint": str(output),
        "adapter_leaves_before": before,
        "adapter_leaves_after": after,
        "fold_stats": fold_stats,
        "lora_fold_exact": True,
        "locon_fold_exact": False,
        "locon_note": (
            "LoCon over StandardizedConv1D is folded into raw conv weights as an "
            "approximation; verify metrics against the adapter checkpoint."
        ),
    }
    with (output / "merge_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
