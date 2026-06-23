#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import jax

from alphagenome_ft import BackboneLoRAConfig, create_model_with_heads, lora, parameter_utils
from alphagenome_ft.finetune.config import prepare_head_specs
from alphagenome_ft.finetune.train import register_predefined_heads


def _dtype_counts(tree) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for leaf in jax.tree_util.tree_leaves(tree):
        counts[str(getattr(leaf, "dtype", None))] += 1
    return dict(sorted(counts.items()))


def _nbytes(tree) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        total += getattr(leaf, "nbytes", 0)
    return total


def _register_probe_head() -> str:
    config = {
        "heads": [
            {
                "id": "precision_probe_atac",
                "source": "predefined",
                "kind": "atac",
                "resolutions": [1, 128],
                "apply_squashing": False,
                "targets": [{"path": "data/example_rna_rep1_chr1.bw", "label": "probe"}],
            }
        ]
    }
    specs = prepare_head_specs(config, organism="HOMO_SAPIENS")
    register_predefined_heads(specs)
    return specs[0].head_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["heads_only", "lora_bf16", "lora_fp8"])
    parser.add_argument("--init-seq-len", type=int, default=131072)
    parser.add_argument("--base-param-dtype", default="float32")
    parser.add_argument("--lora-param-dtype", default="float32")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    print(f"mode={args.mode}")
    print(f"checkpoint_path={checkpoint_path}")
    print(f"checkpoint_exists={checkpoint_path.exists()}")
    print(f"devices={jax.local_devices()}")

    head_name = _register_probe_head()
    lora_config = None
    detach_backbone = True
    if args.mode == "lora_bf16":
        detach_backbone = False
        lora_config = BackboneLoRAConfig(
            rank=16,
            alpha=16.0,
            fp8_enabled=False,
            base_param_dtype=args.base_param_dtype,
            lora_param_dtype=args.lora_param_dtype,
            activation_dtype="bfloat16",
            base_compute_dtype="bfloat16",
            lora_compute_dtype="bfloat16",
        )
    elif args.mode == "lora_fp8":
        detach_backbone = False
        lora_config = BackboneLoRAConfig(
            rank=16,
            alpha=16.0,
            fp8_enabled=True,
            base_param_dtype=args.base_param_dtype,
            lora_param_dtype=args.lora_param_dtype,
            activation_dtype="bfloat16",
            base_compute_dtype="bfloat16",
            lora_compute_dtype="fp8",
        )

    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=checkpoint_path,
        detach_backbone=detach_backbone,
        init_seq_len=args.init_seq_len,
        backbone_lora_config=lora_config,
        runtime_backbone_param_dtype=args.base_param_dtype,
    )
    print(f"param_leaves={len(jax.tree_util.tree_leaves(model._params))}")
    print(f"param_dtype_counts={_dtype_counts(model._params)}")
    print(f"param_nbytes={_nbytes(model._params)}")
    print(f"state_leaves={len(jax.tree_util.tree_leaves(model._state))}")
    print(f"state_dtype_counts={_dtype_counts(model._state)}")
    print(f"state_nbytes={_nbytes(model._state)}")
    print(f"total_params={parameter_utils.count_parameters(model._params)}")
    if lora_config is not None:
        lora_paths = lora.get_lora_parameter_paths(model._params)
        print(f"lora_paths={len(lora_paths)}")
        print(f"lora_params={lora.count_lora_parameters(model._params)}")
        print(f"resolved_lora_compute_dtype={lora_config.resolved_lora_compute_dtype()}")
        if not lora_paths:
            raise RuntimeError("No LoRA parameters were created.")
    print("RESULT=OK")


if __name__ == "__main__":
    main()
