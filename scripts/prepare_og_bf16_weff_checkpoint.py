#!/usr/bin/env python
"""Export an original AlphaGenome Torch checkpoint with bf16 params and W_eff convs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OG_WEIGHTS = Path(
    "/gpfs/commons/home/daknowles/projects/mpragent/outputs/models/alphagenome/model_all_folds.safetensors"
)


def _add_torch_repo(path: Path) -> None:
    path = path.expanduser().resolve()
    for entry in (path / "src", path):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


def _save_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        str(path),
        metadata={"alphagenome_fp4_metadata": json.dumps(metadata, sort_keys=True)},
    )
    with path.with_suffix(path.suffix + ".json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _tensor_bytes(tensors: dict[str, torch.Tensor]) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors.values()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--og-weights", type=Path, default=DEFAULT_OG_WEIGHTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/og_low_vram/alphagenome_og_bf16_weff.safetensors"),
    )
    parser.add_argument(
        "--torch-repo",
        type=Path,
        default=Path("/gpfs/commons/home/daknowles/projects/alphagenome-pytorch"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _add_torch_repo(args.torch_repo)

    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.model import AlphaGenome
    from scripts.torch_effective_conv import materialize_standardized_convs

    og_weights = args.og_weights.expanduser().resolve()
    print(f"Loading full OG weights from {og_weights}")
    model = AlphaGenome.from_pretrained(
        og_weights,
        dtype_policy=DtypePolicy.full_float32(),
        device="cpu",
    )
    model.eval()

    materialized = materialize_standardized_convs(model)
    model.to(dtype=torch.bfloat16)
    state_dict = model.state_dict()

    metadata = {
        "checkpoint_kind": "alphagenome_og_bf16_weff",
        "source_og_weights": str(og_weights),
        "source_torch_repo": str(args.torch_repo.expanduser().resolve()),
        "dtype": "bfloat16",
        "standardized_convs_materialized": len(materialized),
        "standardized_conv_paths": list(materialized),
        "state_dict_bytes": _tensor_bytes(state_dict),
        "note": (
            "Derived from the original AlphaGenome all-folds Torch weights. "
            "StandardizedConv1d modules are replaced by direct W_eff Conv1d modules, "
            "then floating state is stored as bfloat16."
        ),
    }
    _save_safetensors(args.output, state_dict, metadata)
    print(
        f"Saved {args.output.expanduser().resolve()} "
        f"({len(state_dict)} tensors, {metadata['state_dict_bytes'] / 1024**3:.2f} GiB, "
        f"W_eff convs={len(materialized)})"
    )


if __name__ == "__main__":
    main()
