#!/usr/bin/env python
"""Attribute Torch CUDA memory peaks across AlphaGenome inference regions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.run_quant_ablation import (
    CachedTorchDataModule,
    TORCH_TRUE_QUANT_STRATEGIES,
    apply_torch_quant_policy,
    _load_torch_checkpoint_payload,
    _split_torch_strategy,
)
from scripts.torch_effective_conv import (
    jax_effective_paths_to_torch,
    materialize_effective_convs,
    materialize_standardized_convs,
)


def _mib(value: int | float) -> float:
    return float(value) / 1024**2


class MemoryTrace:
    def __init__(self, device: torch.device):
        self.device = device
        self.rows: list[dict[str, Any]] = []

    def measure(self, name: str, fn: Callable[[], Any]) -> Any:
        torch.cuda.synchronize(self.device)
        before_alloc = torch.cuda.memory_allocated(self.device)
        before_reserved = torch.cuda.memory_reserved(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        after_alloc = torch.cuda.memory_allocated(self.device)
        after_reserved = torch.cuda.memory_reserved(self.device)
        peak_alloc = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        self.rows.append(
            {
                "stage": name,
                "elapsed_sec": elapsed,
                "alloc_before_mib": _mib(before_alloc),
                "alloc_after_mib": _mib(after_alloc),
                "alloc_delta_mib": _mib(after_alloc - before_alloc),
                "peak_alloc_mib": _mib(peak_alloc),
                "temp_above_before_mib": _mib(max(0, peak_alloc - before_alloc)),
                "reserved_before_mib": _mib(before_reserved),
                "reserved_after_mib": _mib(after_reserved),
                "peak_reserved_mib": _mib(peak_reserved),
            }
        )
        return result


def _load_model(args: argparse.Namespace, device: torch.device):
    torch_repo = args.torch_repo.expanduser().resolve()
    src_dir = torch_repo / "src"
    for path in (str(src_dir), str(torch_repo)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.model import AlphaGenome
    from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
    from alphagenome_pytorch.extensions.finetuning.transfer import add_head, remove_all_heads

    base_strategy, precompute_standardized_convs = _split_torch_strategy(args.strategy)
    dtype_policy = DtypePolicy.full_float32()
    # Match run_quant_ablation.evaluate_torch.
    if base_strategy == "bf16_params" or base_strategy in TORCH_TRUE_QUANT_STRATEGIES:
        dtype_policy = DtypePolicy.aggressive_bfloat16()

    weights_path = args.torch_weights.expanduser().resolve()
    payload = _load_torch_checkpoint_payload(weights_path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Torch checkpoint lacks model_state_dict: {weights_path}")

    data_module = CachedTorchDataModule(
        target_cache_dir=args.target_cache_dir,
        fasta_path=args.fasta_path,
        head_id=args.head_id,
        batch_size=args.batch_size,
    )
    head_specs = data_module.head_specs
    head_name = head_specs[0].head_id

    assay_type = payload.get("assay_type", "atac")
    resolutions = tuple(int(res) for res in payload.get("resolutions", (128,)))
    num_organisms = int(payload.get("num_organisms", 1))
    model = AlphaGenome(num_organisms=2, dtype_policy=dtype_policy)
    model = remove_all_heads(model)
    add_head(
        model,
        head_name,
        create_finetuning_head(
            assay_type,
            n_tracks=len(head_specs[0].tracks),
            resolutions=resolutions,
            num_organisms=num_organisms,
        ),
    )
    effective_torch_modules = tuple(payload.get("torch_effective_conv_modules") or ())
    if not effective_torch_modules:
        effective_jax_paths = tuple(str(path) for path in payload.get("backbone_effective_conv_paths", ()))
        effective_torch_modules = jax_effective_paths_to_torch(effective_jax_paths)
    materialize_effective_convs(model, effective_torch_modules)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()

    standardized_conv_stats: dict[str, Any] = {"standardized_convs_materialized": 0}
    if precompute_standardized_convs:
        materialized = materialize_standardized_convs(model)
        standardized_conv_stats = {"standardized_convs_materialized": len(materialized)}
    quant_stats = apply_torch_quant_policy(model, base_strategy)
    quant_stats.update(standardized_conv_stats)
    quant_stats["strategy"] = args.strategy
    return model, data_module, head_name, quant_stats


def _first_batch(data_module: CachedTorchDataModule, split: str, head_name: str, device: torch.device):
    batch_np = next(iter(data_module.iter_batches(split, shuffle=False)))
    seq = torch.as_tensor(batch_np["sequences"], device=device, dtype=torch.float32)
    org = torch.zeros((seq.shape[0],), device=device, dtype=torch.long)
    return seq, org


def _encoder_staged(
    model: Any,
    dna_sequence: torch.Tensor,
    trace: MemoryTrace,
    *,
    store_intermediates: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    encoder = model.encoder
    intermediates: dict[str, torch.Tensor] = {}

    x = trace.measure("encoder.input.transpose_nlc_to_ncl", lambda: dna_sequence.transpose(1, 2))
    conv1 = trace.measure("encoder.dna_embedder.conv1", lambda: encoder.dna_embedder.conv1(x))
    block_out = trace.measure("encoder.dna_embedder.block", lambda: encoder.dna_embedder.block(conv1))
    x = trace.measure("encoder.dna_embedder.residual_add", lambda: conv1 + block_out)
    del conv1, block_out
    if store_intermediates:
        intermediates["bin_size_1"] = x
    x = trace.measure("encoder.pool.bin_size_1_to_2", lambda: encoder.pool(x))

    for idx, block in enumerate(encoder.down_blocks):
        x = trace.measure(f"encoder.down_blocks.{idx}", lambda block=block, x=x: block(x))
        bin_size = encoder.bin_sizes[idx]
        if store_intermediates:
            intermediates[f"bin_size_{bin_size}"] = x
        x = trace.measure(f"encoder.pool.bin_size_{bin_size}_to_next", lambda x=x: encoder.pool(x))

    return x, intermediates


def run_attribution(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory attribution.")
    device = torch.device("cuda")
    model, data_module, head_name, quant_stats = _load_model(args, device)
    seq, org = _first_batch(data_module, args.split, head_name, device)
    trace = MemoryTrace(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        outputs = trace.measure(
            "eval.forward_exact_128bp",
            lambda: model(
                seq,
                org,
                heads=(head_name,),
                resolutions=(128,),
                return_scaled_predictions=False,
                channels_last=True,
            ),
        )
        pred = trace.measure("eval.pred_float_for_metrics", lambda: outputs[head_name][128].float())
        del pred, outputs
        torch.cuda.empty_cache()

        dna_sequence = trace.measure("input.cast_to_compute", lambda: model.dtype_policy.cast_to_compute(seq))
        trunk, intermediates = trace.measure("encoder.total", lambda: model.encoder(dna_sequence))
        del trunk, intermediates
        torch.cuda.empty_cache()
        trunk, intermediates = _encoder_staged(
            model,
            dna_sequence,
            trace,
            store_intermediates=not bool(quant_stats.get("encoder_no_intermediates")),
        )

        trunk = trace.measure("trunk.transpose_ncl_to_nlc", lambda: trunk.transpose(1, 2).contiguous())
        org_emb = trace.measure("organism_embed", lambda: model.organism_embed(org).unsqueeze(1))
        trunk = trace.measure("add_organism_embedding", lambda: trunk + org_emb)

        pair_x = None
        for idx, block in enumerate(model.tower.blocks):
            if block["pair_update"] is not None:
                pair_x = trace.measure(
                    f"tower.block{idx}.pair_update",
                    lambda block=block, trunk=trunk, pair_x=pair_x: block["pair_update"](
                        trunk, pair_x, compute_dtype=model.dtype_policy.compute_dtype
                    ),
                )
            mha_bias = trace.measure(
                f"tower.block{idx}.attention_bias",
                lambda block=block, pair_x=pair_x: block["attn_bias"](pair_x),
            )
            mha_out = trace.measure(
                f"tower.block{idx}.mha",
                lambda block=block, trunk=trunk, mha_bias=mha_bias: block["mha"](
                    trunk, mha_bias, compute_dtype=model.dtype_policy.compute_dtype
                ),
            )
            trunk = trace.measure(f"tower.block{idx}.mha_residual_add", lambda trunk=trunk, mha_out=mha_out: trunk + mha_out)
            del mha_out, mha_bias
            mlp_out = trace.measure(f"tower.block{idx}.mlp", lambda block=block, trunk=trunk: block["mlp"](trunk))
            trunk = trace.measure(f"tower.block{idx}.mlp_residual_add", lambda trunk=trunk, mlp_out=mlp_out: trunk + mlp_out)
            del mlp_out

        trunk_ncl = trace.measure("trunk.transpose_nlc_to_ncl", lambda: trunk.transpose(1, 2).contiguous())
        embeddings_128bp = trace.measure(
            "embedder_128bp",
            lambda: model.embedder_128bp(trunk_ncl, org, channels_last=False),
        )
        del intermediates
        embeddings_pair = trace.measure("embedder_pair", lambda: model.embedder_pair(pair_x, org))
        embeddings_dict = {128: embeddings_128bp}
        head = model.heads[head_name]
        outputs = trace.measure(
            f"head.{head_name}",
            lambda: head(embeddings_dict, org, return_scaled=False, channels_last=True),
        )
        trace.measure("output.sync", lambda: outputs[128].shape)

    return {
        "strategy": args.strategy,
        "batch_size": args.batch_size,
        "split": args.split,
        "head": head_name,
        "quantization": quant_stats,
        "rows": trace.rows,
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "memory_attribution.json").open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    rows = result["rows"]
    if rows:
        with (output_dir / "memory_attribution.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    top = sorted(rows, key=lambda row: row["peak_alloc_mib"], reverse=True)[:20]
    lines = [
        "# Torch Memory Attribution",
        "",
        f"- strategy: `{result['strategy']}`",
        f"- batch_size: `{result['batch_size']}`",
        f"- split: `{result['split']}`",
        "",
        "| rank | stage | peak alloc MiB | alloc before MiB | alloc after MiB | temp above before MiB | peak reserved MiB | sec |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(top, start=1):
        lines.append(
            "| {rank} | `{stage}` | {peak:.0f} | {before:.0f} | {after:.0f} | {temp:.0f} | {reserved:.0f} | {sec:.3f} |".format(
                rank=rank,
                stage=row["stage"],
                peak=row["peak_alloc_mib"],
                before=row["alloc_before_mib"],
                after=row["alloc_after_mib"],
                temp=row["temp_above_before_mib"],
                reserved=row["peak_reserved_mib"],
                sec=row["elapsed_sec"],
            )
        )
    (output_dir / "memory_attribution.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="bf16_triton_conv_stdconv_effective")
    parser.add_argument("--torch-repo", type=Path, default=Path(__file__).resolve().parents[2] / "alphagenome-pytorch")
    parser.add_argument("--torch-weights", type=Path, required=True)
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--fasta-path", type=Path, default=Path("/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa"))
    parser.add_argument("--head-id", default="humanbraindev_atac")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_attribution(args)
    write_outputs(result, args.output_dir.expanduser().resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
