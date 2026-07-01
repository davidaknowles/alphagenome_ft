# True Quant Test Ablation Findings

Run root: `outputs/quant_ablation/20260630_155324_torch_true_quant_test_full`

Submitted Slurm array: `18470589_[0-39]`.

Status:
- Completed metric files: `40/40`.
- Current `squeue` lookup for the job returns no active entries, consistent with completion.
- No `Traceback`, `RuntimeError`, `CUDA out of memory`, or `error:` signatures were found in `logs/quant_ablation/ag-torch-quant_18470589_*.err`.
- Split/head: `test` / `humanbraindev_atac`.
- Test examples per run: `1055`.
- Raw full table: `quant_ablation_results.md` in the run root.

## Main Takeaways

- Best overall throughput was `bf16_params` at batch `8`: `11.81 examples/s`, torch peak allocated `13122 MiB`.
- Best <=4 GiB torch-allocated run was `torchao_nvfp4_weight_only_linear` at batch `2`: `11.01 examples/s`, torch peak allocated `3628 MiB`.
- Compared with default batch 1 (`8.20 examples/s`, `3902 MiB`), NF4/NVFP4 batch 2 gives about `1.34x` examples/s at slightly lower torch allocation (`~3621 MiB`).
- Batch size improves throughput up to roughly batch 8-12, but memory scales almost linearly with batch size. For low VRAM, batch 2 is the strongest compromise in this sweep.
- The `*_tower_linear` and unrestricted linear variants converted the same number of linears (`109`) and produced nearly identical metrics here, so the include filter did not materially change this checkpoint/model path.

## Best Row Per Strategy

| strategy | quant backend | converted | best batch | examples/s | torch alloc MiB | torch reserved MiB | loss | diff Pearson |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| default | none | 0 | 12 | 9.42 | 24678 | 34730 | 325934.2 | 0.8103 |
| bf16_params | none | -1 | 8 | 11.81 | 13122 | 14712 | 318083.2 | 0.8102 |
| torchao_float8_tower_linear | torchao | 109 | 12 | 10.64 | 19286 | 21640 | 318707.1 | 0.8100 |
| torchao_float8_linear | torchao | 109 | 12 | 11.05 | 19286 | 21640 | 318707.1 | 0.8100 |
| torchao_nvfp4_weight_only_tower_linear | torchao/nvfp4 | 109 | 4 | 10.86 | 6879 | 11540 | 284893.0 | 0.8101 |
| torchao_nvfp4_weight_only_linear | torchao/nvfp4 | 109 | 12 | 11.51 | 19031 | 21782 | 286599.3 | 0.8101 |
| bnb_nf4_weight_only_tower_linear | bitsandbytes/nf4 | 109 | 4 | 11.16 | 6863 | 11474 | 309174.4 | 0.8094 |
| bnb_nf4_weight_only_linear | bitsandbytes/nf4 | 109 | 8 | 11.75 | 12856 | 14800 | 309753.5 | 0.8093 |

## Batch Size Sweep

Cells are `examples/s (torch alloc MiB)`.

| strategy | b1 | b2 | b4 | b8 | b12 |
|---|---:|---:|---:|---:|---:|
| default | 8.20 (3902) | 7.97 (6211) | 8.01 (10824) | 8.92 (16987) | 9.42 (24678) |
| bf16_params | 9.18 (3174) | 9.18 (3881) | 11.16 (7130) | 11.81 (13122) | 11.23 (19284) |
| torchao_float8_tower_linear | 8.85 (3175) | 10.08 (3882) | 8.93 (7131) | 8.94 (13124) | 10.64 (19286) |
| torchao_float8_linear | 8.41 (3175) | 9.90 (3882) | 8.86 (7131) | 8.87 (13124) | 11.05 (19286) |
| torchao_nvfp4_weight_only_tower_linear | 8.19 (2921) | 9.86 (3628) | 10.86 (6879) | 10.15 (12871) | 10.14 (19031) |
| torchao_nvfp4_weight_only_linear | 8.47 (2921) | 11.01 (3628) | 9.16 (6879) | 9.16 (12871) | 11.51 (19031) |
| bnb_nf4_weight_only_tower_linear | 10.57 (2906) | 10.96 (3615) | 11.16 (6863) | 10.71 (12856) | 10.72 (19016) |
| bnb_nf4_weight_only_linear | 10.08 (2906) | 10.96 (3615) | 10.09 (6863) | 11.75 (12856) | 11.75 (19016) |

## Accuracy/Metric Notes

- Differential Pearson was very stable across strategies, around `0.809-0.810` on the test split.
- MSE/loss differs by strategy and batch size. Because this is inference-only and should be batch-invariant, the batch-size-dependent loss drift is worth treating as a separate numerical/data-order issue before using loss for model quality comparisons.
- Throughput values include the full evaluation loop over cached targets and metric accumulation, not just model forward time.

## Files

- Raw collated results: `outputs/quant_ablation/20260630_155324_torch_true_quant_test_full/quant_ablation_results.md`
- Per-run metrics: `outputs/quant_ablation/20260630_155324_torch_true_quant_test_full/torch_<strategy>_batch<batch>/metrics.json`
- Slurm logs: `logs/quant_ablation/ag-torch-quant_18470589_<task>.out|err`
