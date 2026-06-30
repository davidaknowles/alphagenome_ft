# Quantization Ablation After Exact Effective LoCon Merge

Date: 2026-06-29

Branch: `ablate-quant`

## Inputs

- Merged JAX checkpoint: `outputs/quant_ablation/20260629_effective_merge/merged_jax_default_lora_locon_eff`
- Merged Torch checkpoint: `outputs/quant_ablation/20260629_effective_merge/merged_torch_default_lora_locon_eff.pth`
- Final JAX metrics: `outputs/quant_ablation/20260629_effective_merge_fullrerun/jax_validtest/`
- Final Torch metrics: `outputs/quant_ablation/20260629_effective_merge_fullrerun/torch_b6k/`
- Target cache: `/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/alphagenome_target_cache/humanbraindev_atac_w131072_float16`

## Jobs

- JAX valid/test array: `18203521`, 11/11 completed, exit `0`.
- Torch Blackwell array: `18203793`, 10/10 completed, exit `0`.

Torch was run on RTX PRO 6000 Blackwell nodes using `~/venv/torchfix` with PyTorch/TorchAO CUDA 12.8 wheels. The older CUDA 12.6 Torch wheel did not support `sm_120`.

## Metric

Primary accuracy metric is `differential_pearson_r`: subtract cell-type mean, then locus/bin mean, then compute Pearson correlation over the residualized predictions and targets. The loss values are included for provenance but are not the primary comparison, and are not directly comparable across the JAX and Torch implementations because the output scaling/loss conventions differ.

## Results

| Backend | Strategy | Valid diff Pearson | Test diff Pearson | Valid loss | Test loss | Batches/s | Elapsed s | Avg GPU % | Max VRAM GiB | Converted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JAX | `bf16_params` | 0.7910 | 0.8036 | 29.4886 | 33.1894 | 1.09 | 249.7 | 24.7 | 16.6 | 138 |
| JAX | `default` | 0.7910 | 0.8035 | 29.4885 | 33.1894 | 1.08 | 251.3 | 26.5 | 16.6 | 0 |
| JAX | `fp8_1x1conv` | 0.7910 | 0.8034 | 29.4910 | 33.1919 | 1.10 | 247.1 | 23.6 | 16.6 | 110 |
| JAX | `fp8_late_conv` | 0.7910 | 0.8034 | 29.4910 | 33.1917 | 1.11 | 245.2 | 23.5 | 16.6 | 110 |
| JAX | `fp8_linear_aggressive` | 0.7910 | 0.8034 | 29.4910 | 33.1917 | 1.63 | 165.8 | 34.5 | 16.6 | 110 |
| JAX | `fp8_linear_conservative` | 0.7910 | 0.8034 | 29.4908 | 33.1915 | 1.69 | 160.3 | 36.5 | 16.6 | 61 |
| JAX | `nf4_1x1conv` | 0.7885 | 0.8001 | 29.4961 | 33.1949 | 1.12 | 240.9 | 25.0 | 16.6 | 110 |
| JAX | `nf4_all_conv` | 0.7885 | 0.8000 | 29.4961 | 33.1949 | 1.63 | 166.5 | 32.4 | 16.6 | 110 |
| JAX | `nf4_late_conv` | 0.7885 | 0.8000 | 29.4962 | 33.1950 | 1.13 | 239.1 | 23.9 | 16.6 | 110 |
| JAX | `nf4_linear_aggressive` | 0.7885 | 0.8000 | 29.4961 | 33.1950 | 1.09 | 249.0 | 23.1 | 16.6 | 110 |
| JAX | `nf4_linear_conservative` | 0.7885 | 0.7999 | 29.4974 | 33.1969 | 1.08 | 252.0 | 23.8 | 16.6 | 61 |
| Torch | `bf16_params` | 0.7988 | 0.8102 | 256768.6748 | 317812.5718 | 10.47 | 206.4 | 34.0 | 3.8 | -1 |
| Torch | `default` | 0.7992 | 0.8103 | 262550.0170 | 324472.3350 | 9.22 | 234.4 | 53.5 | 4.8 | 0 |
| Torch | `fp8_linear_aggressive` | 0.7993 | 0.8102 | 261410.7276 | 323596.0346 | 5.65 | 382.4 | 33.0 | 4.9 | 109 |
| Torch | `fp8_linear_conservative` | 0.7993 | 0.8102 | 261410.7276 | 323596.0346 | 8.01 | 269.8 | 46.5 | 4.8 | 109 |
| Torch | `nf4_1x1conv` | 0.7980 | 0.8089 | 269195.7652 | 333795.0420 | 6.21 | 347.9 | 36.4 | 4.9 | 119 |
| Torch | `nf4_all_conv` | 0.7963 | 0.8078 | 262603.5756 | 326163.5178 | 8.42 | 256.7 | 45.1 | 4.8 | 135 |
| Torch | `nf4_late_conv` | 0.7979 | 0.8090 | 267024.2645 | 330721.5067 | 6.23 | 347.0 | 36.6 | 4.9 | 113 |
| Torch | `nf4_linear_aggressive` | 0.7980 | 0.8092 | 267246.3198 | 331036.2838 | 6.12 | 353.2 | 35.6 | 4.8 | 109 |
| Torch | `nf4_linear_conservative` | 0.7980 | 0.8092 | 267246.3198 | 331036.2838 | 6.12 | 353.2 | 34.2 | 4.8 | 109 |
| Torch | `nvfp4_weight_only` | 0.7983 | 0.8102 | 234537.6131 | 290844.8353 | 5.67 | 381.4 | 36.8 | 4.2 | 109 |

## Conclusions

- Exact effective LoCon merge is working. JAX default reaches `0.8035` test differential Pearson and Torch default reaches `0.8103`.
- FP8 is effectively accuracy-neutral in both backends for these strategies. JAX FP8 stays within about `0.0001` test differential Pearson of default; Torch FP8 is within about `0.0001`.
- NF4 has a small but measurable accuracy cost. JAX NF4 is about `0.0035` below JAX default on test differential Pearson; Torch NF4 ranges from about `0.0011` to `0.0025` below Torch default.
- Torch `nvfp4_weight_only` is close to default accuracy on this eval: `0.8102` test differential Pearson versus `0.8103` for default, with 109 linears converted.
- Torch evaluation is much faster than JAX in these runs, but uses much less VRAM. That suggests the two backends are not saturated in the same way; GPU utilization remains a performance target rather than an accuracy issue.
- Treat Torch `bf16_params` converted count `-1` as a sentinel for dtype conversion, not a layer count.

## Cleanup

Deleted deprecated/superseded run outputs:

- Pre-effective-merge JAX/Torch ablations from `20260628_212653*` and `20260629_090303_torch_full`.
- Local smoke outputs under `outputs/quant_ablation/20260629_effective_merge/local_*`.
- Valid-only JAX rerun under `outputs/quant_ablation/20260629_effective_merge_fullrerun/jax`.
- Failed old Torch run under `outputs/quant_ablation/20260629_effective_merge_fullrerun/torch`.
- Canceled L40S workaround directory `outputs/quant_ablation/20260629_effective_merge_fullrerun/torch_l40s`.
- Blackwell smoke outputs under `outputs/quant_ablation/torchfix_blackwell_smoke`.

Deleted deprecated Slurm logs for the removed runs while keeping the final JAX valid/test and Torch Blackwell logs.
