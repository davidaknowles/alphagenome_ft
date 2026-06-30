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

## Strategy Descriptions

| Strategy | Description |
|---|---|
| `default` | No additional quantization or dtype conversion beyond the backend's normal merged-checkpoint evaluation path. |
| `bf16_params` | Evaluate with eligible floating-point parameters cast to `bfloat16`. In Torch, `Converted = -1` is a sentinel for whole-model dtype conversion rather than a layer count. |
| `fp8_linear_conservative` | JAX: simulated FP8 roundtrip for large non-sensitive linear weights only. Torch: TorchAO FP8 conversion for eligible linear modules, skipping heads, norms, adapters, LoRA, and LoCon. |
| `fp8_linear_aggressive` | JAX: simulated FP8 roundtrip for all non-sensitive linear weights. Torch: same TorchAO FP8 conversion policy as `fp8_linear_conservative`; the suffix was retained for naming parity but does not change Torch layer selection. |
| `fp8_1x1conv` | JAX only. Simulated FP8 roundtrip for non-sensitive linear weights plus 1x1 convolution weights. |
| `fp8_late_conv` | JAX only. Simulated FP8 roundtrip for non-sensitive linear weights plus wider convolution weights in late downsampling blocks 4 and 5. |
| `nf4_linear_conservative` | JAX: simulated NF4 block quantize/dequantize roundtrip for large non-sensitive linear weights only. Torch: simulated NF4 roundtrip for eligible linear modules, skipping heads, norms, embeddings, adapters, LoRA, and LoCon. |
| `nf4_linear_aggressive` | JAX: simulated NF4 roundtrip for all non-sensitive linear weights. Torch: same eligible-linear NF4 policy as `nf4_linear_conservative`; the suffix was retained for naming parity but does not change Torch layer selection. |
| `nf4_1x1conv` | Simulated NF4 roundtrip for eligible linear weights plus 1x1 convolution weights. |
| `nf4_late_conv` | Simulated NF4 roundtrip for eligible linear weights plus wider convolution weights in late downsampling blocks 4 and 5. |
| `nf4_all_conv` | Simulated NF4 roundtrip for eligible linear, 1x1 convolution, and wider convolution weights, excluding sensitive/stem convolution paths. |
| `nvfp4_weight_only` | Torch only. TorchAO NVFP4 weight-only conversion for frozen eligible linear modules, skipping heads, norms, adapters, LoRA, and LoCon. This uses compact TorchAO weight storage rather than only simulating quantize/dequantize. |

## Results

JAX used batch size 8 and Torch used batch size 1. `Examples/s` is therefore the relevant cross-backend throughput column; `Batches/s` is retained only for reproducibility.

| Backend | Strategy | Valid diff Pearson | Test diff Pearson | Batch size | Batches/s | Examples/s | Elapsed s | Avg GPU % | Max VRAM GiB | Converted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JAX | `bf16_params` | 0.7910 | 0.8036 | 8 | 1.09 | 8.68 | 249.7 | 24.7 | 16.6 | 138 |
| JAX | `default` | 0.7910 | 0.8035 | 8 | 1.08 | 8.63 | 251.3 | 26.5 | 16.6 | 0 |
| JAX | `fp8_1x1conv` | 0.7910 | 0.8034 | 8 | 1.10 | 8.77 | 247.1 | 23.6 | 16.6 | 110 |
| JAX | `fp8_late_conv` | 0.7910 | 0.8034 | 8 | 1.11 | 8.84 | 245.2 | 23.5 | 16.6 | 110 |
| JAX | `fp8_linear_aggressive` | 0.7910 | 0.8034 | 8 | 1.63 | 13.08 | 165.8 | 34.5 | 16.6 | 110 |
| JAX | `fp8_linear_conservative` | 0.7910 | 0.8034 | 8 | 1.69 | 13.52 | 160.3 | 36.5 | 16.6 | 61 |
| JAX | `nf4_1x1conv` | 0.7885 | 0.8001 | 8 | 1.12 | 9.00 | 240.9 | 25.0 | 16.6 | 110 |
| JAX | `nf4_all_conv` | 0.7885 | 0.8000 | 8 | 1.63 | 13.02 | 166.5 | 32.4 | 16.6 | 110 |
| JAX | `nf4_late_conv` | 0.7885 | 0.8000 | 8 | 1.13 | 9.07 | 239.1 | 23.9 | 16.6 | 110 |
| JAX | `nf4_linear_aggressive` | 0.7885 | 0.8000 | 8 | 1.09 | 8.71 | 249.0 | 23.1 | 16.6 | 110 |
| JAX | `nf4_linear_conservative` | 0.7885 | 0.7999 | 8 | 1.08 | 8.60 | 252.0 | 23.8 | 16.6 | 61 |
| Torch | `bf16_params` | 0.7988 | 0.8102 | 1 | 10.47 | 10.47 | 206.4 | 34.0 | 3.8 | -1 |
| Torch | `default` | 0.7992 | 0.8103 | 1 | 9.22 | 9.22 | 234.4 | 53.5 | 4.8 | 0 |
| Torch | `fp8_linear_aggressive` | 0.7993 | 0.8102 | 1 | 5.65 | 5.65 | 382.4 | 33.0 | 4.9 | 109 |
| Torch | `fp8_linear_conservative` | 0.7993 | 0.8102 | 1 | 8.01 | 8.01 | 269.8 | 46.5 | 4.8 | 109 |
| Torch | `nf4_1x1conv` | 0.7980 | 0.8089 | 1 | 6.21 | 6.21 | 347.9 | 36.4 | 4.9 | 119 |
| Torch | `nf4_all_conv` | 0.7963 | 0.8078 | 1 | 8.42 | 8.42 | 256.7 | 45.1 | 4.8 | 135 |
| Torch | `nf4_late_conv` | 0.7979 | 0.8090 | 1 | 6.23 | 6.23 | 347.0 | 36.6 | 4.9 | 113 |
| Torch | `nf4_linear_aggressive` | 0.7980 | 0.8092 | 1 | 6.12 | 6.12 | 353.2 | 35.6 | 4.8 | 109 |
| Torch | `nf4_linear_conservative` | 0.7980 | 0.8092 | 1 | 6.12 | 6.12 | 353.2 | 34.2 | 4.8 | 109 |
| Torch | `nvfp4_weight_only` | 0.7983 | 0.8102 | 1 | 5.67 | 5.67 | 381.4 | 36.8 | 4.2 | 109 |

## Conclusions

- Exact effective LoCon merge is working. JAX default reaches `0.8035` test differential Pearson and Torch default reaches `0.8103`.
- FP8 is effectively accuracy-neutral in both backends for these strategies. JAX FP8 stays within about `0.0001` test differential Pearson of default; Torch FP8 is within about `0.0001`.
- NF4 has a small but measurable accuracy cost. JAX NF4 is about `0.0035` below JAX default on test differential Pearson; Torch NF4 ranges from about `0.0011` to `0.0025` below Torch default.
- Torch `nvfp4_weight_only` is close to default accuracy on this eval: `0.8102` test differential Pearson versus `0.8103` for default, with 109 linears converted.
- Torch has much higher `Batches/s` because it used batch size 1 while JAX used batch size 8. On the comparable `Examples/s` metric, Torch default is only modestly faster than JAX default (`9.22` versus `8.63` examples/s). Some JAX quantized strategies are faster than both defaults.
- Max VRAM is almost unchanged across quantization strategies within each backend because these are inference/evaluation peaks, not pure model-weight footprints. Activations, temporary workspaces, framework allocators, cached buffers, and data pipeline buffers dominate the measured peak; changing storage precision for a subset of weights only moves a smaller component of total memory. JAX also appears to allocate a stable evaluation buffer footprint near `16.6` GiB across strategies, while Torch stays around `4-5` GiB.
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
