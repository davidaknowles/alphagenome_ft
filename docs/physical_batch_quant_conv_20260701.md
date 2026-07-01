# Physical-Batch Quantization Ablation

Goal: maximize inference throughput while staying under a practical `<=48 GiB` actual device-memory target, without forward microbatching.

## Implementation Notes

- The Torch evaluator uses cached targets and vectorized FASTA one-hot encoding to avoid CPU-bound input preparation.
- `*_1x1conv` strategies convert eligible `Conv1d(kernel_size=1)` modules to position-wise Linear wrappers before applying true Linear quantization.
- `_stdconv_effective` strategies precompute 23 `StandardizedConv1d` weights and replace them with direct `EffectiveConv1d` modules for inference.
- Wider-conv experiments include TorchAO Conv1d-as-Conv2d intx paths and custom Triton int8 Conv1d kernels. TorchAO Conv2d intx was numerically poor on these convs; Triton preserved metrics but was slower.

## Batch 20

| strategy | notes | converted | stdconv eff | examples/s | nvidia-smi MiB | torch alloc MiB | diff Pearson |
|---|---|---:|---:|---:|---:|---:|---:|
| default | fp32-ish baseline | 0 | 0 | 9.47 | 57511 | 40063 | 0.8103 |
| bf16_params | bf16 model policy | -1 | 0 | 12.55 | 36213 | 31607 | 0.8102 |
| torchao_float8_linear | Linear float8 | 109 | 0 | 11.50 | 36213 | 31608 | 0.8101 |
| torchao_nvfp4_weight_only_all_linear_1x1conv | Linear + 1x1 conv NVFP4 | 119 | 0 | 13.28 | 36385 | 31326 | 0.8093 |
| bnb_nf4_weight_only_all_linear_1x1conv | Linear + 1x1 conv NF4 | 119 | 0 | 13.38 | 36287 | 31310 | 0.8092 |
| bf16_params_stdconv_effective | materialized std convs | -1 | 23 | 13.29 | 36507 | 31600 | 0.8102 |
| torchao_nvfp4_weight_only_all_linear_1x1conv_stdconv_effective | Linear/1x1 NVFP4 + stdconv materialization | 119 | 23 | 13.29 | 36597 | 31318 | 0.8093 |
| bnb_nf4_weight_only_all_linear_1x1conv_stdconv_effective | Linear/1x1 NF4 + stdconv materialization | 119 | 23 | 13.22 | 36521 | 31304 | 0.8092 |
| triton_int8_weight_only_effective_conv1d | 4 original EffectiveConv1d only | 4 | 0 | 12.78 | 36605 | 31568 | 0.8100 |
| triton_int8_dynamic_effective_conv1d | dynamic int8 activations, 4 EffectiveConv1d only | 4 | 0 | 12.11 | 36605 | 31568 | 0.8100 |
| triton_int8_weight_only_conv1d_stdconv_effective | all wider convs, int8 weights | 27 | 23 | 9.35 | 25685 | 17653 | 0.8101 |
| triton_int8_dynamic_conv1d_stdconv_effective | all wider convs, dynamic int8 activations | 27 | 23 | 8.52 | 25685 | 17653 | 0.8101 |

## Batch 32

| strategy | notes | converted | stdconv eff | examples/s | nvidia-smi MiB | torch alloc MiB | diff Pearson |
|---|---|---:|---:|---:|---:|---:|---:|
| default | over 48 GiB actual memory | 0 | 0 | 9.18 | 92631 | 63140 | 0.8103 |
| bf16_params | over 48 GiB actual memory | -1 | 0 | 11.33 | 66951 | 42415 | 0.8102 |
| torchao_float8_linear | over 48 GiB actual memory | 109 | 0 | 10.06 | 66935 | 42417 | 0.8101 |
| torchao_nvfp4_weight_only_linear_1x1conv | over 48 GiB actual memory | 119 | 0 | 10.38 | 67143 | 42134 | 0.8093 |
| bnb_nf4_weight_only_linear_1x1conv | over 48 GiB actual memory | 119 | 0 | 7.92 | 67043 | 42120 | 0.8092 |
| bf16_params_stdconv_effective | 10-batch smoke; over 48 GiB actual memory | -1 | 23 | 13.40 | 67263 | 42405 | 0.8109 |
| triton_int8_weight_only_conv1d_stdconv_effective | all wider convs, int8 weights | 27 | 23 | 8.93 | 43609 | 27880 | 0.8101 |
| triton_int8_dynamic_conv1d_stdconv_effective | all wider convs, dynamic int8 activations | 27 | 23 | 8.13 | 43609 | 27880 | 0.8101 |

## Takeaways

- Best throughput under actual `<=48 GiB`: `bf16_params_stdconv_effective` at batch 20.
- Best actual-memory reduction: `triton_int8_weight_only_conv1d_stdconv_effective`, but it is much slower.
- Batch 32 only fits under `<=48 GiB` with all wider convs quantized by Triton int8; the extra batch size does not recover throughput.
- A Triton batch-size metric bug was fixed by using int64 pointer offsets for long 131k activations.

## Main Run Roots

- Batch 20 linear/1x1 retest: `outputs/quant_ablation/20260701_115524_aggressive_all_1x1conv_batch20_test`
- Batch 20 stdconv materialization: `outputs/quant_ablation/20260701_121806_stdconv_effective_batch20_test`
- Batch 20 Triton full test: `outputs/quant_ablation/20260701_134238_triton_conv1d_batch20_fulltest`
- Batch 32 baseline/linear quant: `outputs/quant_ablation/20260701_112655_torch_batch32_1x1conv_fulltest`
- Batch 32 patched Triton full test: `outputs/quant_ablation/20260701_143654_triton_int64_offsets_batch32_fulltest`
