# Physical-Batch True Quant + Pointwise Conv Ablation

## Motivation

The earlier batch-size sweep showed that weight-only quantization at batch 12 only modestly reduced memory (`~24 GiB` default to `~19 GiB` quantized by torch allocation). The goal here was higher-throughput inference under a practical `<=48 GiB` VRAM target, without forward microbatching.

## Implementation

All low-precision conversion helpers used by `scripts/run_quant_ablation.py` now live in this repo:

- `alphagenome_ft/torch_low_precision.py`

The Torch-only evaluator uses the cached target manifest and `pyfaidx` directly, so it does not import the JAX/Orbax/TensorFlow path at startup.

The Torch cached-target path now vectorizes DNA one-hot encoding with a NumPy lookup table instead of looping over bases in Python. The earlier benchmark was partly CPU/input-bound, which muted the measured throughput difference between `bf16_params` and true quantized modules.

New true-quant strategies covering pointwise convs:

- `torchao_float8_linear_1x1conv`
- `torchao_float8_all_linear_1x1conv`
- `torchao_nvfp4_weight_only_linear_1x1conv`
- `torchao_nvfp4_weight_only_all_linear_1x1conv`
- `bnb_nf4_weight_only_linear_1x1conv`
- `bnb_nf4_weight_only_all_linear_1x1conv`

These replace eligible `nn.Conv1d(kernel_size=1)` modules with a position-wise `nn.Linear` wrapper, then apply the existing true Linear quantization backend. This covered 10 additional pointwise conv modules in the tested checkpoint.

Wider Conv1d kernels are intentionally not quantized here: the available true CUDA quantization paths in this environment target Linear modules, not arbitrary Conv1d kernels. Quantizing those would require a real CUDA quantized Conv1d implementation or a model rewrite.

## Physical Batch 32

Run root: `outputs/quant_ablation/20260701_112655_torch_batch32_1x1conv_fulltest`

| strategy | converted | pointwise convs | examples/s | torch alloc MiB | nvidia-smi max MiB | diff Pearson |
|---|---:|---:|---:|---:|---:|---:|
| default | 0 | 0 | 9.18 | 63140 | 92631 | 0.8103 |
| bf16_params | -1 | 0 | 11.33 | 42415 | 66951 | 0.8102 |
| torchao_float8_linear | 109 | 0 | 10.06 | 42417 | 66935 | 0.8101 |
| torchao_float8_linear_1x1conv | 119 | 10 | 10.05 | 42416 | 66957 | 0.8097 |
| torchao_nvfp4_weight_only_linear | 109 | 0 | 10.38 | 42162 | 67105 | 0.8102 |
| torchao_nvfp4_weight_only_linear_1x1conv | 119 | 10 | 10.38 | 42134 | 67143 | 0.8093 |
| bnb_nf4_weight_only_linear | 109 | 0 | 7.92 | 42147 | 67043 | 0.8093 |
| bnb_nf4_weight_only_linear_1x1conv | 119 | 10 | 7.92 | 42120 | 67043 | 0.8092 |

Physical batch 32 keeps quantized torch allocation around `42 GiB`, but CUDA reserved/device memory is around `67 GiB`, so it is too large if the budget is actual `nvidia-smi` memory on a 48 GiB GPU.

## Physical Batch 20

Run root: `outputs/quant_ablation/20260701_113028_torch_batch20_1x1conv_fulltest`

| strategy | converted | pointwise convs | examples/s | torch alloc MiB | nvidia-smi max MiB | diff Pearson |
|---|---:|---:|---:|---:|---:|---:|
| default | 0 | 0 | 9.47 | 40063 | 57511 | 0.8103 |
| bf16_params | -1 | 0 | 11.23 | 31607 | 36229 | 0.8102 |
| torchao_float8_linear | 109 | 0 | 11.50 | 31608 | 36213 | 0.8101 |
| torchao_float8_linear_1x1conv | 119 | 10 | 10.30 | 31609 | 36241 | 0.8098 |
| torchao_nvfp4_weight_only_linear | 109 | 0 | 11.09 | 31354 | 36349 | 0.8102 |
| torchao_nvfp4_weight_only_linear_1x1conv | 119 | 10 | 11.88 | 31326 | 36385 | 0.8093 |
| bnb_nf4_weight_only_linear | 109 | 0 | 9.08 | 31339 | 36287 | 0.8094 |
| bnb_nf4_weight_only_linear_1x1conv | 119 | 10 | 9.08 | 31310 | 36287 | 0.8092 |

Physical batch 20 is the better `<=48 GiB` point by actual device memory. Best throughput was `torchao_nvfp4_weight_only_linear_1x1conv` at `11.88 examples/s`, `31.3 GiB` torch allocation, and `36.4 GiB` by `nvidia-smi`.

## Batch 20 Retest After Input-Path Fix

Run root: `outputs/quant_ablation/20260701_115524_aggressive_all_1x1conv_batch20_test`

| strategy | converted | pointwise convs | examples/s | torch alloc MiB | nvidia-smi max MiB | diff Pearson |
|---|---:|---:|---:|---:|---:|---:|
| bf16_params | -1 | 0 | 12.55 | 31607 | 36213 | 0.8102 |
| torchao_nvfp4_weight_only_all_linear_1x1conv | 119 | 10 | 13.28 | 31326 | 36385 | 0.8093 |
| bnb_nf4_weight_only_all_linear_1x1conv | 119 | 10 | 13.38 | 31310 | 36287 | 0.8092 |

With the faster input path, the aggressive true-quant options are now measurably faster than `bf16_params` at the same physical batch size while staying under the `<=48 GiB` actual-device-memory target. In this checkpoint, the `all_linear` skip policy did not increase the converted module count beyond the existing linears plus eligible pointwise convs; the remaining skipped linears fail shape constraints, and the remaining skipped convs are not plain eligible `Conv1d(kernel_size=1)` modules.

## Batch 20 Retest With Standardized Conv Materialization

Run root: `outputs/quant_ablation/20260701_121806_stdconv_effective_batch20_test`

The `_stdconv_effective` variants precompute the runtime-standardized weights for the 23 remaining `StandardizedConv1d` modules, then replace those modules with direct `EffectiveConv1d` modules for inference. This removes per-forward mean/variance/scale work without changing the conv math.

| strategy | materialized std convs | converted linears/convs | examples/s | torch alloc MiB | nvidia-smi max MiB | diff Pearson |
|---|---:|---:|---:|---:|---:|---:|
| bf16_params_stdconv_effective | 23 | -1 | 13.47 | 31600 | 36507 | 0.8102 |
| torchao_nvfp4_weight_only_all_linear_1x1conv_stdconv_effective | 23 | 119 | 13.29 | 31318 | 36597 | 0.8093 |
| bnb_nf4_weight_only_all_linear_1x1conv_stdconv_effective | 23 | 119 | 13.22 | 31304 | 36521 | 0.8092 |

Materializing standardized convs helped the bf16 baseline enough that it is now the fastest tested batch-20 path. The quantized variants still save a few hundred MiB of torch allocation, but they do not beat `bf16_params_stdconv_effective` on throughput in this run.

## Conv Quantization Follow-Up

Two wider-conv quantization paths were added after checking available package support:

- TorchAO intx Conv2d weight-only, by wrapping eligible Conv1d modules as Conv2d over a singleton height dimension.
- Custom Triton Conv1d kernels for int8 weight-only and dynamic-int8 activation plus int8 weight.

Short batch-20 smoke runs showed:

| strategy | convs converted | examples/s | nvidia-smi max MiB | diff Pearson | note |
|---|---:|---:|---:|---:|---|
| bf16_params_stdconv_effective | 0 | 13.33 | 36507 | 0.7918 | 10-batch smoke baseline |
| torchao_int8_weight_only_effective_conv2d | 4 | 13.31 | 36267 | 0.7656 | package path, metric drop |
| torchao_int4_weight_only_effective_conv2d | 4 | 13.26 | 36267 | 0.7575 | package path, larger metric drop |
| torchao_int8_weight_only_conv2d_stdconv_effective | 27 | 13.38 | 36579 | 0.7668 | package path, metric drop |
| torchao_int4_weight_only_conv2d_stdconv_effective | 27 | 13.38 | 36579 | 0.7435 | package path, larger metric drop |
| triton_int8_weight_only_effective_conv1d | 4 | 12.37 | 36605 | 0.7914 | custom kernel, metric preserved, slower |
| triton_int8_weight_only_conv1d_stdconv_effective | 27 | 9.00 | 25685 | 0.7913 | custom kernel, much lower memory, slower |
| triton_int8_dynamic_effective_conv1d | 4 | 12.15 | 36605 | 0.7914 | custom dynamic activation, slower |
| triton_int8_dynamic_conv1d_stdconv_effective | 27 | 8.07 | 25685 | 0.7913 | custom dynamic activation, much lower memory, slower |

The package-backed TorchAO Conv2d intx path is fast but not numerically acceptable for these convs. The custom Triton kernels preserve the smoke metric and substantially reduce memory when applied to all wider convs, but they do not meet the throughput goal.

## Triton Full Test Runs

Batch 20 full test root: `outputs/quant_ablation/20260701_134238_triton_conv1d_batch20_fulltest`

| strategy | convs converted | stdconv eff | examples/s | nvidia-smi max MiB | torch alloc MiB | diff Pearson |
|---|---:|---:|---:|---:|---:|---:|
| bf16_params_stdconv_effective | -1 | 23 | 13.29 | 36507 | 31600 | 0.8102 |
| triton_int8_weight_only_effective_conv1d | 4 | 0 | 12.78 | 36605 | 31568 | 0.8100 |
| triton_int8_dynamic_effective_conv1d | 4 | 0 | 12.11 | 36605 | 31568 | 0.8100 |
| triton_int8_weight_only_conv1d_stdconv_effective | 27 | 23 | 9.35 | 25685 | 17653 | 0.8101 |
| triton_int8_dynamic_conv1d_stdconv_effective | 27 | 23 | 8.52 | 25685 | 17653 | 0.8101 |

Batch 20 confirms the custom Triton kernels preserve the full-test metric. Quantizing all wider convs cuts actual device memory from `36.5 GiB` to `25.7 GiB`, but throughput falls from `13.29` to `9.35 examples/s` for the best Triton all-conv variant.

Batch 32 full test root: `outputs/quant_ablation/20260701_135127_triton_conv1d_batch32_fulltest`

| strategy | examples/s | nvidia-smi max MiB | torch alloc MiB | diff Pearson | status |
|---|---:|---:|---:|---:|---|
| triton_int8_weight_only_conv1d_stdconv_effective | 9.24 | 43609 | 27880 | 0.4348 | invalid metric |
| triton_int8_dynamic_conv1d_stdconv_effective | 8.59 | 43609 | 27880 | 0.4348 | invalid metric |

Batch 32 stays within the `<=48 GiB` actual-memory target for the all-conv Triton kernels, but the metric collapses. A threshold smoke (`outputs/quant_ablation/20260701_135617_triton_batch_threshold_smoke`) showed the all-conv Triton path is metric-stable for batch sizes 16 and 20, then degrades at batch 24 and above. Standalone layer checks at batch 32 have low relative error, so this appears to be a model-level interaction that still needs debugging before batch 32 can be trusted.

## Takeaways

- Do not use forward microbatching for this goal; physical batch 20 gives strong throughput and stays below 48 GiB actual device memory for quantized/bf16 runs.
- Pointwise Conv1d(k=1) quantization converts 10 additional modules. Wider Conv1d kernels still need a real quantized Conv1d backend or model rewrite; unfolding them into Linear would likely trade weight memory for much larger activations.
- For the best measured throughput under the `<=48 GiB` target, use `bf16_params_stdconv_effective` at physical batch 20.
- Use the NVFP4/NF4 `_stdconv_effective` variants only if the small torch-allocation reduction is worth the lower throughput and slightly lower differential Pearson.
- Use `triton_int8_weight_only_conv1d_stdconv_effective` at batch 20 only when actual device memory is more important than throughput; it is the best wider-conv memory reducer tested here, not the fastest path.
- Do not use the all-conv Triton variants at batch 24 or higher until the batch-size-dependent metric collapse is fixed.
