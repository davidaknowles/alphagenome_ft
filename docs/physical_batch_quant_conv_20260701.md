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

## Takeaways

- Do not use forward microbatching for this goal; physical batch 20 gives strong throughput and stays below 48 GiB actual device memory for quantized/bf16 runs.
- Pointwise Conv1d(k=1) quantization converts 10 additional modules. Wider Conv1d kernels still need a real quantized Conv1d backend or model rewrite; unfolding them into Linear would likely trade weight memory for much larger activations.
- For the best speed/memory tradeoff from the retest, use `bnb_nf4_weight_only_all_linear_1x1conv` or `torchao_nvfp4_weight_only_all_linear_1x1conv` at physical batch 20.
- If metric fidelity is prioritized over the small throughput edge, `torchao_float8_linear` at batch 20 is close in speed and has slightly better differential Pearson.
