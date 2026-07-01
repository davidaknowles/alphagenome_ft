# Encoder VRAM Reduction

Goal: reduce batch-32 128bp Torch inference VRAM without sacrificing numerical parity.

## Changes Tested

- `bf16_triton_conv_no_intermediates_stdconv_effective`: keeps the existing bf16 + Triton int8 weight-only Conv1d path, but patches the encoder to skip U-Net skip tensors when evaluating `resolutions=(128,)`.
- `bf16_triton_conv_no_intermediates_tritonpool_stdconv_effective`: also replaces the shared encoder `Pool1d(kernel_size=2, stride=2, method=max)` with a Triton max-pool kernel that does not materialize max-pool indices.

## Batch 32 Full Run

| strategy | examples/s | nvidia-smi MiB | torch alloc MiB | torch reserved MiB | test diff Pearson | valid diff Pearson |
|---|---:|---:|---:|---:|---:|---:|
| `bf16_triton_conv_stdconv_effective` | 8.47 | 43611 | 27879 | 42922 | 0.8101 | 0.7989 |
| `bf16_triton_conv_no_intermediates_tritonpool_stdconv_effective` | 8.46 | 34907 | 25319 | 34220 | 0.8101 | 0.7989 |

Run root for the new strategy:

`outputs/quant_ablation/20260701_no_intermediates_tritonpool_batch32_full`

## Attribution Summary

| strategy | exact forward peak alloc MiB | exact forward peak reserved MiB | main remaining peak |
|---|---:|---:|---|
| `bf16_triton_conv_stdconv_effective` | 27828 | 41814 | encoder skip tensors plus high-res conv/pool temporaries |
| `bf16_triton_conv_no_intermediates_stdconv_effective` | 25268 | 41814 | PyTorch max-pool transient/reserved memory |
| `bf16_triton_conv_no_intermediates_tritonpool_stdconv_effective` | 25268 | 33110 | `encoder.dna_embedder.block` |

Attribution roots:

- `outputs/quant_ablation/20260701_mem_attribution/bf16_triton_conv_batch32_valid1_eval128_encoder_staged`
- `outputs/quant_ablation/20260701_mem_attribution/bf16_triton_conv_no_intermediates_batch32_valid1_eval128_encoder_staged`
- `outputs/quant_ablation/20260701_mem_attribution/bf16_triton_conv_no_intermediates_tritonpool_batch32_valid1_eval128_encoder_staged`

## Takeaways

- Skipping encoder intermediates is numerically neutral and saves about `2.6 GiB` of allocated memory, but does not reduce observed memory by itself because PyTorch max-pool still drives allocator reservation.
- The custom Triton no-indices max-pool removes the large pooling transient. First-pool temporary memory falls from about `15.4 GiB` to the expected `3.1 GiB` output allocation.
- The combined strategy keeps throughput and metrics at parity while reducing observed batch-32 memory by about `8.7 GiB`.
