# Linear + Triton Conv Quantization Combo Smoke

Branch: `ablate-quant`

Goal: test whether combining true Linear/1x1 Conv quantization with `triton_int8_weight_only_conv1d_stdconv_effective` improves the throughput/memory tradeoff. These strategies explicitly cast unquantized parameters to bf16 before applying quantized wrappers.

## Strategies Added

- `bf16_triton_conv_stdconv_effective`
- `nvfp4_linear1x1_triton_conv_stdconv_effective`
- `nf4_linear1x1_triton_conv_stdconv_effective`

The `bf16_triton_conv` strategy is the explicit bf16-parameter + Triton wider-conv path. The Linear combo strategies materialize `StandardizedConv1d` first, convert eligible `Conv1d(kernel_size=1)` modules to Linear wrappers, quantize eligible Linear modules, then replace all eligible wider Conv1d modules with the Triton int8 weight-only Conv1d path.

## Debug Notes

- The initial comparison was invalid: it compared short combo smokes to full Triton-only runs.
- The table below uses matched `--max-batches 4` smoke settings for the rows shown.
- These short runs still include first-use overhead and are too noisy for final ranking; use full runs for deciding whether NF4 is actually faster.

## Full Results

Run root: `outputs/quant_ablation/20260701_172244_linear_triton_combo_full`

| batch | strategy | converted | examples/s | nvidia-smi MiB | torch alloc MiB | test diff Pearson | valid diff Pearson |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | `bf16_triton_conv_stdconv_effective` | 27 | 8.60 | 25705 | 17652 | 0.8101 | 0.7989 |
| 20 | `nf4_linear1x1_triton_conv_stdconv_effective` | 146 | 8.26 | 25701 | 17354 | 0.8091 | 0.7984 |
| 20 | `nvfp4_linear1x1_triton_conv_stdconv_effective` | 146 | 8.48 | 25777 | 17367 | 0.8093 | 0.7981 |
| 32 | `bf16_triton_conv_stdconv_effective` | 27 | 8.47 | 43611 | 27879 | 0.8101 | 0.7989 |
| 32 | `nf4_linear1x1_triton_conv_stdconv_effective` | 146 | 8.53 | 43623 | 27582 | 0.8091 | 0.7984 |
| 32 | `nvfp4_linear1x1_triton_conv_stdconv_effective` | 146 | 8.58 | 43699 | 27596 | 0.8093 | 0.7981 |

## Matched Smoke Results

| batch | strategy | converted | examples/s | nvidia-smi MiB | torch alloc MiB | test diff Pearson | valid diff Pearson |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | `bf16_triton_conv_stdconv_effective` | 27 | 7.06 | 25685 | 17653 | 0.8062 | 0.7233 |
| 20 | `nvfp4_linear1x1_triton_conv_stdconv_effective` | 146 | 6.64 | 25777 | 17369 | 0.8065 | 0.7237 |
| 20 | `nf4_linear1x1_triton_conv_stdconv_effective` | 146 | 8.54 | 25701 | 17355 | 0.8052 | 0.7208 |
| 32 | `bf16_triton_conv_stdconv_effective` | 27 | 8.67 | 43609 | 27880 | 0.7933 | 0.7482 |
| 32 | `nf4_linear1x1_triton_conv_stdconv_effective` | 146 | 8.13 | 43623 | 27583 | 0.7926 | 0.7460 |

## Conclusion

The memory story is consistent: Linear/1x1 quantization barely changes actual device memory once all wider convs are already quantized by Triton. Full runs show `bf16_triton_conv_stdconv_effective` is best at batch 20. At batch 32, NF4/NVFP4 Linear+1x1 combos are slightly faster than bf16+Triton, but memory is effectively unchanged and differential Pearson drops by about `0.0008-0.0010`.
