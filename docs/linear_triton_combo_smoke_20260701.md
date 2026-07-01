# Linear + Triton Conv Quantization Combo Smoke

Branch: `ablate-quant`

Goal: test whether combining true Linear/1x1 Conv quantization with `triton_int8_weight_only_conv1d_stdconv_effective` improves the throughput/memory tradeoff. These strategies still use the Torch bf16 dtype policy for unquantized parameters.

## Strategies Added

- `torchao_nvfp4_weight_only_all_linear_1x1conv_triton_int8_weight_only_conv1d_stdconv_effective`
- `bnb_nf4_weight_only_all_linear_1x1conv_triton_int8_weight_only_conv1d_stdconv_effective`

Both materialize `StandardizedConv1d` first, convert eligible `Conv1d(kernel_size=1)` modules to Linear wrappers, quantize eligible Linear modules, then replace all eligible wider Conv1d modules with the Triton int8 weight-only Conv1d path.

## Smoke Results

All rows used `--max-batches 4` per split on `valid,test`, so compare them as directional smokes rather than full-test replacements.

| batch | strategy | converted | examples/s | nvidia-smi MiB | torch alloc MiB | test diff Pearson | valid diff Pearson |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | `torchao_nvfp4_weight_only_all_linear_1x1conv_triton_int8_weight_only_conv1d_stdconv_effective` | 146 | 6.64 | 25777 | 17369 | 0.8065 | 0.7237 |
| 20 | `bnb_nf4_weight_only_all_linear_1x1conv_triton_int8_weight_only_conv1d_stdconv_effective` | 146 | 8.54 | 25701 | 17355 | 0.8052 | 0.7208 |
| 32 | `bnb_nf4_weight_only_all_linear_1x1conv_triton_int8_weight_only_conv1d_stdconv_effective` | 146 | 8.13 | 43623 | 27583 | 0.7926 | 0.7460 |

## Conclusion

The combo does not improve the current best tradeoff. The extra Linear/1x1 quantization barely changes actual device memory once all wider convs are already quantized by Triton, and it reduces throughput. NF4 is less slow than NVFP4 in this combo, but batch 32 NF4 still trails the existing `triton_int8_weight_only_conv1d_stdconv_effective` batch-32 full result (`8.93 examples/s`, `43609 MiB`).
