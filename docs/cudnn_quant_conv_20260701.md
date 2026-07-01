# cuDNN Conv1d Quantization Smoke

Branch: `cudnn`

Goal: test whether cuDNN int8 convolution is worth keeping as an alternative to the custom Triton Conv1d kernels.

## Implementation

- Added opt-in Torch strategies:
  - `cudnn_int8_dynamic_effective_conv1d`
  - `cudnn_int8_dynamic_conv1d_stdconv_effective`
- Conv1d is represented as cuDNN Conv2d over `N,1,L,C` NHWC tensors.
- cuDNN accepts int8 activation and int8 weight convolution in NHWC form; NCHW int8 returned `CUDNN_STATUS_NOT_SUPPORTED`.
- The implemented path uses tensorwise dynamic activation scaling, per-output-channel weight scaling, cuDNN int32 accumulation, then dequantizes back to the model activation dtype.
- Torch C++ extension compilation requires `ninja`; the Slurm Torch ablation wrapper now tries to load `Ninja/1.12.1-GCCcore-14.2.0`.

## Batch 20 Smoke Results

All rows used `--max-batches 4` per split on `valid,test`, so these are integration/performance smokes, not replacement full-test results.

| strategy | converted | stdconv eff | examples/s | nvidia-smi MiB | torch alloc MiB | test diff Pearson | valid diff Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bf16_params_stdconv_effective` | -1 | 23 | 11.70 | 36507 | 31600 | 0.8064 | 0.7231 |
| `cudnn_int8_dynamic_effective_conv1d` | 4 | 0 | 10.18 | 36613 | 31608 | 0.7897 | 0.6764 |
| `cudnn_int8_dynamic_conv1d_stdconv_effective` | 27 | 23 | 4.61 | 48733 | 46961 | 0.7113 | 0.5349 |

## Conclusion

This cuDNN path is not worth promoting over the existing Triton kernels in its current form. It is slower, does not improve memory for the effective-only case, and the all-wider-conv path crosses the `<=48 GiB` target in the short smoke while substantially degrading the metric. The likely issue is the required activation quantization/dequantization and tensor layout traffic around cuDNN; tensorwise activation scaling is also too coarse for the full wider-conv replacement.
