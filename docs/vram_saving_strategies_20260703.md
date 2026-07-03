# VRAM Saving Strategies for Torch AlphaGenome Inference

This note summarizes successful inference-memory reductions tested on the
recreated merged Torch checkpoint:

`outputs/quant_ablation/recreated_e4v7bejy/merged_torch_default_lora_locon.safetensors`

All runs evaluate the human brain development ATAC head at resolution 128 on
the full valid/test splits. The peak-memory column is observed device memory
sampled with `nvidia-smi`, reported as GiB. Accuracy is reported as test
differential Pearson only.

## Implementation Details

The strategies are cumulative in the order below, except for rows that omit one
of the optional fused blocks.

**Bfloat16 policy.** The `bf16_params` path uses the Torch model's aggressive
bfloat16 dtype policy for inference. This stores/casts floating-point model
state and persistent activations in `bfloat16` where the Torch implementation
supports it, while keeping integer inputs and indexing tensors unchanged.

**Effective standardized convolutions.** AlphaGenome's `StandardizedConv1d`
standardizes weights at every forward pass:

`w_eff = (w - mean(w)) * rsqrt(max(var(w) * fan_in, 1e-4)) * learned_scale`

For inference, `_stdconv_effective` precomputes this effective weight once and
replaces the module with `EffectiveConv1d`, preserving AlphaGenome SAME padding
but avoiding runtime standardization and making the layer eligible for the
Triton Conv1d path.

**Triton int8 weight-only Conv1d.** Eligible wide Conv1d layers are replaced by
a custom Triton kernel. Weights are quantized per output channel using symmetric
int8 storage:

`scale_c = max(abs(W_c)) / 127`, `qW_c = round(W_c / scale_c)`

The kernel dequantizes inside the matmul, accumulates in fp32 via `tl.dot`, and
stores outputs in the input activation dtype. Heads, adapters, LoRA/LoCon, and
other sensitive paths are skipped.

**No-intermediates encoder path.** For `resolutions=(128,)`, the decoder does
not consume the encoder U-Net skip tensors. The patched encoder therefore avoids
retaining those intermediate tensors during evaluation.

**Triton no-indices max-pool.** The encoder pool is `kernel_size=2, stride=2`.
The custom Triton pool loads the two candidate positions and stores only their
maximum, matching the ceil-length SAME behavior used by the local pooling layer.
This avoids PyTorch's hidden max-pool index/workspace allocation.

**Fused DNA embedder block.** `encoder.dna_embedder.block` is fused into one
Triton kernel:

`RMSBatchNorm -> JAX GELU -> int8 weight-only Conv1d`

The RMSBatchNorm part uses the stored running variance and affine parameters:

`x_norm = x * weight * rsqrt(running_var + eps) + bias`

The activation is AlphaGenome/JAX's approximate GELU, not PyTorch's default
erf GELU:

`gelu_jax(x) = x * sigmoid(1.702 * x)`

The fused kernel applies this activation before the int8 weight-only Conv1d and
does not materialize the normalized or activated full-resolution tensor.

**Fused down block 0.** `encoder.down_blocks.0` is wrapped with two fused
ConvBlocks and lean residual accumulation. In inference, the first block output
already has the expanded channel count, so the residual input is added in-place
into the leading input-channel slice instead of creating an explicit padded
residual tensor. The second residual add is also in-place.

## Strategy Names

| short name | exact strategy |
|---|---|
| default | `default` |
| bf16 params | `bf16_params` |
| triton conv | `bf16_triton_conv_stdconv_effective` |
| no intermediates | `bf16_triton_conv_no_intermediates_stdconv_effective` |
| triton pool | `bf16_triton_conv_no_intermediates_tritonpool_stdconv_effective` |
| fused embed | `bf16_triton_conv_no_intermediates_tritonpool_fusedembed_stdconv_effective` |
| fused down0 | `bf16_triton_conv_no_intermediates_tritonpool_fuseddown0_stdconv_effective` |
| fused embed+down0 | `bf16_triton_conv_no_intermediates_tritonpool_fusedembed_fuseddown0_stdconv_effective` |

## 131 kb Windows

Batch size was 32. The full valid/test evaluation used 2162 examples over 68
batches.

| strategy | bf16 | effective conv | int8 conv | no intermediates | triton pool | fused embed | fused down0 | peak GiB | test r |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|
| default |  |  |  |  |  |  |  | 90.46 | 0.80891 |
| bf16 params | ✓ |  |  |  |  |  |  | 65.37 | 0.80874 |
| triton conv | ✓ | ✓ | ✓ |  |  |  |  | 42.59 | 0.80877 |
| no intermediates | ✓ | ✓ | ✓ | ✓ |  |  |  | 42.59 | 0.80877 |
| triton pool | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | 34.09 | 0.80877 |
| fused embed | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | 31.59 | 0.80872 |
| fused down0 | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | 30.59 | 0.80860 |
| fused embed+down0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 24.59 | 0.80860 |

At 131 kb, the lowest-memory successful configuration was
`fused embed+down0`, reducing observed peak memory from 90.46 GiB to 24.59 GiB
relative to default. The best non-fused memory-efficient option was
`triton pool`, at 34.09 GiB with no visible test metric change versus the
Triton Conv1d baseline.

## 1 Mb Windows

Batch size was 2. The full valid/test evaluation used 269 examples over 135
batches. The 1 Mb cache contains valid/test only:

`/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/alphagenome_target_cache/humanbraindev_atac_w1048576_s1048576_validtest_float16`

| strategy | bf16 | effective conv | int8 conv | no intermediates | triton pool | fused embed | fused down0 | peak GiB | test r |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|
| default |  |  |  |  |  |  |  | 51.36 | 0.82707 |
| bf16 params | ✓ |  |  |  |  |  |  | 46.48 | 0.82640 |
| triton conv | ✓ | ✓ | ✓ |  |  |  |  | 32.58 | 0.82662 |
| no intermediates | ✓ | ✓ | ✓ | ✓ |  |  |  | 28.58 | 0.82662 |
| triton pool | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | 30.08 | 0.82662 |
| fused embed | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | 28.83 | 0.82658 |
| fused down0 | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | 26.58 | 0.82643 |
| fused embed+down0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 23.58 | 0.82634 |

At 1 Mb, default inference exceeded 48 GiB. Plain bf16 params fit just under
48 GiB, while the Triton/no-intermediates family reduced memory substantially
further. The fused blocks primarily reduced observed/reserved memory at this
longer context length; they were useful for minimum VRAM but not for maximum
throughput.

## Run Roots

- 131 kb optimized strategies:
  `outputs/quant_ablation/20260702_recreated_encoder_vram_131kb_batch32_full`
- 131 kb default/bf16 baselines:
  `outputs/quant_ablation/20260702_recreated_baselines_131kb_batch32_full`
- 1 Mb optimized strategies:
  `outputs/quant_ablation/20260702_recreated_encoder_vram_1mb_batch2_full`
- 1 Mb default/bf16 baselines:
  `outputs/quant_ablation/20260702_recreated_baselines_1mb_batch2_full`

