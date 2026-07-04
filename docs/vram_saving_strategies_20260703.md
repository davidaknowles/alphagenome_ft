# VRAM Saving Strategies for Torch AlphaGenome Inference

This note summarizes successful inference-memory reductions tested on the
recreated merged Torch checkpoint:

`outputs/quant_ablation/recreated_e4v7bejy/merged_torch_default_lora_locon.safetensors`

All runs evaluate the human brain development ATAC head at resolution 128 on
the full valid/test splits. The peak-memory column is observed device memory
sampled with `nvidia-smi`, reported as GiB. Accuracy is reported as test
differential Pearson only.

The feature matrix columns are:

- **BF16**: aggressive bfloat16 Torch dtype policy.
- **Eff.**: precomputed effective standardized convolution weights.
- **Int8**: Triton int8 weight-only Conv1d.
- **NoInt**: 128bp-only encoder path that omits unused skip tensors.
- **Pool**: Triton max-pool without PyTorch's index/workspace allocation.
- **FEmb**: fused `encoder.dna_embedder.block`.
- **FD0**: fused `encoder.down_blocks.0`.

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

This saves VRAM because the unfused path creates separate full-resolution
temporary tensors for normalization, activation, and convolution input. At
131 kb and batch size 32, these temporaries are large enough to drive allocator
reservation and observed peak memory. The fused kernel streams through the
operation and only stores the final block output.

**Fused down block 0.** `encoder.down_blocks.0` is wrapped with two fused
ConvBlocks and lean residual accumulation. In inference, the first block output
already has the expanded channel count, so the residual input is added in-place
into the leading input-channel slice instead of creating an explicit padded
residual tensor. The second residual add is also in-place.

This saves VRAM by avoiding both the ConvBlock temporaries described above and
the explicit padded residual tensor for the first channel-expanding down block.
The benefit appears mostly as lower reserved/observed device memory because the
peak allocator state is sensitive to high-resolution transient tensors.

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

| strategy | BF16 | Eff. | Int8 | NoInt | Pool | FEmb | FD0 | peak GiB | examples/s | test r |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|
| default |  |  |  |  |  |  |  | 90.46 | 9.41 | 0.80891 |
| bf16 params | ✓ |  |  |  |  |  |  | 65.37 | 12.02 | 0.80874 |
| triton conv | ✓ | ✓ | ✓ |  |  |  |  | 42.59 | 8.49 | 0.80877 |
| no intermediates | ✓ | ✓ | ✓ | ✓ |  |  |  | 42.59 | 8.51 | 0.80877 |
| triton pool | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | 34.09 | 8.78 | 0.80877 |
| fused embed | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | 31.59 | 8.03 | 0.80872 |
| fused down0 | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | 30.59 | 8.11 | 0.80860 |
| fused embed+down0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 24.59 | 7.45 | 0.80860 |

At 131 kb, the lowest-memory successful configuration was
`fused embed+down0`, reducing observed peak memory from 90.46 GiB to 24.59 GiB
relative to default. The best non-fused memory-efficient option was
`triton pool`, at 34.09 GiB with no visible test metric change versus the
Triton Conv1d baseline.

## 1 Mb Windows

Batch size was 2. The full valid/test evaluation used 269 examples over 135
batches. The 1 Mb cache contains valid/test only:

`/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/alphagenome_target_cache/humanbraindev_atac_w1048576_s1048576_validtest_float16`

| strategy | BF16 | Eff. | Int8 | NoInt | Pool | FEmb | FD0 | peak GiB | examples/s | test r |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|
| default |  |  |  |  |  |  |  | 51.36 | 0.721 | 0.82707 |
| bf16 params | ✓ |  |  |  |  |  |  | 46.48 | 0.937 | 0.82640 |
| triton conv | ✓ | ✓ | ✓ |  |  |  |  | 32.58 | 0.736 | 0.82662 |
| no intermediates | ✓ | ✓ | ✓ | ✓ |  |  |  | 28.58 | 0.749 | 0.82662 |
| triton pool | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | 30.08 | 0.753 | 0.82662 |
| fused embed | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | 28.83 | 0.701 | 0.82658 |
| fused down0 | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | 26.58 | 0.697 | 0.82643 |
| fused embed+down0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 23.58 | 0.650 | 0.82634 |

At 1 Mb, default inference exceeded 48 GiB. Plain bf16 params fit just under
48 GiB, while the Triton/no-intermediates family reduced memory substantially
further. The fused blocks primarily reduced observed/reserved memory at this
longer context length; they were useful for minimum VRAM but not for maximum
throughput.

### 1 Mb FlexAttention Follow-Up

FlexAttention was re-tested at 1 Mb because attention and attention-bias tensors
become more important at longer context. These runs used the same batch size 2,
checkpoint, target cache, and `bf16_triton_conv_stdconv_effective` baseline.

| attention path | bias representation | peak GiB | torch alloc GiB | torch reserved GiB | examples/s | test r |
|---|---|---:|---:|---:|---:|---:|
| eager baseline | full repeated bias | 32.58 | 21.98 | 31.91 | 0.736 | 0.82662 |
| FlexAttention | full repeated bias | 28.58 | 18.01 | 27.91 | 0.746 | 0.82663 |
| FlexAttention | low-res bias | 20.58 | 13.91 | 19.91 | 0.827 | 0.82663 |

At 1 Mb, the low-res-bias FlexAttention path is useful: it avoids expanding the
attention bias to the full repeated tensor, reduces observed peak memory by
about 12 GiB relative to eager attention, and improves throughput by about 12%,
with unchanged test differential Pearson at displayed precision.

## Considered But Excluded

**Float8 and NF4 linear quantization.** `torchao_float8_linear` and
`bnb_nf4_weight_only_linear` are implemented in `scripts/run_quant_ablation.py`
and were benchmarked earlier, but they are not included in the recommended
stack. Linear-only float8 did not reduce actual device memory relative to
`bf16_params` in the physical-batch runs, and was slower at batch 20
(`11.50` examples/s vs `12.55` for `bf16_params`, both about `36.2` GiB).
NF4/NVFP4 linear and 1x1-conv quantization helped in small weight-focused
sweeps, but once the wide convolutions are handled by the Triton int8 Conv1d
path, the peak is dominated by activations, high-resolution convolution
temporaries, pooling workspace, and allocator reservation. In the matched
batch-32 combo run, adding NF4 linears/1x1 convs to the Triton-conv stack used
essentially the same observed memory (`43.62` GiB vs `43.61` GiB) with about a
`0.001` drop in test differential Pearson.

**FlexAttention at 131 kb.** A parity-preserving FlexAttention `MHABlock`
replacement was implemented as `flex_mha`, including the AlphaGenome score
transform: additive attention bias followed by tanh soft-cap before softmax.
Standalone parity was close, and full-model test differential Pearson was
unchanged at displayed precision. At 131 kb it did not reduce peak memory,
because the measured peak is dominated elsewhere after Triton convs are
enabled. At batch 32, plain FlexAttention was slower than eager attention in the
recorded full run (`8.44` vs `9.10` examples/s) with the same torch allocation.
This makes FlexAttention context-dependent rather than a default 131 kb
recommendation; the 1 Mb low-res-bias variant is the useful case.

## Run Roots

- 131 kb optimized strategies:
  `outputs/quant_ablation/20260702_recreated_encoder_vram_131kb_batch32_full`
- 131 kb default/bf16 baselines:
  `outputs/quant_ablation/20260702_recreated_baselines_131kb_batch32_full`
- 1 Mb optimized strategies:
  `outputs/quant_ablation/20260702_recreated_encoder_vram_1mb_batch2_full`
- 1 Mb default/bf16 baselines:
  `outputs/quant_ablation/20260702_recreated_baselines_1mb_batch2_full`
- 1 Mb FlexAttention follow-up:
  `outputs/quant_ablation/20260703_1mb_flexattn_batch2`
