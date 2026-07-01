# Fast Attention Backend Ablation

Branch: `fast-attend`

Goal: benchmark parity-preserving MHA alternatives on top of `bf16_triton_conv_stdconv_effective`.

## Backend Status

- `flex_mha`: implemented with PyTorch FlexAttention and a `score_mod` matching the existing score path: additive attention bias, then tanh soft-cap, then softmax.
- `triton_mha`: implemented as a local custom Triton streaming-softmax MHA kernel with the same additive bias and tanh soft-cap.
- FlashAttention package: not installed in `~/venv/torchfix`.
- SageAttention package: not installed in `~/venv/torchfix`.
- Standard PyTorch SDPA exists, but it cannot directly express the required `tanh((qk/sqrt(d) + bias) / 5) * 5` score transform, so it is not a parity-preserving MHA replacement here.

Both implemented backends patch only `MHABlock`. `RowAttentionBlock` remains eager, and `AttentionBiasBlock` still materializes the full `B,8,S,S` bias tensor.

## Parity Smoke

On a standalone bf16 `MHABlock` with `S=32`, both FlexAttention and custom Triton matched the eager block closely:

| backend | max abs diff | mean abs diff |
|---|---:|---:|
| `flex_mha` | 0.0059 | 0.00051 |
| `triton_mha` | 0.0059 | 0.00051 |

The full-model differential Pearson metrics are unchanged at displayed precision in the full runs below.

## Full Results

Run root: `outputs/quant_ablation/20260701_181415_fast_attend_full`

| batch | strategy | attention backend | examples/s | nvidia-smi MiB | torch alloc MiB | test diff Pearson | valid diff Pearson |
|---:|---|---|---:|---:|---:|---:|---:|
| 20 | `bf16_triton_conv_stdconv_effective` | eager | 8.68 | 25689 | 17652 | 0.8101 | 0.7989 |
| 20 | `bf16_triton_conv_flexattn_stdconv_effective` | flex_mha | 8.71 | 25689 | 17652 | 0.8101 | 0.7989 |
| 20 | `bf16_triton_conv_tritonattn_stdconv_effective` | triton_mha | 8.73 | 25689 | 17652 | 0.8101 | 0.7989 |
| 32 | `bf16_triton_conv_stdconv_effective` | eager | 9.10 | 43627 | 27879 | 0.8101 | 0.7989 |
| 32 | `bf16_triton_conv_flexattn_stdconv_effective` | flex_mha | 8.44 | 43611 | 27879 | 0.8101 | 0.7989 |
| 32 | `bf16_triton_conv_tritonattn_stdconv_effective` | triton_mha | 8.89 | 43611 | 27879 | 0.8101 | 0.7989 |

## Conclusion

At the current `131072` bp context, neither fast-attention backend materially changes peak VRAM. This is expected because the MHA bias tensor is still materialized, and the overall peak is still dominated by other activations/reservations once wider convs are quantized.

Throughput is roughly tied at batch 20. At batch 32, eager remains fastest in this run; custom Triton is closer than FlexAttention. These kernels may become more relevant at larger context lengths, but the first larger-context memory target should be avoiding or recomputing the full attention-bias tensor, not only replacing the softmax/value attention kernel.
