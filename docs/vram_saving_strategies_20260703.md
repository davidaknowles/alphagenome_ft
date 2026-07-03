# VRAM Saving Strategies for Torch AlphaGenome Inference

This note summarizes the successful inference-memory reductions tested on the
recreated merged Torch checkpoint:

`outputs/quant_ablation/recreated_e4v7bejy/merged_torch_default_lora_locon.safetensors`

All runs evaluate the human brain development ATAC head at resolution 128 with
full valid/test splits. The main memory column is peak device memory sampled via
`nvidia-smi`; PyTorch allocated/reserved peaks are included to distinguish true
live tensor memory from allocator reservation.

## Strategy Summary

| short name | exact strategy | description |
|---|---|---|
| default | `default` | Standard Torch inference path with no extra dtype or quantization policy. |
| bf16 params | `bf16_params` | Uses the aggressive bf16 Torch dtype policy for parameters and activations. |
| triton conv | `bf16_triton_conv_stdconv_effective` | bf16 policy plus precomputed standardized conv weights and Triton int8 weight-only Conv1d for eligible wide convs. |
| no intermediates | `bf16_triton_conv_no_intermediates_stdconv_effective` | Triton conv path plus a 128bp-only encoder path that avoids retaining U-Net skip tensors not needed by the requested output. |
| triton pool | `bf16_triton_conv_no_intermediates_tritonpool_stdconv_effective` | Adds a Triton max-pool kernel that does not materialize PyTorch max-pool indices. |
| fused embed | `bf16_triton_conv_no_intermediates_tritonpool_fusedembed_stdconv_effective` | Adds a fused Triton kernel for `encoder.dna_embedder.block` (`RMSBatchNorm -> GELU -> int8 Conv1d`). |
| fused down0 | `bf16_triton_conv_no_intermediates_tritonpool_fuseddown0_stdconv_effective` | Adds a fused wrapper for `encoder.down_blocks.0` using fused conv blocks and lean residual adds. |
| fused embed+down0 | `bf16_triton_conv_no_intermediates_tritonpool_fusedembed_fuseddown0_stdconv_effective` | Combines Triton conv, no-intermediates, Triton pool, fused embedder, and fused down block 0. |

## Results: 131 kb Windows

Batch size was 32. The full valid/test evaluation used 2162 examples over 68
batches.

| short name | examples/s | nvidia MiB | torch alloc MiB | torch reserved MiB | valid diff Pearson | test diff Pearson |
|---|---:|---:|---:|---:|---:|---:|
| default | 9.41 | 92631 | 63140 | 91928 | 0.79893 | 0.80891 |
| bf16 params | 12.02 | 66935 | 42415 | 66248 | 0.79839 | 0.80874 |
| triton conv | 8.49 | 43611 | 27879 | 42924 | 0.79842 | 0.80877 |
| no intermediates | 8.51 | 43611 | 25319 | 42924 | 0.79842 | 0.80877 |
| triton pool | 8.78 | 34907 | 25319 | 34220 | 0.79842 | 0.80877 |
| fused embed | 8.03 | 32347 | 21735 | 31660 | 0.79845 | 0.80872 |
| fused down0 | 8.11 | 31323 | 25319 | 30636 | 0.79829 | 0.80860 |
| fused embed+down0 | 7.45 | 25179 | 19174 | 24492 | 0.79834 | 0.80860 |

For 131 kb windows, the strongest memory-saving configuration is
`fused embed+down0`, reducing observed peak memory from 92.6 GiB to 25.2 GiB
relative to default, and from 66.9 GiB to 25.2 GiB relative to bf16 params. The
tradeoff is throughput: 7.45 examples/s versus 12.02 examples/s for bf16 params.
The best throughput under 48 GiB is `triton pool` at 8.78 examples/s and
34.9 GiB.

## Results: 1 Mb Windows

Batch size was 2. The full valid/test evaluation used 269 examples over 135
batches. The 1 Mb cache contains valid/test only:

`/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/alphagenome_target_cache/humanbraindev_atac_w1048576_s1048576_validtest_float16`

| short name | examples/s | nvidia MiB | torch alloc MiB | torch reserved MiB | valid diff Pearson | test diff Pearson |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.721 | 52595 | 38507 | 51908 | 0.81825 | 0.82707 |
| bf16 params | 0.937 | 47597 | 25445 | 46910 | 0.81522 | 0.82640 |
| triton conv | 0.736 | 33359 | 22508 | 32672 | 0.81558 | 0.82662 |
| no intermediates | 0.749 | 29263 | 15452 | 28576 | 0.81558 | 0.82662 |
| triton pool | 0.753 | 30799 | 15452 | 30112 | 0.81558 | 0.82662 |
| fused embed | 0.701 | 29519 | 15453 | 28832 | 0.81554 | 0.82658 |
| fused down0 | 0.697 | 27215 | 15452 | 26528 | 0.81541 | 0.82643 |
| fused embed+down0 | 0.650 | 24143 | 15452 | 23456 | 0.81534 | 0.82634 |

For 1 Mb windows, the default path exceeds 48 GiB. Plain bf16 params fits just
under 48 GiB and is the fastest measured setting. The Triton/no-intermediates
path cuts allocated memory further, from 25.4 GiB to 15.5 GiB, while staying well
below 48 GiB observed. Fused variants reduce allocator-reserved and observed
memory further, but they reduce throughput.

## Practical Recommendations

- Use `bf16_params` when the model fits and throughput is the priority.
- Use `bf16_triton_conv_no_intermediates_tritonpool_stdconv_effective` as the
  default memory-efficient option for 131 kb windows: it stays well below
  48 GiB and was faster than the more fused variants.
- For the lowest observed memory, use
  `bf16_triton_conv_no_intermediates_tritonpool_fusedembed_fuseddown0_stdconv_effective`.
  This reached 25.2 GiB at 131 kb and 24.1 GiB at 1 Mb, but with a throughput
  penalty.
- On 1 Mb windows, attention and tower activations are a larger fraction of
  runtime. The convolution fusions still reduce observed memory, but they no
  longer improve throughput.

## Run Roots

- 131 kb optimized strategies:
  `outputs/quant_ablation/20260702_recreated_encoder_vram_131kb_batch32_full`
- 131 kb default/bf16 baselines:
  `outputs/quant_ablation/20260702_recreated_baselines_131kb_batch32_full`
- 1 Mb optimized strategies:
  `outputs/quant_ablation/20260702_recreated_encoder_vram_1mb_batch2_full`
- 1 Mb default/bf16 baselines:
  `outputs/quant_ablation/20260702_recreated_baselines_1mb_batch2_full`

