# Human Brain Development Finetuning Results

Collated on 2026-06-26 from `checkpoints/humanbraindev_precision_compare`, Slurm accounting, and local profiling/smoke checks.

## Summary

- Best completed validation/test accuracy in the current comparable runs is `lora_default_npbigwig32`: validation `r2_global=0.8396`, test `r2_global=0.8470`.
- `lora_lora32_base16_npbigwig32` is the next strongest completed LoRA run: validation `r2_global=0.8306`, test `r2_global=0.8315`.
- `heads_default_npbigwig32` and `heads_default_targetcache` are numerically matched, which validates that the precomputed target cache preserves training behavior for the completed heads-only task.
- `r2_over_cell_types` is currently extremely negative for many runs. Treat it as a diagnostic rather than a ranking metric until the aggregation is audited for low-variance tracks/cell-type slices.
- The target cache improved the data path substantially, but the first full cache-backed Slurm array used node-local `/scratch`; only task 0 completed, while tasks 1-5 failed because the cache manifest was not visible on their nodes.

## Accuracy

Metrics are from `last/metrics.json`. R2 columns are for `humanbraindev_atac`.

| Run | Epoch | Step | Train loss | Train R2 global | Train R2 loci | Train R2 cell types | Valid loss | Valid R2 global | Valid R2 loci | Valid R2 cell types | Test loss | Test R2 global | Test R2 loci | Test R2 cell types |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora_default_npbigwig32` | 5 | 13095 | 34.5915 | 0.8588 | 0.8489 | -2444.4 | 29.3973 | 0.8396 | 0.8339 | -7307.2 | 33.0940 | 0.8470 | 0.8428 | -452344.7 |
| `lora_lora32_base16_npbigwig32` | 5 | 13095 | 34.6256 | 0.8461 | 0.8362 | -2932.4 | 29.4236 | 0.8306 | 0.8261 | -3492.2 | 33.1298 | 0.8315 | 0.8284 | -314094.8 |
| `lora_default_r2` | 2 | 41904 | 34.7515 | 0.8208 | 0.8086 | -2961.9 | 29.2191 | 0.8111 | 0.8024 | -11047.1 | 33.1883 | 0.8029 | 0.7942 | -353181.0 |
| `heads_default_r2` | 2 | 41904 | 34.8879 | 0.8031 | 0.7948 | -67288.3 | 29.3333 | 0.7916 | 0.7864 | -32691.2 | 33.2714 | 0.7930 | 0.7879 | -1348570.6 |
| `heads_default_npbigwig32` | 5 | 6550 | 34.8619 | 0.7654 | 0.7644 | -70678.0 | 30.0479 | 0.7688 | 0.7693 | -50071.5 | 33.3616 | 0.7702 | 0.7714 | -1332671.8 |
| `heads_default_targetcache` | 5 | 6550 | 34.8619 | 0.7654 | 0.7644 | -70889.1 | 30.0479 | 0.7688 | 0.7693 | -49820.4 | 33.3616 | 0.7702 | 0.7714 | -1333030.5 |
| `lora_lora32_base16_r2` | 2 | 41904 | 36.2142 | 0.1489 | 0.1654 | -45463.6 | 30.3954 | 0.1545 | 0.1727 | -22754.9 | 34.4580 | 0.1525 | 0.1703 | -68572.8 |
| `heads_lora32_base16_npbigwig32` | 5 | 6550 | 35.5250 | 0.1448 | 0.1753 | -5353.0 | 30.6291 | 0.1489 | 0.1811 | -4910.3 | 33.9786 | 0.1464 | 0.1784 | -53148.7 |
| `heads_lora32_base16_r2` | 2 | 41904 | 35.6860 | 0.1345 | 0.1671 | -4867.8 | 30.0224 | 0.1368 | 0.1708 | -4431.3 | 34.0075 | 0.1345 | 0.1682 | -46169.0 |

## Slurm Status

| Job | Purpose | Status | Notes |
|---|---|---|---|
| `17829944_[0-5]` | `npbigwig32` comparison array | tasks 0-3 completed; tasks 4-5 failed | Completed tasks ran in `01:38:25` to `01:58:57`. Failed tasks stopped with non-finite training loss at epoch 1, step 50. |
| `17840007` | Full target-cache build | completed in `00:16:46` | Built `/scratch/daknowles/alphagenome_fp4/humanbraindev_atac_w131072_float16`: train `(20952, 131072, 134)`, valid `(1107, 131072, 134)`, test `(1055, 131072, 134)`, dtype `float16`. |
| `17840008_[0-5]` | target-cache comparison array | task 0 completed; tasks 1-5 failed | Task 0 completed in `01:28:22`. Tasks 1-5 failed with missing `manifest.json` because `/scratch` is node-local. |

## Performance

| Measurement | Result | Interpretation |
|---|---:|---|
| Original host-side profile, 3 batches | `23.710s` | `pyBigWig.values`, `np.asarray`, and NaN cleanup dominated host time. |
| After `pyBigWig.values(..., numpy=True)`, 3 batches | `2.810s` | BigWig path improved by about `8.4x`, but still host-bound. |
| Target-cache iterator, 8 batches | `0.345s` total, `0.0431s/batch` | Cached targets remove the BigWig read/decode bottleneck from the batch path. |
| Cached GPU smoke, 32 async steps | avg SM `69.8%`, median SM `92.5%`, max SM `100%`, active avg SM `89.7%` | Post-compile training can keep the GPU busy when the target cache is local and no per-step synchronization is forced. |
| Cached GPU smoke with per-step profiling/sync | avg SM `6.7%` | The low number was self-inflicted by forced synchronization from `--progress-interval 1 --profile-host-timing`; do not use that mode to estimate sustained utilization. |

## Failure Notes

- `17829944_4` and `17829944_5` failed with `FloatingPointError: Non-finite training loss encountered at epoch=1, epoch_step=50, global_step=50: loss=nan`. These correspond to the lower-precision/base8-style configurations that did not produce metrics JSONs.
- `17840008_1` through `17840008_5` failed with `FileNotFoundError: Target cache manifest not found: /scratch/daknowles/alphagenome_fp4/humanbraindev_atac_w131072_float16/manifest.json`.
- The target-cache failure is operational, not a cache-format failure: `/scratch` is local to the node that built the cache. The Slurm scripts now default to a shared cache next to the source BigWig data under `.../humanbraindev/alphagenome_target_cache/`.

## Artifacts

- Target cache format documentation: `docs/windowed_target_cache.md`
- Cache builder Slurm script: `scripts/slurm_build_humanbraindev_target_cache.sbatch`
- Finetuning Slurm script with cache flags: `scripts/slurm_humanbraindev_precision_compare.sbatch`
- Completed metrics: `checkpoints/humanbraindev_precision_compare/*/last/metrics.json`
- LoRA/checkpoint outputs are under each run's `best/` and `last/` checkpoint directories when the run reached checkpoint save.
