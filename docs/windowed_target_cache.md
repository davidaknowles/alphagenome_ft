# Windowed Target Cache

The windowed target cache is a dense binary replacement for repeated
`pyBigWig.values()` calls during fine-tuning. It is designed for fixed-width
training windows where every batch needs all target tracks over the same genomic
span.

## Layout

```
cache_dir/
  manifest.json
  <cached_split>/
    <head_id>.npy
```

Each `.npy` file is a NumPy-format memmap array:

```
shape = [num_windows, window_size, num_tracks]
dtype = float16 or float32
order = C
```

For the human brain development ATAC runs, `head_id` is
`humanbraindev_atac`, `window_size` is usually `131072`, and `num_tracks` is
the number of BigWig files discovered for the head.

Missing or NaN BigWig values are stored as `0.0`. The training loader casts
cached targets to `float32` before handing the batch to JAX, so `float16` cache
storage reduces disk I/O while preserving the existing model input dtype.

## Manifest

`manifest.json` records:

- cache format name and version
- dtype
- ordered intervals for each split
- ordered source BigWig paths, file sizes, and mtimes for each head

The loader validates the manifest against the requested cached splits, intervals,
and target tracks before training starts. A cache built for one split definition,
window size, target set, or BigWig version is intentionally rejected for another.

## Build And Use

Build a cache and then train from it:

```bash
python scripts/run_humanbraindev_finetune.py \
  --target-cache-dir /gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/alphagenome_target_cache/humanbraindev_atac_w131072_float16 \
  --build-target-cache \
  --target-cache-splits valid,test \
  --target-cache-dtype float16 \
  --target-cache-workers 8 \
  ...
```

Use an existing cache without rebuilding:

```bash
python scripts/run_humanbraindev_finetune.py \
  --target-cache-dir /gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/alphagenome_target_cache/humanbraindev_atac_w131072_float16 \
  ...
```

The cache is split/window-major so cached batches read complete target windows
from the memmap. `--target-cache-splits valid,test` is useful for large
multi-source training: it caches repeated evaluation targets while training
continues to stream the much larger training split from BigWigs.

Cache builds also write a `README.md` inside the cache directory with the
format summary, split counts, source BigWig provenance, and a link to the
repository branch that produced it.
