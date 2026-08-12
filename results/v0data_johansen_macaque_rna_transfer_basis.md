# Johansen macaque RNA training-basis transfer audit

The ceiling projects each held-out chromosome target onto cell-group factors learned only from training chromosomes. It measures cross-chromosome target structure and does not use sequence or model predictions.

Training genes: 7,640. Support: `fixed_window_full_span`.

## Raw counts per million

| Chromosome | Genes | Groups | Rank-1 | Rank-2 | Rank-4 | Rank-8 | Rank-16 | Rank-32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NC_041761.1 | 266 | 47 | 0.2502 | 0.3720 | 0.5540 | 0.7997 | 0.9470 | 0.9896 |
| NC_041762.1 | 269 | 47 | 0.3668 | 0.5557 | 0.6911 | 0.8194 | 0.9323 | 0.9943 |

## log1p counts per million

| Chromosome | Genes | Groups | Rank-1 | Rank-2 | Rank-4 | Rank-8 | Rank-16 | Rank-32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NC_041761.1 | 266 | 47 | 0.5251 | 0.6378 | 0.7339 | 0.8282 | 0.9205 | 0.9858 |
| NC_041762.1 | 269 | 47 | 0.5908 | 0.6752 | 0.7680 | 0.8543 | 0.9337 | 0.9850 |
