# Johansen marmoset RNA training-basis transfer audit

The ceiling projects each held-out chromosome target onto cell-group factors learned only from training chromosomes. It measures cross-chromosome target structure and does not use sequence or model predictions.

Training genes: 7,572. Support: `fixed_window_full_span`.

## Raw counts per million

| Chromosome | Genes | Groups | Rank-1 | Rank-2 | Rank-4 | Rank-8 | Rank-16 | Rank-32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chr8 | 248 | 47 | 0.1698 | 0.2163 | 0.2777 | 0.4132 | 0.7631 | 0.9658 |
| chr9 | 431 | 47 | 0.5462 | 0.6375 | 0.6934 | 0.8277 | 0.9714 | 0.9925 |

## log1p counts per million

| Chromosome | Genes | Groups | Rank-1 | Rank-2 | Rank-4 | Rank-8 | Rank-16 | Rank-32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chr8 | 248 | 47 | 0.5668 | 0.6380 | 0.7436 | 0.8234 | 0.9139 | 0.9818 |
| chr9 | 431 | 47 | 0.6068 | 0.6879 | 0.7912 | 0.8681 | 0.9424 | 0.9899 |
