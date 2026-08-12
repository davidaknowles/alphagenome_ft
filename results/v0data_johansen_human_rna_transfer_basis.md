# Johansen human RNA training-basis transfer audit

The ceiling projects each held-out chromosome target onto cell-group factors learned only from training chromosomes. It measures cross-chromosome target structure and does not use sequence or model predictions.

Training genes: 7,510. Support: `fixed_window_full_span`.

## Raw counts per million

| Chromosome | Genes | Groups | Rank-1 | Rank-2 | Rank-4 | Rank-8 | Rank-16 | Rank-32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chr8 | 263 | 47 | 0.3971 | 0.5616 | 0.6878 | 0.8405 | 0.9592 | 0.9891 |
| chr9 | 308 | 47 | 0.5360 | 0.6257 | 0.8162 | 0.8979 | 0.9798 | 0.9964 |

## log1p counts per million

| Chromosome | Genes | Groups | Rank-1 | Rank-2 | Rank-4 | Rank-8 | Rank-16 | Rank-32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chr8 | 263 | 47 | 0.5325 | 0.6459 | 0.7447 | 0.8375 | 0.9268 | 0.9850 |
| chr9 | 308 | 47 | 0.5806 | 0.6422 | 0.7585 | 0.8570 | 0.9372 | 0.9878 |
