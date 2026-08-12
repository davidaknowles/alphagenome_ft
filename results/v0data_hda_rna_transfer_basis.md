# HDA RNA training-basis transfer audit

The ceiling projects each held-out chromosome target onto cell-group factors learned only from training chromosomes. It measures cross-chromosome target structure and does not use sequence or model predictions.

Training genes: 45,251. Support: `fixed_window_full_span`.

## Raw counts per million

| Chromosome | Genes | Groups | Rank-1 | Rank-2 | Rank-4 | Rank-8 | Rank-16 | Rank-32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chr8 | 1,968 | 134 | 0.4406 | 0.4801 | 0.6128 | 0.7453 | 0.8778 | 0.9364 |
| chr9 | 1,880 | 134 | 0.5326 | 0.6339 | 0.7435 | 0.8256 | 0.9418 | 0.9755 |

## log1p counts per million

| Chromosome | Genes | Groups | Rank-1 | Rank-2 | Rank-4 | Rank-8 | Rank-16 | Rank-32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chr8 | 1,968 | 134 | 0.5312 | 0.6516 | 0.7453 | 0.8132 | 0.8704 | 0.9191 |
| chr9 | 1,880 | 134 | 0.4842 | 0.6450 | 0.7321 | 0.8040 | 0.8679 | 0.9176 |
