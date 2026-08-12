# Liu held-out chromosome RNA target rank

For each target matrix $Y\in\mathbb{R}^{G\times C}$, $G$ is genes on one chromosome and $C$ is modeled cell groups. Both axes are centered before singular-value decomposition. Rank ceilings describe target structure, not achievable sequence-model accuracy.

## Raw counts per million

| Chromosome | Evaluated / available genes | Groups | Effective rank | Effective genes | Top-gene variance | Rank for R=0.8 | Rank-2 ceiling | Rank-8 ceiling | Rank-16 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chr8 | 1,067 / 1,525 | 186 | 5.32 | 4.82 | 0.4160 | 2 | 0.8631 | 0.9602 | 0.9838 |
| chr9 | 972 / 1,340 | 186 | 12.10 | 10.78 | 0.2051 | 4 | 0.6722 | 0.9186 | 0.9661 |

## log1p counts per million

| Chromosome | Evaluated / available genes | Groups | Effective rank | Effective genes | Top-gene variance | Rank for R=0.8 | Rank-2 ceiling | Rank-8 ceiling | Rank-16 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chr8 | 1,067 / 1,525 | 186 | 30.92 | 231.41 | 0.0206 | 8 | 0.5701 | 0.8056 | 0.8708 |
| chr9 | 972 / 1,340 | 186 | 33.75 | 283.55 | 0.0146 | 9 | 0.5516 | 0.7892 | 0.8638 |
