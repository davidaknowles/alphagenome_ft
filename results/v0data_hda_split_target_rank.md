# Mannens HDA held-out chromosome RNA target rank

Support: `fixed_window_full_span`.

For each target matrix $Y\in\mathbb{R}^{G\times C}$, $G$ is genes on one chromosome and $C$ is modeled cell groups. Both axes are centered before singular-value decomposition. Rank ceilings describe target structure, not achievable sequence-model accuracy.

## Raw counts per million

| Chromosome | Evaluated / available genes | Groups | Effective rank | Effective genes | Top-gene variance | Rank for R=0.8 | Rank-2 ceiling | Rank-8 ceiling | Rank-16 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chr8 | 1,968 / 2,429 | 134 | 7.01 | 8.10 | 0.2220 | 3 | 0.7509 | 0.9656 | 0.9898 |
| chr9 | 1,880 / 2,272 | 134 | 6.47 | 9.26 | 0.2208 | 2 | 0.8245 | 0.9606 | 0.9847 |

## log1p counts per million

| Chromosome | Evaluated / available genes | Groups | Effective rank | Effective genes | Top-gene variance | Rank for R=0.8 | Rank-2 ceiling | Rank-8 ceiling | Rank-16 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chr8 | 1,968 / 2,429 | 134 | 23.05 | 211.78 | 0.0220 | 7 | 0.6564 | 0.8225 | 0.8825 |
| chr9 | 1,880 / 2,272 | 134 | 24.96 | 259.49 | 0.0174 | 7 | 0.6481 | 0.8136 | 0.8766 |
