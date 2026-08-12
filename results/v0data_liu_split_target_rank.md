# Liu held-out chromosome RNA target rank

For each target matrix $Y\in\mathbb{R}^{G\times C}$, $G$ is genes on one chromosome and $C$ is modeled cell groups. Both axes are centered before singular-value decomposition. Rank ceilings describe target structure, not achievable sequence-model accuracy.

## Raw counts per million

| Chromosome | Genes | Groups | Effective rank | Rank for R=0.8 | Rank-2 ceiling | Rank-8 ceiling | Rank-16 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| chr8 | 1,525 | 186 | 9.22 | 3 | 0.7736 | 0.9280 | 0.9692 |
| chr9 | 1,340 | 186 | 4.99 | 1 | 0.8568 | 0.9446 | 0.9777 |

## log1p counts per million

| Chromosome | Genes | Groups | Effective rank | Rank for R=0.8 | Rank-2 ceiling | Rank-8 ceiling | Rank-16 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| chr8 | 1,525 | 186 | 27.03 | 7 | 0.5950 | 0.8265 | 0.8889 |
| chr9 | 1,340 | 186 | 29.16 | 8 | 0.5776 | 0.8138 | 0.8838 |
