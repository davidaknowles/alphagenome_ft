# Gene-level RNA target-rank audit

For a double-centered gene-by-cell-group target matrix, the rank-k ceiling is the correlation with its optimal rank-k singular-value approximation. It measures target representability by a shared cell-group basis, not achievable sequence-model accuracy.

## Raw counts per million

| Dataset | Genes | Groups | Effective rank | Rank for R=0.8 | Rank-1 ceiling | Rank-4 ceiling | Rank-8 ceiling | Rank-16 ceiling | Rank-32 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zemke2023-human | 23,264 | 20 | 10.78 | 5 | 0.4902 | 0.7820 | 0.9257 | 0.9938 | - |
| zemke2023-macaque | 15,841 | 20 | 8.49 | 4 | 0.6134 | 0.8394 | 0.9360 | 0.9955 | - |
| zemke2023-marmoset | 22,046 | 20 | 8.47 | 4 | 0.6269 | 0.8396 | 0.9273 | 0.9943 | - |
| zemke2023-mouse | 26,291 | 20 | 7.37 | 3 | 0.6074 | 0.8895 | 0.9551 | 0.9955 | - |
| zemke2024 | 36,474 | 22 | 8.02 | 3 | 0.5910 | 0.8604 | 0.9454 | 0.9959 | - |

## log1p counts per million

| Dataset | Genes | Groups | Effective rank | Rank for R=0.8 | Rank-1 ceiling | Rank-4 ceiling | Rank-8 ceiling | Rank-16 ceiling | Rank-32 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zemke2023-human | 23,264 | 20 | 8.15 | 3 | 0.5779 | 0.8629 | 0.9442 | 0.9937 | - |
| zemke2023-macaque | 15,841 | 20 | 7.47 | 3 | 0.5993 | 0.8654 | 0.9501 | 0.9950 | - |
| zemke2023-marmoset | 22,046 | 20 | 6.65 | 2 | 0.6531 | 0.8863 | 0.9527 | 0.9948 | - |
| zemke2023-mouse | 26,291 | 20 | 8.20 | 3 | 0.5939 | 0.8502 | 0.9457 | 0.9939 | - |
| zemke2024 | 36,474 | 22 | 4.61 | 2 | 0.7744 | 0.9190 | 0.9679 | 0.9977 | - |

A low rank requirement would support a factorized cell-group output head. A high requirement would instead favor full channel-specific heads and objective or backbone changes.
