# Gene-level RNA target-rank audit

For a double-centered gene-by-cell-group target matrix, the rank-k ceiling is the correlation with its optimal rank-k singular-value approximation. It measures target representability by a shared cell-group basis, not achievable sequence-model accuracy.

## Raw counts per million

| Dataset | Genes | Groups | Effective rank | Rank for R=0.8 | Rank-1 ceiling | Rank-4 ceiling | Rank-8 ceiling | Rank-16 ceiling | Rank-32 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HDA | 59,310 | 134 | 17.99 | 6 | 0.5272 | 0.7258 | 0.8515 | 0.9336 | 0.9756 |
| Liu | 37,400 | 186 | 7.90 | 2 | 0.5933 | 0.8991 | 0.9432 | 0.9701 | 0.9852 |
| Johansen-human | 13,509 | 47 | 19.77 | 8 | 0.4701 | 0.6957 | 0.8024 | 0.9141 | 0.9885 |
| Johansen-macaque | 13,495 | 47 | 19.76 | 8 | 0.4699 | 0.6999 | 0.8081 | 0.9131 | 0.9867 |
| Johansen-marmoset | 13,506 | 47 | 21.60 | 9 | 0.4519 | 0.6731 | 0.7868 | 0.9006 | 0.9866 |

## log1p counts per million

| Dataset | Genes | Groups | Effective rank | Rank for R=0.8 | Rank-1 ceiling | Rank-4 ceiling | Rank-8 ceiling | Rank-16 ceiling | Rank-32 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HDA | 59,310 | 134 | 22.77 | 7 | 0.5420 | 0.7546 | 0.8271 | 0.8852 | 0.9314 |
| Liu | 37,400 | 186 | 29.42 | 8 | 0.4427 | 0.7063 | 0.8169 | 0.8833 | 0.9219 |
| Johansen-human | 13,509 | 47 | 15.34 | 6 | 0.5740 | 0.7583 | 0.8480 | 0.9304 | 0.9857 |
| Johansen-macaque | 13,495 | 47 | 15.32 | 6 | 0.5766 | 0.7616 | 0.8489 | 0.9278 | 0.9851 |
| Johansen-marmoset | 13,506 | 47 | 13.76 | 5 | 0.6085 | 0.7801 | 0.8585 | 0.9352 | 0.9875 |

A low rank requirement would support a factorized cell-group output head. A high requirement would instead favor full channel-specific heads and objective or backbone changes.
