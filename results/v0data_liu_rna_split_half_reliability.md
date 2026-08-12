# Liu RNA split-half reliability

Biological samples were assigned independently for each cell group to two greedily library-depth-balanced halves. Counts were summed and normalized to counts per million, CPM, within each half. The Spearman-Brown correction estimates reliability of the complete pseudobulk from equal-half correlation.

| Quantity | Value |
|---|---:|
| Samples | 76 |
| Cell groups estimable in both halves | 175 / 186 |
| Genes | 38705 |
| Raw CPM split-half double-centered R | 0.8969 |
| Raw CPM estimated full-pseudobulk reliability | 0.9457 |
| log1p CPM split-half double-centered R | 0.5343 |
| log1p CPM estimated full-pseudobulk reliability | 0.6965 |

## Chromosome strata

CPM normalization remains genome-wide before genes are stratified. This table therefore measures target reliability on each chromosome in the same expression units used for evaluation.

| Chromosome | Genes | Raw CPM split-half R | Raw CPM full reliability | log1p CPM split-half R | log1p CPM full reliability |
|---|---:|---:|---:|---:|---:|
| chr1 | 3464 | 0.8995 | 0.9471 | 0.5497 | 0.7094 |
| chr10 | 1417 | 0.9110 | 0.9534 | 0.5339 | 0.6961 |
| chr11 | 2117 | 0.9024 | 0.9487 | 0.5291 | 0.6920 |
| chr12 | 1972 | 0.8887 | 0.9411 | 0.5420 | 0.7030 |
| chr13 | 805 | 0.8427 | 0.9146 | 0.5257 | 0.6891 |
| chr14 | 1507 | 0.9019 | 0.9484 | 0.5173 | 0.6819 |
| chr15 | 1307 | 0.8542 | 0.9214 | 0.5225 | 0.6864 |
| chr16 | 1709 | 0.9474 | 0.9730 | 0.4864 | 0.6544 |
| chr17 | 2055 | 0.8482 | 0.9178 | 0.5083 | 0.6740 |
| chr18 | 787 | 0.9120 | 0.9539 | 0.5479 | 0.7079 |
| chr19 | 2134 | 0.7069 | 0.8283 | 0.4760 | 0.6450 |
| chr2 | 2577 | 0.9020 | 0.9485 | 0.5609 | 0.7187 |
| chr20 | 990 | 0.8298 | 0.9070 | 0.5045 | 0.6707 |
| chr21 | 544 | 0.8496 | 0.9187 | 0.5091 | 0.6747 |
| chr22 | 930 | 0.8460 | 0.9166 | 0.4821 | 0.6505 |
| chr3 | 1940 | 0.9228 | 0.9599 | 0.5525 | 0.7117 |
| chr4 | 1549 | 0.9005 | 0.9476 | 0.5862 | 0.7392 |
| chr5 | 1831 | 0.8664 | 0.9284 | 0.5543 | 0.7132 |
| chr6 | 1877 | 0.8475 | 0.9174 | 0.5660 | 0.7228 |
| chr7 | 1734 | 0.8572 | 0.9231 | 0.5435 | 0.7043 |
| chr8 | 1525 | 0.8860 | 0.9396 | 0.5596 | 0.7176 |
| chr9 | 1340 | 0.9202 | 0.9585 | 0.5268 | 0.6901 |
| chrM | 13 | 0.4520 | 0.6226 | 0.1379 | 0.2423 |
| chrX | 1196 | 0.8197 | 0.9009 | 0.5242 | 0.6878 |
| chrY | 110 | 0.3357 | 0.5027 | 0.4577 | 0.6280 |
