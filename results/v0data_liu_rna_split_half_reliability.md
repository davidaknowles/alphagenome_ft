# Liu RNA split-half reliability

Support: `fixed_window_full_span`.

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
| chr1 | 2581 | 0.8339 | 0.9095 | 0.5099 | 0.6754 |
| chr10 | 1003 | 0.6993 | 0.8230 | 0.4673 | 0.6369 |
| chr11 | 1628 | 0.9037 | 0.9494 | 0.4903 | 0.6579 |
| chr12 | 1435 | 0.7199 | 0.8372 | 0.4774 | 0.6463 |
| chr13 | 546 | 0.6549 | 0.7915 | 0.4592 | 0.6294 |
| chr14 | 1181 | 0.8136 | 0.8972 | 0.4747 | 0.6438 |
| chr15 | 944 | 0.7083 | 0.8293 | 0.4802 | 0.6489 |
| chr16 | 1371 | 0.6126 | 0.7598 | 0.4510 | 0.6216 |
| chr17 | 1657 | 0.8594 | 0.9244 | 0.4867 | 0.6547 |
| chr18 | 564 | 0.6752 | 0.8061 | 0.4547 | 0.6251 |
| chr19 | 1814 | 0.6331 | 0.7753 | 0.4629 | 0.6328 |
| chr2 | 1769 | 0.8068 | 0.8931 | 0.4816 | 0.6501 |
| chr20 | 767 | 0.7061 | 0.8277 | 0.4644 | 0.6342 |
| chr21 | 407 | 0.5481 | 0.7081 | 0.4551 | 0.6255 |
| chr22 | 782 | 0.6451 | 0.7843 | 0.4525 | 0.6231 |
| chr3 | 1283 | 0.8417 | 0.9141 | 0.4957 | 0.6629 |
| chr4 | 993 | 0.6607 | 0.7957 | 0.5246 | 0.6882 |
| chr5 | 1213 | 0.6685 | 0.8013 | 0.4928 | 0.6602 |
| chr6 | 1336 | 0.6717 | 0.8036 | 0.5118 | 0.6771 |
| chr7 | 1252 | 0.8249 | 0.9040 | 0.4848 | 0.6530 |
| chr8 | 1067 | 0.8851 | 0.9390 | 0.5008 | 0.6673 |
| chr9 | 972 | 0.7621 | 0.8650 | 0.4742 | 0.6433 |
| chrX | 847 | 0.3695 | 0.5396 | 0.4755 | 0.6445 |
| chrY | 87 | 0.3041 | 0.4664 | 0.3721 | 0.5424 |
