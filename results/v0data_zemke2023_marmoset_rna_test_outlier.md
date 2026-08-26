# Zemke 2023 marmoset RNA test outlier audit

RNA target tracks are summarized in equal-width genomic bins. Double-centered standard deviation (DC SD) and variance fractions refer to the target after centering over genomic bins and cell-subclass tracks.

| Species | Chromosome | Nonzero | DC SD | Median track R | Max locus variance | Max track variance |
|---|---|---:|---:|---:|---:|---:|
| human | chr8 | 0.9355 | 1.5049 | 0.5704 | 0.0349 | 0.1714 |
| human | chr9 | 0.8501 | 1.4202 | 0.6472 | 0.3387 | 0.2445 |
| macaque | chr8 | 0.9554 | 1.4798 | 0.6149 | 0.1526 | 0.1837 |
| macaque | chr9 | 0.9544 | 1.5201 | 0.4801 | 0.1201 | 0.1760 |
| marmoset | chr8 | 0.9655 | 1.3272 | 0.6524 | 0.0731 | 0.2214 |
| marmoset | chr9 | 0.9646 | 2.3134 | 0.9106 | 0.7069 | 0.4698 |
| mouse | chr8 | 0.9158 | 1.8399 | 0.5843 | 0.0976 | 0.2404 |
| mouse | chr9 | 0.9224 | 1.7046 | 0.7319 | 0.1378 | 0.3191 |

## ANO6 expression

| Species | Chromosome | Max CPM | Max group | Median CPM |
|---|---|---:|---|---:|
| human | chr12 | 122.49 | OPC | 44.83 |
| macaque | chr11 | 188.75 | VLMC | 68.34 |
| marmoset | chr9 | 7951.52 | Endo | 2295.91 |
| mouse | not matched | NA | NA | NA |

The ribosomal repeat contains the chromosome-wide maximum in 20 of 20 marmoset RNA tracks. Repeat-region maxima range from 23345.7 to 106835.0 RPKM, with the largest value in Endo.

The coarse bin overlapping the SSU-rRNA_Hsa repeat inside marmoset ANO6 accounts for 70.7% of chromosome-9 double-centered target variance.

## Matched checkpoint evaluation

| Target | Baseline R | Repeat-window excluded R | Difference |
|---|---:|---:|---:|
| ATAC | 0.6046 | 0.6156 | +0.0110 |
| RNA | 0.1570 | 0.6291 | +0.4720 |

Both evaluations restore the same selected checkpoint. The baseline uses all 1,022 chromosome-9 windows; the diagnostic omits only the one 131 kb window overlapping the ribosomal repeat.
