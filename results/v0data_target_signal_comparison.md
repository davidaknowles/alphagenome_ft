# Target signal comparison

This audit characterizes prediction targets before model fitting. Assay for transposase-accessible chromatin, ATAC, and RNA sequencing, RNA, tracks are expanded exactly as in training, uncovered bases are zero, and each 131,072 bp window is averaged into 1,024 bins of 128 bp. Double-centered, DC, values remove the mean of every genomic bin and every track and add back the grand mean. `DC var.` is the fraction of total centered sum of squares retained after double centering. `Track R` is the median ordinary Pearson correlation between target tracks.

HDA, Liu, and Johansen use eight sampled windows per manifest. Zemke uses 32. Absolute standard deviations are comparable only within a shared target construction and unit. In particular, the Zemke RNA tracks represent released coordinate-resolved coverage, whereas HDA, Liu, and Johansen RNA tracks are exon-density auxiliaries derived from gene-level counts per million.

| Dataset | Modality | Tracks | Nonzero | SD | DC SD | DC var. | Track R |
|---|---|---:|---:|---:|---:|---:|---:|
| HDA | ATAC | 134 | 63.65% | 4.0417 | 2.7347 | 45.78% | 0.718 |
| HDA | RNA | 268 | 2.92% | 0.000972 | 0.000927 | 90.97% | 0.098 |
| Liu human | ATAC | 186 | 69.01% | 0.4887 | 0.4366 | 79.82% | 0.451 |
| Liu human | RNA | 372 | 3.76% | 0.001777 | 0.001412 | 63.11% | 0.280 |
| Johansen human | ATAC | 47 | 74.44% | 0.0938 | 0.0665 | 50.23% | 0.637 |
| Johansen human | RNA | 94 | 2.81% | 0.001780 | 0.001231 | 47.88% | 0.263 |
| Johansen macaque | ATAC | 47 | 87.70% | 0.1216 | 0.0697 | 32.90% | 0.764 |
| Johansen macaque | RNA | 94 | 1.67% | 0.002392 | 0.001787 | 55.83% | -0.006 |
| Johansen marmoset | ATAC | 47 | 79.79% | 0.0816 | 0.0522 | 40.91% | 0.698 |
| Johansen marmoset | RNA | 94 | 1.04% | 0.002499 | 0.001842 | 54.34% | -0.003 |
| Zemke human | ATAC | 20 | 78.37% | 12.3037 | 8.5873 | 48.71% | 0.763 |
| Zemke human | RNA | 20 | 28.77% | 4.9957 | 3.8067 | 58.06% | 0.698 |
| Zemke macaque | ATAC | 20 | 73.07% | 10.6367 | 5.3615 | 25.41% | 0.805 |
| Zemke macaque | RNA | 20 | 31.85% | 13.5187 | 7.5037 | 30.81% | 0.709 |
| Zemke marmoset | ATAC | 20 | 85.55% | 6.3930 | 3.2092 | 25.20% | 0.782 |
| Zemke marmoset | RNA | 20 | 31.35% | 6.5410 | 3.9128 | 35.78% | 0.836 |
| Zemke mouse | ATAC | 20 | 80.03% | 10.6541 | 6.2940 | 34.90% | 0.724 |
| Zemke mouse | RNA | 20 | 26.07% | 7.8227 | 5.0678 | 41.97% | 0.738 |

## Interpretation

The synthetic exon RNA magnitude does not explain why reconstructed studies underperform HDA. Liu and every Johansen species have larger RNA SD and DC SD than HDA, yet HDA has the strongest model correlation. Scalar amplification would therefore move these targets farther from HDA in absolute variation, and earlier Allen scaling screens already showed that amplification reduced validation correlation.

HDA RNA retains 91.0% of centered variance after double centering, compared with 63.1% for Liu and 47.9% to 55.8% for Johansen. This confirms that the evaluated contrast is present in every dataset, but the retained fraction is not a predictability score. It does not establish that sequence explains that contrast, and Pearson correlation is invariant to a positive scalar.

Released Zemke RNA is an order of magnitude denser than synthetic exon-density RNA. That difference reflects target representation rather than a recoverable scalar mismatch. For datasets measured only at gene level, direct 128 bp gene aggregation is the defensible primary objective; synthetic exon density is at most an auxiliary localization term. The queued gene-only comparisons test this conclusion without discarding measured expression counts.

ATAC units differ across source pipelines, but their scale-free structure also provides no single normalization rule. HDA and Johansen human have similar DC variance fractions despite a large SD difference, while Liu has the largest DC fraction and the lowest median cross-track correlation. This supports retaining each study's native library-normalized units and evaluating preprocessing through held-out DC Pearson correlation rather than forcing matched marginal distributions.

## Pseudobulk depth

Normalized target quality remains depth-dependent. Across Liu and each Johansen species, Spearman correlation between log10 fragment depth and nonzero 128 bp coverage is 0.95 to 0.99. Correlation with each track's median correlation to the other tracks is 0.48 to 0.62. In contrast, log depth has a weakly negative correlation, -0.15 to -0.42, with log root mean square signal, consistent with signal-per-million normalization amplifying sampling noise in shallow pseudobulks rather than restoring their information content.

| Dataset | Tracks | <5M fragments | <10M | <25M | Depth vs nonzero | Depth vs track R | Depth vs RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Liu human | 186 | 15 | 46 | 95 | 0.953 | 0.618 | -0.152 |
| Johansen human | 47 | 3 | 3 | 10 | 0.992 | 0.500 | -0.183 |
| Johansen macaque | 47 | 1 | 6 | 10 | 0.991 | 0.537 | -0.264 |
| Johansen marmoset | 47 | 2 | 6 | 14 | 0.992 | 0.480 | -0.418 |

HDA peak-calling pseudobulks were downsampled to 25 million fragments, but imposing that threshold on these datasets would retain only 91 Liu groups and 29 Johansen groups shared across species. A conservative 10-million-fragment screen retains 140 Liu groups and 38 Johansen groups. It filters ATAC and paired RNA channels synchronously, retains the corresponding direct gene-expression rows, and combines the filter with gene-only RNA supervision. This is an explicit target-quality experiment rather than an attempt to rescale low-depth observations.
