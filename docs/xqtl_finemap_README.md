# XQTL single-cell brain eQTL fine-mapping (SuSiE PIPs)

SuSiE fine-mapping results for cis-eQTLs in six brain cell types. These are the files used for the XQTL variant-effect benchmark of the AlphaGenome v0data fine-tunes. They were copied unchanged from the lab's `mlxqtl_analysis` data release `susie_vars_pips_04_2024` (April 2024), cell-type sets named `*_mega_eQTL`.

The source release does not document the cohorts or the eQTL mapping pipeline. Sibling directories in that release hold cohort-specific results (for example DeJager and Kellis sets for the same cell types), which suggests the `mega` sets combine cohorts, but this is not confirmed. Ask the data owners before relying on cohort, sample size or covariate details.

## Contents

| Path | Contents |
|---|---|
| `<Cell>_mega_eQTL/PIP_all_parquet/PIP_all.parquet/chr=chrN/part.K.parquet` | per-variant, per-gene PIPs, hive-partitioned by chromosome |
| `<Cell>_mega_eQTL/PIP_top_parquet/PIP_top.parquet/chr=chrN/part.K.parquet` | top loci per gene with marginal effect sizes, MAF and credible-set membership |
| `derived/records.jsonl` | the 10,556 variant-gene records sampled for the benchmark |
| `derived/sampling_counts.json` | row counts and positive/negative counts per cell type after filtering |
| `derived/finemap_overlap.tsv` | high-PIP overlap with the SingleBrain fine-mapping panels |

Cell types are `Ast` (astrocyte), `Exc` (excitatory neuron), `Inh` (inhibitory neuron), `Mic` (microglia), `OPC` (oligodendrocyte precursor) and `Oli` (oligodendrocyte). Each has partitions for chromosomes 1 to 22; there is no chrX or chrY.

The `PIP_all_parquet` and `PIP_top_parquet` sets were copied. The source release also has `PIP_all` and `PIP_top` (per-gene CSV versions of the same tables), `PIP_all_other_parquet` and `PIP_top_other_parquet` (the same columns for a separate "other" variant set, including indels and `*` alleles) for each cell type; they were not copied.

## Columns of `PIP_all_parquet`

| Column | Type | Meaning |
|---|---|---|
| `variant_id` | string | `chr:pos:ref:alt`, e.g. `chr22:46320139:A:G` |
| `pos` | int64 | 1-based position |
| `ref`, `alt` | string | reference and alternate alleles |
| `pip` | double | SuSiE posterior inclusion probability for this variant and gene |
| `gene_id` | string | Ensembl gene ID without version |
| `chr` | dictionary | partition column, `chr1` to `chr22` |
| `__null_dask_index__` | int64 | index left by the writer; not meaningful |

Each row is one variant tested for one gene, so a variant appears once per gene whose cis window contains it. Coordinates are hg38: of 10,556 sampled single-nucleotide variant records, all but 104 matched the hg38 reference allele and could be scored.

Read one cell type with
```python
import pyarrow.dataset as ds
data = ds.dataset("Mic_mega_eQTL/PIP_all_parquet/PIP_all.parquet", format="parquet", partitioning="hive")
high = data.to_table(filter=ds.field("pip") > 0.75).to_pandas()
```

## Top loci with effect sizes (`PIP_top_parquet`)

`PIP_all_parquet` has no effect sizes. `PIP_top_parquet` has them for a subset of variants per gene, the "top loci" reported alongside the fine-mapping. It has the `PIP_all_parquet` columns plus the following.

| Column | Type | Meaning |
|---|---|---|
| `betahat`, `sebetahat` | double | effect estimate and standard error |
| `maf` | double | minor allele frequency |
| `cs_coverage_0.95`, `cs_coverage_0.7`, `cs_coverage_0.5` | int | credible set index (1 to 10) at 95%, 70% and 50% coverage; 0 means not in a credible set |
| `cs_coverage_0.95_min_corr`, `cs_coverage_0.7_min_corr`, `cs_coverage_0.5_min_corr` | int | the same after the minimum within-set correlation (purity) filter |

The column names follow the susieR summary-statistic convention, so `betahat` and `sebetahat` are likely marginal (single-variant) eQTL estimates. The release does not state the effect allele or the expression scale; check sign conventions before combining them with other effect estimates. About 19% of rows are indels.

| Cell type | Rows | Genes | PIP > 0.75 | In a 95% credible set |
|---|---:|---:|---:|---:|
| Ast | 141,758 | 6,023 | 914 | 100,956 |
| Exc | 293,569 | 9,143 | 2,864 | 231,739 |
| Inh | 204,486 | 7,764 | 1,369 | 151,866 |
| Mic | 45,449 | 2,303 | 249 | 31,723 |
| OPC | 87,629 | 3,707 | 514 | 62,532 |
| Oli | 159,241 | 6,367 | 1,271 | 116,104 |

PIP > 0.75 counts here exceed the benchmark positives below because they include indels and are not restricted to GENCODE genes or the 512 kb window.

## Benchmark use and counts

The benchmark keeps rows whose gene is in GENCODE v38 basic on the same chromosome, whose variant is a biallelic single-nucleotide variant, and that lie within 512 kb of the gene's TSS ("eligible"). Positives have PIP above 0.75 and negatives PIP below 0.01. All positives and up to 1,000 negatives per cell type were sampled with a fixed seed.

| Cell type | Rows | Eligible | Positives (PIP > 0.75) | Negatives (PIP < 0.01) |
|---|---:|---:|---:|---:|
| Ast | 167,399,779 | 31,850,543 | 581 | 31,769,295 |
| Exc | 218,708,555 | 41,150,681 | 1,784 | 40,976,915 |
| Inh | 207,868,448 | 39,394,880 | 813 | 39,281,800 |
| Mic | 70,865,864 | 14,332,701 | 168 | 14,305,646 |
| OPC | 122,683,778 | 24,317,596 | 358 | 24,269,005 |
| Oli | 178,552,659 | 33,830,282 | 852 | 33,738,029 |

`derived/records.jsonl` uses the SingleBrain panel names for each cell type (`Ast`, `Ext`, `IN`, `MG`, `OPC`, `OD`) and the Zemke 2023 model cell groups (`ASC`, `EXC`, `INH`, `MGC`, `OPC`, `ODC`).

## Agreement with SingleBrain fine-mapping

High-PIP (above 0.75) gene-variant pairs were compared with the broad SingleBrain panel for the same cell type after the same filters. Overlap is modest.

| Cell type | XQTL high-PIP | SingleBrain high-PIP | Shared | Jaccard | XQTL recall | SingleBrain recall |
|---|---:|---:|---:|---:|---:|---:|
| Ast | 581 | 1,231 | 172 | 0.105 | 0.296 | 0.140 |
| Exc | 1,784 | 3,420 | 504 | 0.107 | 0.283 | 0.147 |
| Inh | 813 | 1,838 | 236 | 0.098 | 0.290 | 0.128 |
| Mic | 168 | 668 | 62 | 0.080 | 0.369 | 0.093 |
| OPC | 358 | 929 | 122 | 0.105 | 0.341 | 0.131 |
| Oli | 852 | 1,820 | 283 | 0.118 | 0.332 | 0.155 |

"XQTL recall" is the fraction of XQTL high-PIP pairs also high-PIP in SingleBrain; "SingleBrain recall" is the reverse.
