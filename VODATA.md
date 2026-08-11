# NYGC genomic resource overview

This note summarizes the Genomic Resource Repositories, GRRs, maintained for the New York Genome Center, NYGC, artificial-intelligence initiative. Counts reflect the collection inspected on 11 August 2026. A GRR contains primary data plus resource configuration, indexes, statistics, and generated web pages, so the number of files is much larger than the number of biological datasets.

## Repository families

| Repository | Scope | Main uses |
|---|---|---|
| `SC_Summaries_GRR` | Four single-cell and multiomic studies with primary matrices, fragments, pseudobulk genomic tracks, and supplementary tables | Fine-tuning targets, cross-species comparisons, target reconstruction |
| `grr_encode` | Curated Encyclopedia of DNA Elements, ENCODE, experiments | General functional-genomics pretraining, evaluation, and annotation |
| `iossifov_lab_grr_mirror` | Reference genomes, gene models, scores, variant resources, gene sets, and annotation pipelines | Coordinate systems, covariates, variant annotation, and liftover |

## Single-cell studies

### Prediction-target suitability

A valid sequence-to-signal target here means an experimentally measured signal associated with genomic coordinates in a known reference assembly. Gene-expression matrices remain valid for gene-level supervision, but they are not base-resolution RNA coverage. Predictions and attribution scores from another model are not ground-truth assay targets.

| Dataset | ATAC target | RNA target | Other targets | Main qualification |
|---|---|---|---|---|
| `johansen2025Crossspecies` | Yes, released pseudobulk BigWigs or targets reconstructed from fragments | Gene-level only | None in the core release | Exon pseudo-coverage may be derived from expression, but is not measured RNA coverage |
| `liu2026Multiomics` | Yes, preferably reconstructed from fragments | Gene-level only | None in the core release | The released observed track is a nonlinear significance score; the predicted and contribution tracks are model outputs |
| `zemke2023Conserved` | Yes, 20 published cell types per species | Yes, 20 published cell types per species | Hi-C and methylation | Chandelier-cell ATAC and RNA tracks were not published; do not treat them as failed required downloads |
| `zemke2024Epigenetic` | Yes, released pseudobulks or targets reconstructed from fragments | Yes, released pseudobulk BigWigs, plus gene-level matrices | None in the core release | Some age-by-cell-type combinations are absent and require a validity mask rather than zero targets |

All four studies therefore provide useful measured prediction targets, but only `zemke2023Conserved` and `zemke2024Epigenetic` provide released coordinate-resolved RNA tracks. For joint training, target transformations, genome assemblies, strand conventions, cell-group definitions, and missing-channel masks must be represented explicitly.

### `johansen2025Crossspecies`

This is a cross-species basal-ganglia atlas for human, macaque, and marmoset. It is the source of the Allen Human Multiome Brain Atlas data used in the current fine-tuning work.

| Species | Cell-group assay for transposase-accessible chromatin tracks | Fragment libraries | Comprehensive expression matrix |
|---|---:|---:|---:|
| Human | 60 | 204 | 1 |
| Macaque | 58 | 90 | 1 |
| Marmoset | 51 | 112 | 1 |

The assay for transposase-accessible chromatin, ATAC, tracks are cell-group pseudobulk BigWigs. Fragment resources are tabix-indexed five-column records with chromosome, start, end, cell identifier, and multiplicity; per-cell whole-genome fragment histograms are stored alongside them. Marmoset has 112 distinct libraries, each represented by a full fragment resource and an H5AD-filtered exact subset, yielding 224 resource directories but not 224 independent libraries. Reprocessing uses the filtered copy because it retains every fragment for cells in the comprehensive metadata and avoids double-counting the same cells. The comprehensive expression matrices use the AnnData Hierarchical Data Format 5, H5AD, format and contain cell identifiers, group assignments, and expression. Additional resources include aligned 10x multiome matrices, expression metadata, taxonomy mappings, spatial-expression data from MERSCOPE and Xenium platforms, macaque Patch-seq data, and supplementary tables.

This is the strongest collection for aligned primate basal-ganglia training. The fragment records support rebuilding targets with controlled normalization, while the released BigWigs provide a direct baseline. Species use different reference assemblies and should not share genomic coordinates without an explicit orthology or liftover mapping.

### `liu2026Multiomics`

This is Liu et al.'s Human Development Multiomic Atlas, HDMA, a human fetal multi-organ atlas generated with simultaneous high-throughput ATAC and ribonucleic-acid expression sequencing, SHARE-seq. It contains 82 matched sample-level ATAC fragment resources and 82 cell-expression matrices spanning approximately 10 to 23 post-conception weeks.

HDMA is not the earlier dataset called the Human Developmental Atlas, HDA, in this repository's fine-tuning experiments. That HDA shorthand refers to Mannens et al.'s first-trimester human brain chromatin-accessibility atlas, which covers the whole brain at 6 to 13 post-conception weeks and uses 10x Genomics single-cell ATAC and multiome assays. HDMA covers 12 fetal organs at 10 to 23 post-conception weeks and uses SHARE-seq. The similar names do not indicate shared samples or targets. The studies are described in the [HDMA article](https://www.nature.com/articles/s41586-026-10326-9) and [first-trimester brain article](https://www.nature.com/articles/s41586-024-07234-1).

The repository covers adrenal gland, brain, eye, heart, liver, lung, muscle, skin, spleen, stomach, thymus, and thyroid. Its 186 organ-specific cell clusters each have three BigWig tracks, for 558 tracks total:

| Track | Meaning |
|---|---|
| `obs_pval_signal` | Observed accessibility significance from Model-based Analysis of ChIP-Seq version 2, MACS2 |
| `mean_pred_corrected` | Bias-corrected accessibility predicted by ChromBPNet |
| `mean_counts_contribs` | Base-resolution contribution scores for the ChromBPNet counts head, computed with Deep Learning Important FeaTures, DeepLIFT |

This collection is useful for developmental and multi-organ sequence modeling. The observed track is a transformed significance signal rather than depth-normalized coverage, while the prediction and contribution tracks are model outputs. The raw fragment resources are the preferred source for constructing measured accessibility targets with a consistent normalization. The expression matrices support gene-level RNA supervision, not direct base-resolution RNA coverage.

### `zemke2023Conserved`

This study compares motor cortex across human, macaque, marmoset, and mouse. It is the broadest cross-species multimodal collection in the GRR.

| Modality | Human | Macaque | Marmoset | Mouse | Total |
|---|---:|---:|---:|---:|---:|
| Pseudobulk ATAC BigWigs | 20 | 20 | 20 | 20 | 80 |
| Pseudobulk ribonucleic-acid BigWigs | 20 | 20 | 20 | 20 | 80 |
| Pseudobulk high-throughput chromosome conformation capture, Hi-C, maps | 20 | 20 | 20 | 20 | 80 |
| Pseudobulk methylated-cytosine bedGraphs | 40 | 40 | 40 | 40 | 160 |

The collection also contains 27 cell-expression H5AD matrices organized by species and sample. The published ATAC and RNA tracks use reads per kilobase per million mapped reads, RPKM, normalization. The WashU source directories contain exactly the 20 downloaded ATAC and 20 downloaded RNA tracks for each species; the attempted chandelier-cell URLs do not exist. The study reports that low-coverage cell types, including chandelier interneurons, were excluded from sequence-model training and evaluation. The eight absent chandelier-cell files are therefore unpublished targets rather than recoverable download failures.

The local Gene Expression Omnibus accession GSE229169 matrices do contain chandelier-cell observations, gene counts, and sample-level ATAC peak counts. These permit derived gene-level RNA and peak-restricted ATAC pseudobulks, but not exact reconstruction of the missing genome-wide RPKM BigWigs. Such targets must be labeled as derived and should not be silently appended to the 20 published tracks. Exact whole-genome reconstruction would require reprocessing the raw sequencing reads with the original alignment and track-generation pipeline; low cell counts make that result less reliable. The recommended fix is to define the released ATAC/RNA panel as 20 cell types and remove the eight nonexistent URLs from required-download checks. Hi-C and methylation retain their published `Pvalb-ChC` channels.

This is the preferred collection for studying conservation across modalities. Track names are aligned at the cell-type level, but coordinate systems, sequence references, assay normalization, and modality-specific cell-type availability remain species-specific. The source study and raw multiome accession are described in the [Nature article](https://www.nature.com/articles/s41586-023-06819-6) and [Gene Expression Omnibus GSE229169 record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE229169).

### `zemke2024Epigenetic`

This study profiles epigenetic and three-dimensional-genome changes during human hippocampal aging. It contains 40 donor-level ATAC fragment resources and 40 donor-level ribonucleic-acid expression matrices.

Pseudobulk ATAC and ribonucleic-acid tracks cover 22 cell types. Each modality contains 92 BigWigs, consisting of 22 all-age tracks plus age-stratified tracks for ages 20–40, 40–60, 60–80, and 80–100. The age strata contain 18, 18, 18, and 16 cell types, respectively, so not every cell type is represented in every age interval.

This collection is suitable for age-aware human hippocampus models and for comparing fragment-derived targets with released pseudobulks. Donor identity and age interval should remain explicit covariates rather than being pooled without evaluation.

## ENCODE collection

The ENCODE GRR organizes one resource directory per ENCODE experiment accession. The current directory contains:

| Assay | Experiment directories |
|---|---:|
| ATAC sequencing | 369 |
| Deoxyribonuclease sequencing | 1,407 |
| Histone chromatin-immunoprecipitation sequencing | 2,943 |
| Transcription-factor chromatin-immunoprecipitation sequencing | 3,203 |

These resources provide broad tissue, cell-line, and regulatory-assay coverage. They are useful for adding diverse genomic supervision, but experiment metadata, genome assembly, strand semantics, signal transformation, controls, and replicate structure must be checked before constructing a shared head.

## Annotation-resource mirror

The broader Iossifov Lab mirror includes resources for the hg19, hg38, and telomere-to-telomere hs1 human assemblies. Major categories include reference genomes, gene models, conservation and pathogenicity scores, population variant frequencies, copy-number-variant collections, enrichment backgrounds, gene properties, gene sets, coordinate liftover chains, and reusable annotation pipelines.

This mirror is primarily supporting infrastructure rather than direct model supervision. It can provide sequence references, blacklist or annotation covariates, gene definitions, variant evaluation sets, and coordinate conversions. Assembly identity must be verified before combining any resource with a sequence or target track.

## File conventions

| Format | Typical content |
|---|---|
| `.bw` | BigWig genomic signal tracks |
| `.gz` with `.tbi` | Block-gzip-compressed, tabix-indexed fragments or bedGraph-like tables |
| `.h5ad` | AnnData matrices with observations, features, and metadata |
| `.h5` | Hierarchical Data Format 5 matrices |
| `.hic` | Binary Hi-C contact maps |
| `.csv`, `.txt`, `.xlsx` | Metadata and supplementary tables |
| `genomic_resource.yaml` | Resource type, payload mapping, labels, provenance, and display metadata |
| `.MANIFEST`, `statistics/`, `.html`, `.png` | Generated integrity, summary, and browsing artifacts |
| `.dvc` | Data Version Control pointer for a managed large payload |

Generated statistics and web files should not be counted as independent biological datasets. Before use, verify that the primary payload is materialized rather than only represented by a Data Version Control pointer.

## Selection guidance

For the current Allen basal-ganglia work, use `johansen2025Crossspecies` human fragments plus the comprehensive human expression matrix when rebuilding ATAC targets. Use the released 60 human BigWigs as the reference baseline. The macaque and marmoset resources support joint training only after species-specific references and homologous genomic splits are configured.

For broader cross-species modeling, `zemke2023Conserved` offers matched ATAC, ribonucleic-acid, methylation, and Hi-C modalities. Use the 20-cell-type intersection for published ATAC/RNA supervision; either mask chandelier ATAC/RNA or exclude chandelier channels when comparing all modalities. For developmental breadth, `liu2026Multiomics` offers fetal multi-organ fragments and expression, but its released genomic tracks include transformed and model-derived signals. For aging, `zemke2024Epigenetic` offers donor-level human hippocampus fragments, expression, and age-stratified pseudobulks.

Across all studies, compare native bin width, sparsity, normalization, assay transformation, genome assembly, cell grouping, and biological domain before combining targets. Similar file formats do not imply comparable numerical targets.
