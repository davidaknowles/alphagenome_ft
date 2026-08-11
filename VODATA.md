# NYGC genomic resource overview

This note summarizes the Genomic Resource Repositories, GRRs, maintained for the New York Genome Center, NYGC, artificial-intelligence initiative. Counts reflect the collection inspected on 11 August 2026. A GRR contains primary data plus resource configuration, indexes, statistics, and generated web pages, so the number of files is much larger than the number of biological datasets.

## Repository families

| Repository | Scope | Main uses |
|---|---|---|
| `SC_Summaries_GRR` | Four single-cell and multiomic studies with primary matrices, fragments, pseudobulk genomic tracks, and supplementary tables | Fine-tuning targets, cross-species comparisons, target reconstruction |
| `grr_encode` | Curated Encyclopedia of DNA Elements, ENCODE, experiments | General functional-genomics pretraining, evaluation, and annotation |
| `iossifov_lab_grr_mirror` | Reference genomes, gene models, scores, variant resources, gene sets, and annotation pipelines | Coordinate systems, covariates, variant annotation, and liftover |

## Single-cell studies

### `johansen2025Crossspecies`

This is a cross-species basal-ganglia atlas for human, macaque, and marmoset. It is the source of the Allen Human Multiome Brain Atlas data used in the current fine-tuning work.

| Species | Cell-group assay for transposase-accessible chromatin tracks | Fragment libraries | Comprehensive expression matrix |
|---|---:|---:|---:|
| Human | 60 | 204 | 1 |
| Macaque | 58 | 90 | 1 |
| Marmoset | 51 | 224 | 1 |

The assay for transposase-accessible chromatin, ATAC, tracks are cell-group pseudobulk BigWigs. Fragment resources are tabix-indexed five-column records with chromosome, start, end, cell identifier, and multiplicity; per-cell whole-genome fragment histograms are stored alongside them. The comprehensive expression matrices use the AnnData Hierarchical Data Format 5, H5AD, format and contain cell identifiers, group assignments, and expression. Additional resources include aligned 10x multiome matrices, expression metadata, taxonomy mappings, spatial-expression data from MERSCOPE and Xenium platforms, macaque Patch-seq data, and supplementary tables.

This is the strongest collection for aligned primate basal-ganglia training. The fragment records support rebuilding targets with controlled normalization, while the released BigWigs provide a direct baseline. Species use different reference assemblies and should not share genomic coordinates without an explicit orthology or liftover mapping.

### `liu2026Multiomics`

This is a human fetal multi-organ atlas generated with simultaneous high-throughput ATAC and ribonucleic-acid expression sequencing, SHARE-seq. It contains 82 matched sample-level ATAC fragment resources and 82 cell-expression matrices spanning approximately 10 to 23 post-conception weeks.

The repository covers adrenal gland, brain, eye, heart, liver, lung, muscle, skin, spleen, stomach, thymus, and thyroid. Its 186 organ-specific cell clusters each have three BigWig tracks, for 558 tracks total:

| Track | Meaning |
|---|---|
| `obs_pval_signal` | Observed accessibility significance from Model-based Analysis of ChIP-Seq version 2, MACS2 |
| `mean_pred_corrected` | Bias-corrected accessibility predicted by ChromBPNet |
| `mean_counts_contribs` | Base-resolution contribution scores for the ChromBPNet counts head, computed with Deep Learning Important FeaTures, DeepLIFT |

This collection is useful for developmental and multi-organ sequence modeling. The observed track is a transformed significance signal rather than a depth-normalized coverage target, while the prediction and contribution tracks are model outputs. They should not be mixed as interchangeable supervision channels.

### `zemke2023Conserved`

This study compares motor cortex across human, macaque, marmoset, and mouse. It is the broadest cross-species multimodal collection in the GRR.

| Modality | Human | Macaque | Marmoset | Mouse | Total |
|---|---:|---:|---:|---:|---:|
| Pseudobulk ATAC BigWigs | 20 | 20 | 20 | 20 | 80 |
| Pseudobulk ribonucleic-acid BigWigs | 20 | 20 | 20 | 20 | 80 |
| Pseudobulk high-throughput chromosome conformation capture, Hi-C, maps | 20 | 20 | 20 | 20 | 80 |
| Pseudobulk methylated-cytosine bedGraphs | 40 | 40 | 40 | 40 | 160 |

The collection also contains 27 cell-expression H5AD matrices organized by species and sample. Its local README reports that chandelier-cell ATAC and ribonucleic-acid BigWigs failed to download for all four species, so those eight tracks should be treated as known omissions.

This is the preferred collection for studying conservation across modalities. Track names are aligned at the cell-type level, but coordinate systems, sequence references, and assay normalization remain species-specific.

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

For broader cross-species modeling, `zemke2023Conserved` offers matched ATAC, ribonucleic-acid, methylation, and Hi-C modalities. For developmental breadth, `liu2026Multiomics` offers fetal multi-organ fragments and expression, but its released genomic tracks include transformed and model-derived signals. For aging, `zemke2024Epigenetic` offers donor-level human hippocampus fragments, expression, and age-stratified pseudobulks.

Across all studies, compare native bin width, sparsity, normalization, assay transformation, genome assembly, cell grouping, and biological domain before combining targets. Similar file formats do not imply comparable numerical targets.
