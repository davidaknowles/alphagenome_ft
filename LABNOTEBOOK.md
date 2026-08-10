# Lab Notebook

## Model and fine-tuning components

AlphaGenome uses a convolutional encoder, transformer tower, decoder, and assay-specific genomic output heads. The current Allen Brain Multiome experiment trains two new heads together with LoRA adapters on linear layers and LoCon adapters on selected convolutions. Frozen backbone parameters are stored in bfloat16, while adapter and head optimization remains compatible with bfloat16 forward compute.

## Allen Brain Multiome v0

The human dataset contains paired cell-type pseudobulks for 60 groups. ATAC supervision is supplied as one unstranded base-resolution BigWig per group. RNA supervision is available as a dense group-by-gene matrix rather than read coverage, so it cannot directly supervise the RNA-seq head at base resolution.

RNA targets are represented as paired positive- and negative-strand pseudo-coverage tracks for every group. Stable Ensembl gene identifiers are matched to GENCODE gene records. For a gene with expression (c) CPM and annotated body length (L) bp, every base in the gene body receives density (c/L). The integral over the body is therefore (c), independent of gene length. Contributions from overlapping genes on the same strand add, while opposite-strand genes remain in separate channels. This retains genomic and strand information without claiming unavailable isoform, exon, or 3-prime coverage structure. The main limitation is that uniform gene-body density is a surrogate for RNA-seq coverage; results should be interpreted as cell-type differential gene-expression prediction rather than read-profile reconstruction.

The ATAC and RNA heads contain 60 and 120 channels, respectively. Chromosome holdouts prevent sequence windows from crossing training, validation, and test partitions.

## Metrics

Predictions and targets have shape ([B,S,C]), where (B) is the number of windows, (S) is the number of genomic positions or 128 bp bins, and (C) is the number of tracks. Differential Pearson correlation first reshapes the first two axes into observations, then subtracts the mean of each track and each observation from both matrices while adding back the grand mean. `double_centered_r2` is the square of this correlation. It measures agreement in cell-type-specific spatial variation after removing locus-wide and track-wide offsets, and it is logged for train, validation, and test splits after every epoch.

## Findings

Preparation matched all 60 RNA groups to ATAC tracks and matched 36,474 of 36,601 expression genes to GENCODE, 99.65%. A one-step 131 kb LoRA+LoCon smoke run completed with finite joint loss 3.9817 and successfully logged `double_centered_r2` for both heads on validation and test windows. The near-zero correlations and unstable conventional (R^2) values are expected from randomly initialized heads after one update and one evaluation window; they are not accuracy estimates. Best-checkpoint selection uses signed differential Pearson correlation rather than its square so a strong anticorrelation cannot be selected as an improvement.
