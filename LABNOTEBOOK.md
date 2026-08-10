# Lab Notebook

## Model and fine-tuning components

AlphaGenome uses a convolutional encoder, transformer tower, decoder, and assay-specific genomic output heads. The current Allen Brain Multiome experiment trains two new heads together with LoRA adapters on linear layers and LoCon adapters on selected convolutions. Frozen backbone parameters are stored in bfloat16, while adapter and head optimization remains compatible with bfloat16 forward compute.

## Allen Brain Multiome v0

The human dataset contains paired cell-type pseudobulks for 60 groups. ATAC supervision is supplied as one unstranded base-resolution BigWig per group. RNA supervision is available as a dense group-by-gene matrix rather than read coverage, so it cannot directly supervise the RNA-seq head at base resolution.

The initial RNA targets used paired positive- and negative-strand pseudo-coverage tracks for every group. For a gene with expression \(c\) CPM and annotated body length \(L\) bp, every base in the gene body received density \(c/L\). Although each gene integral was \(c\), this representation marked introns as expressed and evaluated RNA over every genomic base.

The ATAC and RNA heads contain 60 and 120 channels, respectively. Chromosome holdouts prevent sequence windows from crossing training, validation, and test partitions.

The five-epoch run was stopped during epoch 3 because RNA did not learn a useful signal. At epoch 2, validation RNA global R2 was -0.878 and differential Pearson correlation was 0.0008. Human ATAC did learn, with validation global R2 0.033, R2 over loci 0.378, and differential Pearson correlation 0.106. The contrast indicated a problem with RNA target representation and evaluation rather than a general failure of LoRA and LoCon training.

## Allen Brain Multiome v1 exon supervision

The revised representation merges all annotated exons for each stable Ensembl gene identifier. A gene with union-exon length \(L_e\) bp and expression \(c\) CPM contributes density \(c/L_e\) only to its exonic bases. Overlapping genes add within a strand and opposite strands remain separate. The integral of an isolated gene's pseudo-coverage remains \(c\), introns are zero, and track-specific nonzero means are passed to the AlphaGenome RNA head for target scaling. Preparation matched 36,474 of 36,601 expression genes, 99.65%.

RNA also receives direct gene-level supervision from the 128 bp output. For batch size \(B\), sequence-bin count \(S_{128}=S/128\), at most \(G\) genes per window, and \(C=60\) cell groups, exon weights have dimension \([B,S_{128},G]\), gene targets have dimension \([B,G,C]\), and the validity mask has dimension \([B,G]\). Each weight is the fraction of a 128 bp bin covered by a gene's union exons. Positive- and negative-strand predictions are separated from the interleaved 120 RNA channels, weighted over bins, summed to \([B,G,C]\), and selected using each gene's strand. Only genes whose full union-exon span is contained in the input window enter the gene objective and metrics, preventing partial windows from being compared with full-gene CPM. Across the complete human 131 kb split, the maximum is 70 genes per window.

The RNA objective is log1p mean-squared error over valid gene and cell-group pairs, plus 0.1 times the standard exon pseudo-coverage loss. Gene-level metrics use the same valid mask, so padded genes and intergenic bases do not affect R2. The exon coverage term remains an auxiliary constraint on genomic localization and strand.

## Metrics

Predictions and targets have shape ([B,S,C]), where (B) is the number of windows, (S) is the number of genomic positions or 128 bp bins, and (C) is the number of tracks. Differential Pearson correlation first reshapes the first two axes into observations, then subtracts the mean of each track and each observation from both matrices while adding back the grand mean. `double_centered_r2` is the square of this correlation. It measures agreement in cell-type-specific spatial variation after removing locus-wide and track-wide offsets, and it is logged for train, validation, and test splits after every epoch.

## Findings

The v0 one-step 131 kb LoRA+LoCon smoke run completed with finite joint loss 3.9817 and logged `double_centered_r2` for both heads. Best-checkpoint selection uses signed differential Pearson correlation rather than its square so a strong anticorrelation cannot be selected as an improvement.
