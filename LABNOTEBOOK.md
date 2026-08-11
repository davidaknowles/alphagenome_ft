# Lab Notebook

## GRR prediction-target audit

Excluding the Encyclopedia of DNA Elements collection, all four single-cell Genomic Resource Repository, GRR, studies contain experimentally measured targets suitable for some form of sequence-model supervision. `johansen2025Crossspecies` has released assay for transposase-accessible chromatin, ATAC, BigWigs and raw fragments, while its RNA measurements are gene-level. `liu2026Multiomics` has raw ATAC fragments and gene-level RNA; its released observed BigWig is a Model-based Analysis of ChIP-Seq version 2 significance transformation, and its other two BigWigs are ChromBPNet predictions and Deep Learning Important FeaTures attributions rather than assay targets. `zemke2023Conserved` has coordinate-resolved ATAC, RNA, high-throughput chromosome conformation capture, and methylation targets. `zemke2024Epigenetic` has coordinate-resolved ATAC and RNA targets, with missing age-by-cell-type combinations that require masks.

The eight reported `zemke2023Conserved` download failures are four-species ATAC and RNA tracks for chandelier cells. Inspection of the WashU source directories showed 20 published tracks per species and no chandelier-cell files under the attempted names. The paper states that low-coverage cell types, including chandelier interneurons, were removed from sequence-model training and evaluation. Local Gene Expression Omnibus accession GSE229169 matrices contain chandelier-cell observations and can produce gene-level RNA and peak-restricted ATAC pseudobulks, but these would not reproduce the missing genome-wide reads-per-kilobase-per-million BigWigs. The compatible resolution is to declare the 20-track ATAC/RNA release complete and remove chandelier cells from required-download checks. Any matrix-derived chandelier targets should be a separately labeled experiment; full track reconstruction would require raw-read reprocessing and remains limited by low cell counts.

## Allen and Human Brain Development ATAC preprocessing

The Allen adult basal ganglia and Human Brain Development, HDA, assay for transposase-accessible chromatin, ATAC, pseudobulks were produced by different pipelines and represent different biological domains. The [HDA source study](https://www.nature.com/articles/s41586-024-07234-1) describes developmental pseudobulks downsampled to 25 million fragments followed by Model-based Analysis of ChIP-Seq version 2, MACS2, signal generation using signal per million reads, SPMR. The [Allen Human Multiome Brain Atlas release](https://brain-map.org/consortia/hmba/hmba-release-basal-ganglia) describes SnapATAC2 processing and supplies group-level BigWigs alongside separately called MACS3 peaks. Inspection of the released files shows a second important implementation difference, HDA BigWig values are constant on native 100 bp bins, whereas Allen values use native 10 bp bins.

This bin width explains the apparent tenfold scale difference in base-expanded values. Across eight matched 131,072 bp windows, the stored HDA and Allen means were 0.480 and 0.0499 and their root mean squares were 5.58 and 0.556. Dividing each value by its native bin width gives means of 0.00480 and 0.00499 per bp and root mean squares of 0.0558 and 0.0556 per bp. Their first two moments are therefore already closely matched when expressed in common units. Differences remain in sparsity, upper tails, cell populations, anatomical region, and developmental stage.

An Allen ATAC-only baseline matches the earlier LoRA setup at 131 kb, batch size 8, learning rate \(10^{-3}\), LoRA without LoCon, float32 frozen and adapter parameters, and bfloat16 activations and compute. Model selection uses signed double-centered Pearson correlation, denoted differential Pearson correlation in the training logs. The raw-target baseline reached its best validation correlation at epoch 7, validation \(R=0.1894\) and test \(R=0.2100\).

Target-transform experiments apply transforms inside the JAX loss and retain raw Allen targets in each batch. Predictions are mapped back to raw Allen units before metrics are accumulated. Spatial averaging is not invertible, so its scale factor is removed but the prediction remains locally smoothed and is compared directly with unsmoothed raw targets. Every one-epoch screen used all 20,952 training windows and the same initialization and optimization settings.

| Optimization target | Validation \(R\) | Test \(R\) |
|---|---:|---:|
| Raw Allen baseline | 0.1787 | 0.1986 |
| Scalar 4x | 0.1699 | 0.1904 |
| Scalar 10x | 0.1609 | 0.1803 |
| Scalar 20x | 0.1534 | 0.1722 |
| Per-track mean match | 0.1559 | 0.1749 |
| Per-track RMS match | 0.1539 | 0.1721 |
| 50% quantile blend toward HDA | 0.1309 | 0.1418 |
| Local 100 bp mean | 0.1869 | 0.2046 |
| Local 100 bp mean, then 10x scale | 0.1728 | 0.1937 |

Increasing target magnitude hurts monotonically, and matching each track's mean or RMS is no better. Quantile normalization is substantially worse because it forces Allen's positive-value tail toward a reference pooled across biologically unmatched HDA tracks. Local 100 bp averaging is the only helpful transform, while multiplying that average by ten removes its advantage. The smoothing result indicates that matching HDA's spatial granularity reduces high-frequency target noise; it does not support a missing normalization factor.

With signed-correlation model selection and patience 3, the 100 bp-smoothed run reached its best validation result at epoch 5, validation \(R=0.1948\) and test \(R=0.2120\). This is a modest gain over the raw baseline, 0.0054 validation and 0.0020 test. It does not explain the much larger gap from the earlier HDA result near \(R=0.8\). The remaining gap is more consistent with differences in cohort, target composition, genomic signal structure, and track count than with scalar preprocessing.

A 128 bp-only output head was also exercised but not retained. It requires a coarse pooled-target metric that is not directly comparable with base-resolution fidelity, and its training path was input-bound and severalfold slower than the base-resolution setup. The run was stopped after confirming finite optimization rather than spending further compute on a dominated strategy.

### Fragment-level reconstruction

The managed Allen collection already contains the full human ATAC fragment export, 204 tabix-indexed five-column fragment files with per-cell whole-genome fragment-count summaries. The accompanying comprehensive H5AD contains 1,034,819 observation identifiers and their 60 `Group` assignments. Fragment barcodes join directly to the H5AD observation index, so reconstructing pseudobulks does not require downloading the source H5AD matrices.

The reconstruction streams each chromosome independently and accumulates native 100 bp targets. Signal per million reads, SPMR, denotes signal normalized by the library size in millions; these paired-end ATAC tracks use one million fragments as the denominator. Shifted-insertion SPMR places the two Tn5 cut sites at fragment start +4 bp and fragment end -5 bp and divides bin counts by the group's whole-genome fragments in millions. Coverage SPMR accumulates exact fragment-covered bases, including partial bins, divides by effective bin width, and then divides by whole-genome fragments in millions.

The full chr8 pilot processed 1,831,505,568 records with no unmatched cell identifiers. Across eight 131 kb windows and all 60 groups, the released tracks had mean 0.0499, root mean square 0.2156, and zero fraction 0.2563. Shifted-insertion SPMR had mean 0.0533, root mean square 0.2013, zero fraction 0.2576, ordinary Pearson (R=0.9619), and double-centered (R=0.9610). Coverage SPMR had mean 0.0495, root mean square 0.1522, zero fraction 0.2041, ordinary Pearson (R=0.6997), and double-centered (R=0.6465). The shifted-insertion definition closely recovers the released signal, while coverage represents a smoother alternative target rather than the release pipeline.

The Human Developmental Atlas, HDA, methods used Model-based Analysis of ChIP-Seq version 2, MACS2, with paired-fragment input, pileup output, and signal-per-million-reads normalization. MACS2 paired-end mode piles up each observed fragment and ignores the read-shift and fixed-extension arguments. The reported 25-million-fragment downsampling balanced pseudobulk replicates for peak calling, but it is not needed to define a library-normalized training signal. The all-data analogue retains every Allen fragment and coordinate multiplicity, computes exact paired-fragment coverage, divides by the complete group fragment count in millions, and averages within 100 bp bins. It has the same expectation as random downsampling followed by normalization and lower sampling variance.

On the chr8 pilot windows, the HDA tracks had mean 0.4796, root mean square 5.5794, zero fraction 0.3258, and 99th percentile 7.5544. Allen full-depth fragment pileup had mean 0.0495, root mean square 0.1522, zero fraction 0.2041, and 99th percentile 0.4937. The target definitions are now conceptually aligned, but their marginal distributions remain different. Earlier scalar and quantile transformations reduced fine-tuning performance, so the full-depth pileup is evaluated without forced marginal matching. Fine-tuning performance, rather than scale alone, determines whether this reprocessing addresses the Allen gap.

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

A one-step GPU smoke test completed with finite joint loss and validation RNA differential Pearson correlation 0.158 over four windows. A larger calibration used 512 training windows and 128 validation and test windows. Validation RNA global R2 increased from 0.126 after epoch 1 to 0.206 after epoch 4, while RNA R2 over loci increased from 0.166 to 0.259. Test RNA global R2 reached 0.199 and R2 over loci reached 0.341. Validation ATAC global R2 increased from 0.0047 to 0.0156 and ATAC R2 over loci increased from 0.047 to 0.204. These results establish that both assays learn with the revised target representation. RNA R2 over cell types remains strongly negative because many individual genes have little variance among the 60 groups; global R2, per-track R2 over genes, and double-centered correlation are more stable summaries.

## Joint primate training

The aligned pseudobulk matrices contain 53 cell groups and 13,509 one-to-one ortholog rows shared among human, macaque, and marmoset. Five aligned groups lack marmoset ATAC coverage, so paired-modality joint training uses the 48-group intersection supported by both assays in all species. The shared human Ensembl index is mapped through the provided ortholog table. Human and macaque use stable Ensembl identifiers in their GTFs. The available marmoset NCBI GTF is matched by gene symbol because it does not contain the ortholog table's Ensembl identifiers. Annotation-only checks matched 13,495 macaque genes, 99.90%, and all 13,506 uniquely mapped marmoset genes.

Each species uses its own reference genome, exon geometry, ATAC BigWigs, RNA pseudo-coverage BigWigs, and gene-expression artifact. Macaque Ensembl chromosome labels are translated to the corresponding NCBI accessions used by its FASTA and ATAC BigWigs. Species batches are emitted round-robin and training stops at the shortest species iterator each epoch, yielding equal numbers of batches per species. The output heads and LoRA plus LoCon adapters are shared. All three primates use AlphaGenome's human organism embedding because the pretrained model does not define macaque or marmoset organism indices. Consequently, the RNA head has one scaling vector; each channel's nonzero mean is pooled across species rather than inherited from human alone. Validation and test chromosome holdouts are species-specific homologous chromosome numbers.

## Metrics

Predictions and targets have shape ([B,S,C]), where (B) is the number of windows, (S) is the number of genomic positions or 128 bp bins, and (C) is the number of tracks. Differential Pearson correlation first reshapes the first two axes into observations, then subtracts the mean of each track and each observation from both matrices while adding back the grand mean. `double_centered_r2` is the square of this correlation. It measures agreement in cell-type-specific spatial variation after removing locus-wide and track-wide offsets, and it is logged for train, validation, and test splits after every epoch.

## Findings

The v0 one-step 131 kb LoRA+LoCon smoke run completed with finite joint loss 3.9817 and logged `double_centered_r2` for both heads. Best-checkpoint selection uses signed differential Pearson correlation rather than its square so a strong anticorrelation cannot be selected as an improvement.
