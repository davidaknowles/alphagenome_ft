# Allen ATAC reprocessing

This workflow rebuilds cell-group assay for transposase-accessible chromatin, ATAC, targets from paired-fragment records rather than transforming released BigWigs. It expects one tabix-indexed five-column fragment file per library, a per-cell whole-genome fragment-count summary beside each file, and an AnnData Hierarchical Data Format 5, H5AD, observation table containing the cell identifier and `Group` columns.

`aggregate.py` joins fragment cell identifiers to groups, streams one chromosome from every selected library, and writes a compressed chromosome shard. For (G) cell groups, chromosome length (L) bp, and bin width (W) bp, each target array has dimension (G \times \lceil L/W \rceil).

Two target definitions are retained for comparison. Signal per million reads, SPMR, denotes signal normalized by the library size in millions. For these paired-end ATAC tracks, the normalization denominator is one million fragments rather than individual sequencing reads. `insertion_spmr` counts both fragment ends after the standard Tn5 shifts of +4 bp at the left end and -5 bp at the right end. For group (g), bin (b), insertion count (I_{gb}), and whole-genome fragment count (F_g), the target is (I_{gb}/(F_g/10^6)). `coverage_spmr` accumulates the exact number (C_{gb}) of fragment-covered bases in each bin. For effective bin width (W_b), including a shorter terminal bin, the target is (C_{gb}/(W_bF_g/10^6)). The insertion target counts two shifted cut sites per fragment, while the coverage target measures mean fragment depth per base.

`compare.py` samples matching windows from released group BigWigs, averages them into the same bins, and reports distribution summaries, ordinary Pearson correlation, and double-centered Pearson correlation. Chromosome shards are evaluated before whole-genome BigWigs are generated so target definitions can be rejected without materializing unnecessary data.

## Chromosome pilot

The complete human fragment collection contains 204 tabix-indexed library files and 1,034,819 cells assigned to 60 groups. A chr8 pilot processed 1,831,505,568 fragment records with no unmatched cell identifiers. The table summarizes eight 131 kb windows across all groups. Correlations compare each reprocessed target with the released BigWigs after averaging them to 100 bp bins.

| Target | Mean | Root mean square | Zero fraction | Double-centered \(R\) |
|---|---:|---:|---:|---:|
| Released BigWigs | 0.0499 | 0.2156 | 0.2563 | 1.0000 |
| Shifted insertions, SPMR | 0.0533 | 0.2013 | 0.2576 | 0.9610 |
| Fragment coverage, SPMR | 0.0495 | 0.1522 | 0.2041 | 0.6465 |
| Human Developmental Atlas | 0.4796 | 5.5794 | 0.3258 | not applicable |

The shifted-insertion target recovers both the released distribution and cell-group-specific spatial structure. The coverage target has a similar mean but is smoother and less sparse, consistent with a different target definition rather than a normalization error. Subsequent reprocessing should use `insertion_spmr` unless the experiment explicitly tests coverage supervision.

## Full-depth HDA-style pileup

The Human Developmental Atlas, HDA, study used Model-based Analysis of ChIP-Seq version 2, MACS2, with paired-fragment input, pileup output, and signal-per-million-reads normalization. In paired-end mode, MACS2 piles up observed fragments; its read-shift and fixed-extension arguments do not alter those fragments. The study downsampled pseudobulk replicates to balance depth during peak calling. Random downsampling is unnecessary for a normalized training target and increases sampling variance.

The full-depth strategy therefore uses `coverage_spmr`, retains every valid fragment and its multiplicity, computes exact fragment coverage, and divides by the complete group library size. It averages coverage within native 100 bp bins to match HDA's spatial encoding. This is the all-data expectation of a randomly downsampled normalized pileup, with lower sampling noise. `whole_genome.sbatch` builds independent chromosome shards, `materialize.py` converts completed shards into one BigWig per group, and `finetune.sbatch` evaluates the result with the matched LoRA baseline. Low-rank adaptation, LoRA, is applied using the same settings as the earlier Allen baseline.

## Cross-species targets

The final Johansen comparison applies the same full-depth coverage definition to human, macaque, and marmoset. `slurm_multispecies_chromosomes.sbatch` reconstructs the nonhuman primary chromosomes from their species-specific fragments, cell metadata, and reference coordinates. The completed human shards are reused. `slurm_multispecies_materialize.sbatch` writes group-level BigWigs, and `prepare_multispecies_targets.py` replaces the released ATAC heads in the existing aligned 48-group configuration while preserving its ortholog-mapped RNA targets and gene-level supervision. This isolates the target-definition change from RNA preprocessing and cell-group alignment.

On the full human genome, released-track LoRA reached validation double-centered (R=0.1894). The coverage target reached (R=0.4154) by epoch 5 and was still improving. Although insertion SPMR more closely reproduces the released tracks, coverage SPMR is retained for the cross-species experiment because it is the target definition associated with the stronger held-out sequence performance and is directly comparable with the HDA pileup.
