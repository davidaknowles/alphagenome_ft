# V0Data AlphaGenome tuning summary

## Goal

Train one AlphaGenome model for the non-ENCODE single-cell pseudobulk resources: Mannens human developmental atlas (HDA), Liu Human Development Multiomic Atlas (HDMA), Johansen/Allen, Zemke 2023, and Zemke 2024. The panel spans human, macaque, marmoset, and mouse, with ATAC and RNA endpoints when each source provides them.

The primary evaluation statistic is signed double-centered Pearson correlation, denoted $R$. For target and prediction matrices $Y, \hat{Y}\in\mathbb{R}^{N\times C}$, where $N$ is the number of genomic elements and $C$ is the number of cell-group tracks, the statistic removes row and column means before calculating correlation. It measures recovery of relative genomic and cell-group structure, rather than absolute assay scale. Chromosome-held-out validation and test sets are used throughout.

## Data representation

Each native source retains its reference assembly, DNA sequence, assay definition, target mask, and output scaling. The current joint endpoint models RNA for HDA, Liu, Johansen, Zemke 2023, and Zemke 2024 as direct raw-count counts-per-million pseudobulks over exon-defined genes. Released Zemke coordinate tracks remain an independent benchmark, but direct-gene targets provide the comparable cross-study training endpoint. Zemke 2024 masks four unreleased subtype channels and retains the 18 broad groups supported by its released cell assignments.

Published ATAC targets are used when appropriate. Liu and every Johansen/Allen species require reconstruction from raw ATAC fragments, using full-depth paired-fragment coverage normalized as signal per million reads. The reconstruction preserves the available data rather than downsampling it.

## Model and training strategy

The pretrained AlphaGenome backbone is shared across all sources. Output heads are dataset-specific, and multispecies studies can use source-specific heads when their target distributions differ materially. Macaque and marmoset use their native sequence and assembly but the pretrained human organism row; the model has no separately pretrained primate organism rows.

Adaptation uses low-rank linear adapters (LoRA) together with low-rank convolution adapters (LoCon). New heads are first trained against a frozen backbone, then matched adapter strategies start from the same head state. Native sources are sampled evenly so large studies do not dominate optimization. The principal current comparison combines direct-gene RNA, source-specific heads, semantic head initialization, and broad LoCon coverage.

## Findings

ATAC is consistently easier than RNA. HDA ATAC is near the target level of $R\approx0.8$, while RNA remains the limiting modality across the panel. Broader LoCon coverage improved an earlier published-target control, but it did not generalize to the completed shared-head direct-Zemke comparison, where LoRA has higher validation $R$ and the test difference is negligible. Greater adapter rank and uniform RNA upweighting have not produced a comparable gain.

Direct-gene RNA improves Zemke 2023 relative to the released coordinate-RPKM representation. This supports a target-representation mismatch, rather than read depth alone, as an important limitation for RNA. Source-specific output heads and semantic initialization are the current highest-value refinements under evaluation.

The canonical Zemke 2023 marmoset RNA test metric is dominated by a ribosomal-repeat locus within the annotated ANO6 span. This is a target artifact in the released coordinate representation, not broad model failure or an assembly-routing error. Direct-gene RNA avoids this failure mode while preserving an independent coordinate-track benchmark.

## Remaining work

Complete the matched source-specific direct-gene LoRA and LoRA plus LoCon comparison, then assess performance by source and modality. The fragment-derived Johansen ATAC targets are also undergoing split-half reproducibility assessment, which will distinguish a target ceiling from a model-fitting limitation before further target changes are considered. The main scientific objective remains robust RNA performance across studies, especially Johansen marmoset and the Zemke endpoints. Detailed measurements, target audits, training progression, and operational history are maintained in `LABNOTEBOOK.md`.
