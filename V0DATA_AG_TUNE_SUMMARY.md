# V0Data AlphaGenome tuning summary

## Scope and metric

The goal is one AlphaGenome model for non-ENCODE single-cell pseudobulk data from the Mannens human developmental atlas (HDA), the Liu Human Development Multiomic Atlas (HDMA), Johansen/Allen, Zemke 2023, and Zemke 2024. The panel covers human, macaque, marmoset, and mouse and includes Assay for Transposase-Accessible Chromatin (ATAC) and RNA targets where available.

The primary metric is signed double-centered Pearson correlation, denoted $R$. For an evaluation matrix $Y\in\mathbb{R}^{N\times C}$, where $N$ is the number of evaluated genomic elements and $C$ is the number of cell-group tracks, double centering removes the row mean and column mean and restores the grand mean before correlating flattened predictions and targets. Validation and test sets use held-out chromosomes. This metric measures whether the model recovers relative genomic and cell-group structure rather than absolute assay scale.

## Data strategy

- ATAC targets were reprocessed from fragments where necessary to produce normalized pseudobulk coverage with consistent track semantics.
- HDA, Liu, and Johansen RNA use raw-count counts-per-million (CPM) pseudobulks assigned to exon-defined gene windows. Zemke RNA uses the published coordinate-resolved reads-per-kilobase-per-million (RPKM) tracks because equivalent raw gene-count inputs are not available.
- Each source retains its native genome assembly, species, target mask, and assay definitions. Separate source-specific ATAC and RNA heads prevent incompatible channel definitions and scales from being forced into one output projection.
- The shared AlphaGenome backbone is adapted with low-rank adaptation (LoRA) on linear operations and low-rank convolution adaptation (LoCon) on convolutional operations. Each dataset receives the same optimizer-update budget, and species within multi-species studies rotate evenly.
- Models are selected by the mean validation $R$ across ATAC and RNA heads with early stopping. Metric-aligned training adds direct correlation objectives where screens showed a benefit, rather than increasing every RNA loss uniformly.

## Strategy findings

Results below average ten native dataset/species sources and give each source equal weight.

| Strategy | Validation R | Test R | ATAC validation/test | RNA validation/test |
|---|---:|---:|---:|---:|
| LoRA | 0.5054 | 0.5246 | 0.6177 / 0.6498 | 0.3932 / 0.3993 |
| LoRA + LoCon | 0.5310 | 0.5422 | 0.6451 / 0.6685 | 0.4169 / 0.4158 |
| LoRA + LoCon, uniform RNA weight 2 | 0.5272 | 0.5422 | 0.6398 / 0.6670 | 0.4147 / 0.4174 |
| LoRA + LoCon, metric-aligned | 0.5573 | 0.5608 | 0.6539 / 0.6718 | 0.4608 / 0.4498 |
| LoRA + LoCon, tempered consolidation | **0.5576** | **0.5619** | 0.6531 / **0.6733** | **0.4621** / 0.4506 |

LoCon gives a useful improvement over LoRA alone. Uniformly doubling RNA loss weight does not help. Metric-aligned objectives provide the largest gain, especially for RNA, without reducing aggregate ATAC performance. Lower-rate consolidation changes the result only marginally, so the metric-aligned and tempered checkpoints should be treated as effectively tied rather than as evidence for a further material gain.

## Dataset difficulty

The table reports the nominal tempered checkpoint. Values are validation/test $R$.

| Dataset and species | ATAC | RNA | Assessment |
|---|---:|---:|---|
| HDA/Mannens, human | **0.807 / 0.819** | 0.453 / **0.683** | ATAC is the strongest endpoint. RNA is technically reliable but split-dependent. |
| Liu HDMA, human | 0.765 / 0.786 | **0.682** / 0.475 | ATAC is strong. RNA validation is strong, but test transfer is weaker. |
| Johansen/Allen, human | 0.596 / 0.632 | 0.592 / 0.559 | Intermediate for both assays. |
| Johansen/Allen, macaque | 0.606 / 0.448 | 0.408 / 0.523 | Cross-chromosome performance is variable. |
| Johansen/Allen, marmoset | 0.543 / 0.581 | 0.250 / 0.431 | RNA is among the most difficult endpoints. |
| Zemke 2023, human | 0.704 / 0.730 | 0.508 / 0.483 | ATAC is strong; coordinate-resolved RNA remains moderate. |
| Zemke 2023, macaque | 0.389 / 0.645 | 0.345 / 0.444 | Large validation/test asymmetry indicates chromosome or assembly heterogeneity. |
| Zemke 2023, marmoset | 0.641 / 0.603 | 0.394 / **0.157** | RNA test performance is the weakest endpoint. |
| Zemke 2023, mouse | 0.732 / 0.727 | 0.347 / 0.360 | ATAC is strong and stable; RNA is difficult. |
| Zemke 2024, human | 0.747 / 0.762 | 0.644 / 0.389 | Metric alignment helps validation RNA, but the gain does not transfer fully to test. |

ATAC is consistently easier than RNA. HDA ATAC reaches the target range near $R=0.8$, and Liu and Zemke ATAC are generally strong. Human Johansen is intermediate. The hardest targets are Johansen marmoset RNA and Zemke 2023 RNA, particularly marmoset test RNA. Several studies show large validation/test differences, which makes chromosome-specific target composition and genome/assembly effects important alongside model capacity.

## Interpretation and next step

Target audits argue against raw measurement noise being the sole RNA limitation. Estimated full-depth RNA reliability is about 1.00 for HDA, 0.95 for Liu, and 0.98 to 0.99 for Johansen donor pseudobulks. Low-rank audits also show that modest target ranks can exceed $R=0.8$ for the major gene-count matrices. In contrast, the coordinate-resolved Zemke RNA tracks agree only moderately with direct gene-count representations before log transformation. The remaining gap is therefore more consistent with target-representation mismatch, held-out chromosome shifts, shared-backbone interference, and optimization or adapter/head capacity than with insufficient read depth alone.

No broadly successful RNA configuration has reached $R\approx0.8$. The current follow-up freezes the shared LoRA and LoCon adapters and refits only source-specific heads at several learning rates. Early results do not exceed the tempered source checkpoint, suggesting that head-only optimization is unlikely to remove the remaining ceiling by itself.
