# V0Data AlphaGenome tuning summary

## Scope and metric

The goal is one AlphaGenome model for non-ENCODE single-cell pseudobulk data from the Mannens human developmental atlas (HDA), the Liu Human Development Multiomic Atlas (HDMA), Johansen/Allen, Zemke 2023, and Zemke 2024. The panel covers human, macaque, marmoset, and mouse and includes Assay for Transposase-Accessible Chromatin (ATAC) and RNA targets where available.

The primary metric is signed double-centered Pearson correlation, denoted $R$. For an evaluation matrix $Y\in\mathbb{R}^{N\times C}$, where $N$ is the number of evaluated genomic elements and $C$ is the number of cell-group tracks, double centering removes the row mean and column mean and restores the grand mean before correlating flattened predictions and targets. Validation and test sets use held-out chromosomes. This metric measures whether the model recovers relative genomic and cell-group structure rather than absolute assay scale.

## Data strategy

- Liu HDMA and all three Johansen/Allen species required ATAC reconstruction from raw fragments. The reconstruction retains all selected fragments, computes mean paired-fragment coverage in 100 bp bins, and normalizes by the complete pseudobulk library size as signal per million reads (SPMR). HDA/Mannens, Zemke 2023, and Zemke 2024 use published ATAC BigWigs and did not require fragment processing for the current model.
- HDA, Liu, and Johansen RNA use raw-count counts-per-million (CPM) pseudobulks assigned to exon-defined gene windows. The canonical Zemke benchmark uses published coordinate-resolved reads-per-kilobase-per-million (RPKM) tracks. Zemke 2023 also releases raw UMI matrices that support a separate comparable direct-gene endpoint for 19 of 20 subclasses; `L5_IT` cannot be reconstructed from the released subclass labels and is masked.
- Zemke 2024 releases donor-level raw UMI matrices and filtered-cell assignments for 18 of its 22 published broad RNA channels. A separate all-direct-gene joint endpoint uses their aggregated CPM pseudobulks, masks the four unreleased Astro1/2 and Micro1/2 subtype assignments, disables coordinate RNA loss, and derives output scales from direct CPM per expressed union-exon base. Raw CPM agrees weakly with exon-integrated published RPKM, $R=0.3292$, despite log1p agreement of 0.8666. The direct endpoint therefore measures the released raw counts and is not interpreted as a reconstruction of the published coordinate tracks.
- Each source retains its native genome assembly, species, target mask, and assay definitions. Separate dataset-specific ATAC and RNA heads prevent incompatible study definitions and scales from being forced into one output projection, while aligned species within a study share those heads.
- The shared AlphaGenome backbone is adapted with low-rank adaptation (LoRA) on linear operations and low-rank convolution adaptation (LoCon) on convolutional operations. Each dataset receives the same optimizer-update budget, and species within multi-species studies rotate evenly.
- Models are selected by the mean validation $R$ across ATAC and RNA heads with early stopping. Metric-aligned training adds direct correlation objectives where screens showed a benefit, rather than increasing every RNA loss uniformly.

| Dataset | ATAC target used | Fragment processing required |
|---|---|---|
| HDA/Mannens, human | Published Model-based Analysis of ChIP-Seq version 2 (MACS2) SPMR BigWigs | No |
| Liu HDMA, human | Reconstructed all-fragment coverage SPMR | Yes |
| Johansen/Allen, human, macaque, marmoset | Reconstructed all-fragment coverage SPMR | Yes, all three species |
| Zemke 2023, four species | Published RPKM BigWigs | No |
| Zemke 2024, human | Published pseudobulk BigWigs | No |

## Species and head routing

Heads are dataset-specific, not fully species-specific. HDA, Liu, and Zemke 2024 each have their own human ATAC and RNA heads. Johansen/Allen has one ATAC head and one RNA head shared across its three primates, whose aligned channels denote the same 47 cell groups. Zemke 2023 likewise has one ATAC head and one RNA head shared across its four species, whose channels denote the same 20 cell types.

Every native source uses its own reference genome and DNA sequence. Batches also carry AlphaGenome's pretrained organism index, but that index distinguishes only human and mouse. Human, macaque, and marmoset therefore use the human index; mouse uses the mouse index. The multi-organism Zemke head selects a human parameter row for all three primates and a mouse row for mouse. Macaque and marmoset have no separate learned species token or head row, so they are distinguished by sequence and assembly rather than explicit species conditioning.

## Strategy findings

Results below average ten native dataset/species sources and give each source equal weight. ATAC and RNA are reported separately; averaging them obscures the modality gap.

| Strategy | ATAC validation R | ATAC test R | RNA validation R | RNA test R |
|---|---:|---:|---:|---:|
| LoRA | 0.6177 | 0.6498 | 0.3932 | 0.3993 |
| LoRA + LoCon | 0.6451 | 0.6685 | 0.4169 | 0.4158 |
| LoRA + LoCon, uniform RNA weight 2 | 0.6398 | 0.6670 | 0.4147 | 0.4174 |
| LoRA + LoCon, metric-aligned | **0.6539** | 0.6718 | 0.4608 | 0.4498 |
| LoRA + LoCon, tempered consolidation | 0.6531 | **0.6733** | **0.4621** | **0.4506** |

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
| Zemke 2023, marmoset | 0.641 / 0.603 | 0.394 / 0.157; 0.629 excluding one ribosomal-repeat window | The canonical RNA test metric is dominated by one target artifact. |
| Zemke 2023, mouse | 0.732 / 0.727 | 0.347 / 0.360 | ATAC is strong and stable; RNA is difficult. |
| Zemke 2024, human | 0.747 / 0.762 | 0.644 / 0.389 | Metric alignment helps validation RNA, but the gain does not transfer fully to test. |

ATAC is consistently easier than RNA. HDA ATAC reaches the target range near $R=0.8$, and Liu and Zemke ATAC are generally strong. Human Johansen is intermediate. The hardest remaining targets include Johansen marmoset RNA and Zemke 2023 RNA. Several studies show large validation/test differences, which makes chromosome-specific target composition and genome/assembly effects important alongside model capacity.

### Zemke 2023 marmoset RNA test artifact

The canonical marmoset RNA test value of $R=0.157$ is primarily caused by one chromosome-9 locus rather than broad failure across the test chromosome. All 20 released RNA tracks reach their chromosome-wide maximum at `chr9:27738607-27738948`, with values from 23,346 to 106,835 RPKM. The [UCSC `calJac4` RepeatMasker annotation](https://api.genome.ucsc.edu/getData/track?genome=calJac4&track=rmsk&chrom=chr9&start=27737000&end=27741000) identifies this 341 bp sequence as the small-subunit ribosomal RNA repeat `SSU-rRNA_Hsa`, embedded within the annotated `ANO6` span. The released raw count matrix is concordant with the tracks: marmoset `ANO6` reaches 7,952 counts per million and has a median of 2,296 across subclasses, compared with maxima of 122 in human and 189 in macaque. The source processing assigned both intronic and exonic reads to genes, making repeat-derived intronic alignments a plausible origin of the signal ([GSE229169 processing record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE229169)).

The repeat-overlapping coarse bin accounts for 70.7% of chromosome-9 double-centered RNA target variance. A matched evaluation restored the same selected checkpoint and changed only genomic support: the baseline evaluated all 1,022 chromosome-9 windows, while the diagnostic omitted the single 131 kb window containing the repeat. RNA $R$ increased from 0.1570 to 0.6291; ATAC changed from 0.6046 to 0.6156. The assembly and track chromosome sizes agree, and every strategy had previously produced marmoset RNA test values between 0.142 and 0.170, so neither checkpoint choice nor assembly routing explains the extreme canonical value. The repeat-filtered result remains below 0.8, but it is the appropriate estimate of broader chromosome-9 performance for this checkpoint.

## Interpretation and next step

Target audits argue against raw measurement noise being the sole RNA limitation. Estimated full-depth RNA reliability is about 1.00 for HDA, 0.95 for Liu, and 0.98 to 0.99 for Johansen donor pseudobulks. Low-rank audits also show that modest target ranks can exceed $R=0.8$ for the major gene-count matrices. In contrast, the coordinate-resolved Zemke RNA tracks agree only moderately with direct gene-count representations before log transformation. The remaining gap is therefore more consistent with target-representation mismatch, held-out chromosome shifts, shared-backbone interference, and optimization or adapter/head capacity than with insufficient read depth alone.

No broadly successful RNA configuration has reached $R\approx0.8$. Post hoc refitting of only the dataset heads at learning rates $10^{-4}$, $3\times10^{-4}$, and $10^{-3}$ early-stopped after five non-improving epochs. Their final aggregate validation correlations were 0.59289, 0.59288, and 0.59183, respectively, below the unchanged source value 0.59337. The corresponding RNA means were 0.50638, 0.50565, and 0.50269, compared with 0.50746 at the source. Head-only optimization is therefore unlikely to remove the remaining ceiling by itself.

Expanding LoCon coverage from downsampling blocks 4 and 5 to blocks 2 through 5 also did not improve immediately. Its first full continuation epoch retained essentially unchanged ATAC, 0.67960 versus 0.67928, but reduced mean RNA from 0.50746 to 0.49488 and aggregate validation from 0.59337 to 0.58724. The run remains under the same early-stopping patience because prior RNA trajectories were non-monotonic. A separate source-balanced continuation tests whether equal study weighting underexposes the three Johansen and four Zemke 2023 native genomes; it changes only training order and per-source update counts while preserving every target manifest.

The next matched screen instead warms the newly initialized dataset heads before adapter training. It trains the heads for up to 20 epochs against the frozen pretrained backbone, with validation early stopping after five non-improving epochs, selects that head checkpoint, and then adds either LoRA or LoRA plus LoCon with zero initial residuals. Both adapter arms start from exactly the same warmed heads, reset the optimizer, and continue joint head-and-adapter training. This tests whether preventing random heads from driving early backbone-adapter updates improves optimization without changing the model's initial function at the handoff.

The multispecies studies currently share output projections across native species even though their target distributions differ. Johansen RNA double-centered standard deviation is 0.00123 in human and about 0.0018 in macaque and marmoset. Zemke 2023 RNA ranges from 3.81 to 7.50 in its released RPKM representation, and its ATAC double-centered standard deviation ranges from 3.21 to 8.59. An opt-in source-specific-head screen therefore assigns an independent ATAC/RNA projection to each Johansen and Zemke 2023 species while retaining the shared backbone, aligned channel order, and equal native-source exposure. This is preferable to treating macaque or marmoset as additional AlphaGenome organism classes because the pretrained trunk supports human and mouse organism rows, while the new distinction is needed at the dataset output layer.

A second staged comparison tests RNA representation rather than optimizer ordering. It replaces only Zemke 2023 coordinate RNA with raw-UMI subclass CPM gene supervision, disables coordinate RNA loss, and retains the same ATAC targets. Synthetic channel scales are total positive CPM divided by the union of expressed exon bases and pooled across human, macaque, marmoset, and mouse. This makes the reported Zemke RNA endpoint consistent with the direct-gene metrics used for HDA, Liu, and Johansen while keeping the published-coordinate result as a separate benchmark. Newly scaled heads are warmed before matched LoRA and LoRA plus LoCon branches, avoiding transfer from a coordinate-trained RNA projection.

The prepared all-direct-gene extension applies the same representation to Zemke 2024. Every RNA head in its five-dataset, ten-source manifest uses direct CPM gene supervision with no coordinate coverage loss. Zemke 2023 and Zemke 2024 use correlation weights one, while the Zemke coordinate ATAC objective retains weight three. The new Zemke 2024 output scales range from 0.00754 to 0.00933 CPM per expressed exon base, compared with the unrelated published-track RPKM scales of 1.56 to 16.60. This removes a unit mismatch that would persist if direct targets were attached without rebuilding channel metadata. Its source-specific mode combines the comparable RNA endpoint with independent Johansen and Zemke projections and equal native-source sampling, so checkpoint selection averages 20 per-source assay heads rather than pooling multispecies predictions into 10 dataset heads.

Window ordering is objective-specific. The minibatch double-centered objective uses ordinary randomized sequence windows. Gene-balanced ordering is reserved for row-centered correlation variants because sparse Johansen batches have a median of one gene: balancing reduces empty batches but increases nonempty batches with zero local double-centered variance from about 26% to 53–59%. This ordering choice changes optimization only; validation and test still evaluate every held-out chromosome window and report the same split-wide signed double-centered correlation.
