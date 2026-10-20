# AlphaGenome v0data fine-tune (LoRA + LoCon, raw Zemke 2024 ATAC)

This directory holds an AlphaGenome model fine-tuned jointly on five single-cell pseudobulk studies (ten native sources across human, macaque, marmoset and mouse). It contains only what is needed for inference: the trained output heads, the backbone adapters, the output track layout, and the exon annotations used for gene-level RNA training. The pretrained AlphaGenome weights are not included and must be obtained separately (see Requirements).

## Model

- **Backbone.** Pretrained AlphaGenome JAX `all_folds`, shared across all sources, with low-rank adapters (LoRA, rank 16, alpha 16) on the attention and pair projections and low-rank convolution adapters (LoCon, rank 4, alpha 1) on downsampling blocks 2–5.
- **Heads.** 20 heads, one ATAC and one RNA head per native source. Multispecies studies (Johansen 2025, Zemke 2023) have a separate head per species.
- **Training.** Frozen-backbone head warmup for 12 epochs, then joint training of heads and adapters. Native sources were sampled equally. Chromosomes 8 and 9 (or their equivalents) were held out for validation and test. The checkpoint is epoch 21, selected by mean validation correlation.
- **Targets.** ATAC targets are pseudobulk accessibility coverage. Liu, Johansen and Zemke 2024 ATAC were rebuilt from full-depth fragments as coverage per million fragments; HDA and Zemke 2023 use the released tracks. RNA targets are pseudobulk counts per million (CPM) per gene, built from raw counts and supervised only at the gene level (see RNA outputs).
- **Training window.** 131,072 bp.

Held-out performance is signed double-centered Pearson correlation $R$ between predicted and observed element-by-cell-group matrices, after removing row and column means. It measures recovery of relative structure across genomic elements and cell groups, not absolute scale. Mean test $R$ is 0.684 for ATAC and 0.570 for RNA. Per-source values are in `evaluation.md`, whose `lora+locon` rows correspond to this checkpoint. These held-out values are deprecated: the base `all_folds` checkpoint was pretrained on chromosomes 8 and 9, so they overstate accuracy on unseen sequence. A model fine-tuned from a checkpoint that never saw its test regions is in progress.

## Contents

| Path | Contents |
|---|---|
| `config.json` | checkpoint configuration (heads, adapter settings) |
| `checkpoint/` | Orbax parameters for the heads and adapters (about 49 MB) |
| `heads.json` | head definitions: track labels, strands, output scaling, organism, reference assembly, provenance |
| `genes/<head>.npz` | exon geometry and observed pseudobulk CPM for each RNA head |
| `evaluation.md` | held-out correlations per native source |
| `build-constraints.txt` | build pins for the install command below |
| `SHA256SUMS` | checksums; verify with `sha256sum -c SHA256SUMS` |

## Requirements

- A CUDA GPU. The bundle was tested with JAX 0.9.2 on an NVIDIA L40S (48 GB).
- `alphagenome-ft` at tag `v0data-inference-v1`, which adds the bundle loader used below, with the tested AlphaGenome commits. Install with [uv](https://docs.astral.sh/uv/) using `build-constraints.txt` from this directory:
  ```bash
  uv venv --python 3.12 ag_v0 && source ag_v0/bin/activate
  uv pip install --build-constraint build-constraints.txt \
    "alphagenome-ft @ git+https://github.com/davidaknowles/alphagenome_ft.git@v0data-inference-v1" \
    "alphagenome @ git+https://github.com/google-deepmind/alphagenome.git@8e0a9f5f6ab2cf3222b1a11c1bd524a0f7ea344e" \
    "alphagenome_research @ git+https://github.com/google-deepmind/alphagenome_research.git@232fc695d1eab27bac9e94bcd4b50499139ba4e1" \
    "jax[cuda12]==0.9.2"
  ```
  The pins are required. `alphagenome_research` 0.3.0 and later removed a module the loader uses, and `alphagenome_research` generates protobuf code at install time that fails to import if it is built with protobuf 7. These commands were tested in a new environment.
- The pretrained AlphaGenome JAX weights (`google/alphagenome/jax/all_folds`) from Kaggle, downloaded under your own acceptance of the AlphaGenome model terms. Pass the downloaded directory as `base_checkpoint_path`; if you omit it, the loader downloads through `kagglehub`.

## Usage

```python
import numpy as np
from alphagenome.data import genome
from alphagenome_research.io import fasta
from alphagenome_research.model import one_hot_encoder
from alphagenome_ft.inference_bundle import (
    gene_expression, head_organism_index, load_gene_annotation,
    load_inference_bundle, predict_heads, read_bundle_heads,
)

BUNDLE = "path/to/this/directory"
model = load_inference_bundle(
    BUNDLE,
    base_checkpoint_path="path/to/alphagenome/jax/all_folds/1",
    init_seq_len=131072,
)
heads = {head["id"]: head for head in read_bundle_heads(BUNDLE)}

window = genome.Interval("chr19", 44_840_000, 44_840_000 + 131072)
sequence = fasta.FastaExtractor("hg38.fa").extract(window).upper()
x = one_hot_encoder.DNAOneHotEncoder(dtype=np.float32).encode(sequence)[None]

head = heads["zemke2023_rna_human"]
outputs = predict_heads(model, x, head_organism_index(head))
coverage = outputs["zemke2023_rna_human"]["predictions_128bp"][0]  # (1024 bins, 20 tracks)
labels = [track["label"] for track in head["targets"]]

annotation = load_gene_annotation(BUNDLE, "zemke2023_rna_human")
gene_ids, cpm = gene_expression(coverage, annotation, window)  # (genes, 20 groups)
```

`predict_heads` runs all 20 heads in one forward pass and returns, for each head, a dictionary of arrays with shape (batch, positions, tracks). ATAC heads return `predictions_1bp` and `predictions_128bp`; RNA heads return `predictions_128bp`. The prediction is from the forward strand only; average it with the reverse-complement prediction if you need strand symmetry.

GPU kernel autotuning can change predictions between processes by up to about 0.5% of the maximum value. The model also runs on CPU, but CPU predictions differ more from GPU ones (correlation above 0.998, individual values up to about 13% of an array's maximum). For bit-identical results, set `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"` before importing JAX; the bundle was checked to reproduce the original checkpoint exactly under these flags.

For a variant effect, predict the reference and alternate sequences in the same window and take the difference of the quantity of interest, such as gene-level CPM or ATAC signal near the variant.

## Heads

The organism index selects AlphaGenome's species-specific output embedding: 0 is human and 1 is mouse. Macaque and marmoset heads use the human index with their own genome sequence, as in training. Each head must be given sequence from the assembly it was trained on.

| Head | Assay | Study | Species | Assembly | Organism index | Tracks | Strand layout | Resolutions (bp) |
|---|---|---|---|---|---:|---:|---|---|
| `hda_atac` | ATAC | Mannens first-trimester brain (HDA) | human | hg38 | 0 | 134 | unstranded | 1, 128 |
| `hda_rna` | RNA | Mannens first-trimester brain (HDA) | human | hg38 | 0 | 134 | unstranded | 128 |
| `liu_atac` | ATAC | Liu HDMA fetal multi-organ | human | hg38 | 0 | 186 | unstranded | 1, 128 |
| `liu_rna` | RNA | Liu HDMA fetal multi-organ | human | hg38 | 0 | 372 | +/- interleaved | 128 |
| `allen_atac_human` | ATAC | Johansen 2025 basal ganglia | human | hg38 | 0 | 47 | unstranded | 1, 128 |
| `allen_rna_human` | RNA | Johansen 2025 basal ganglia | human | hg38 | 0 | 94 | +/- interleaved | 128 |
| `allen_atac_macaque` | ATAC | Johansen 2025 basal ganglia | macaque | Mmul_10 (RefSeq names) | 0 | 47 | unstranded | 1, 128 |
| `allen_rna_macaque` | RNA | Johansen 2025 basal ganglia | macaque | Mmul_10 (RefSeq names) | 0 | 94 | +/- interleaved | 128 |
| `allen_atac_marmoset` | ATAC | Johansen 2025 basal ganglia | marmoset | mCalJa1.2.pat.X | 0 | 47 | unstranded | 1, 128 |
| `allen_rna_marmoset` | RNA | Johansen 2025 basal ganglia | marmoset | mCalJa1.2.pat.X | 0 | 94 | +/- interleaved | 128 |
| `zemke2023_atac_human` | ATAC | Zemke 2023 motor cortex | human | hg38 | 0 | 20 | unstranded | 1, 128 |
| `zemke2023_rna_human` | RNA | Zemke 2023 motor cortex | human | hg38 | 0 | 20 | unstranded | 128 |
| `zemke2023_atac_macaque` | ATAC | Zemke 2023 motor cortex | macaque | rheMac10 | 0 | 20 | unstranded | 1, 128 |
| `zemke2023_rna_macaque` | RNA | Zemke 2023 motor cortex | macaque | rheMac10 | 0 | 20 | unstranded | 128 |
| `zemke2023_atac_marmoset` | ATAC | Zemke 2023 motor cortex | marmoset | calJac4 | 0 | 20 | unstranded | 1, 128 |
| `zemke2023_rna_marmoset` | RNA | Zemke 2023 motor cortex | marmoset | calJac4 | 0 | 20 | unstranded | 128 |
| `zemke2023_atac_mouse` | ATAC | Zemke 2023 motor cortex | mouse | mm10 | 1 | 20 | unstranded | 1, 128 |
| `zemke2023_rna_mouse` | RNA | Zemke 2023 motor cortex | mouse | mm10 | 1 | 20 | unstranded | 128 |
| `zemke2024_all_atac` | ATAC | Zemke 2024 hippocampal aging | human | hg38 | 0 | 18 | unstranded | 1, 128 |
| `zemke2024_all_rna` | RNA | Zemke 2024 hippocampal aging | human | hg38 | 0 | 22 | unstranded | 128 |

Track order is the order of `targets` in `heads.json`; always look labels up there rather than hardcoding positions. In stranded RNA heads, each cell group has a `+` channel followed by a `-` channel.

## RNA outputs

RNA heads were trained only against gene-level pseudobulk CPM; the per-bin coverage loss had weight zero. The 128 bp output is therefore not a calibrated coverage track. Its training-time readout is a gene-level value $\hat{y}_{g,c}=\sum_{b} w_{g,b}\,\hat{x}_{b,c}$, where $\hat{x}_{b,c}$ is the prediction for 128 bp bin $b$ and track $c$, and $w_{g,b}\in[0,1]$ is the fraction of bin $b$ covered by exons of gene $g$. For stranded heads, the channel on the gene's strand is used. `gene_expression` computes this readout, so its output is on the scale of the training CPM (the loss compared $\log(1+\hat{y})$ with $\log(1+y)$). Only genes whose full annotated span lies inside the window are returned, which matches training.

Some RNA groups had no training signal and their outputs should not be used:

- Zemke 2023 (all four species): `L5_IT`.
- Zemke 2024: `Astro1_all`, `Astro2_all`, `Micro1_all` and `Micro2_all`.

The `group_valid` array in each `genes/<head>.npz` records this. The same files contain the observed training CPM (`cpm`, groups × genes) for comparison with predictions.

## ATAC outputs

ATAC predictions are in the units of each study's training target, after the per-track scaling stored in `heads.json` (`nonzero_mean`). Units differ between studies, so compare values within a head rather than across heads.

## Caveats

- Performance was evaluated on held-out chromosomes from the training cell groups, not on new samples or new cell types.
- Held-out RNA correlation is lower than ATAC for every study, and lowest for Johansen marmoset.
- On fine-mapped single-nucleus brain eQTLs, variant effects from `zemke2023_rna_human` discriminate high- from low-PIP variants weakly: AUROC 0.52 and 0.56 at the TSS bin and 0.55 and 0.64 over the gene span on two benchmarks. Matching the eQTL cell type to its track adds a small, significant gain, mostly for microglia, astrocytes and oligodendrocytes; effects are largely shared across cell types.
