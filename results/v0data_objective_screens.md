# Objective and preprocessing screens

These non-canonical runs test one change at a time. Paired strategies are compared at their highest common epoch. Independently selected checkpoints are reported separately and must not be used for a strategy comparison when epochs differ. Technical smoke tests and gradient diagnostics are excluded.

## Highest matched epochs

| Dataset | Strategy | Variant | Epoch | Head | Validation R | Test R |
|---|---|---|---:|---|---:|---:|
| `hda-joint` | `lora` | `geneonly_corrw0p1_screen` | 1 | `hda_atac` | 0.7710 | 0.7827 |
| `hda-joint` | `lora` | `geneonly_corrw0p1_screen` | 1 | `hda_rna` | 0.4294 | 0.6274 |
| `hda-joint` | `lora+locon` | `geneonly_corrw0p1_screen` | 1 | `hda_atac` | 0.7755 | 0.7862 |
| `hda-joint` | `lora+locon` | `geneonly_corrw0p1_screen` | 1 | `hda_rna` | 0.4282 | 0.6262 |
| `hda-joint` | `lora` | `geneonly_unstranded_balanced_screen` | 1 | `hda_atac` | 0.7692 | 0.7835 |
| `hda-joint` | `lora` | `geneonly_unstranded_balanced_screen` | 1 | `hda_rna` | 0.4788 | 0.5537 |
| `hda-joint` | `lora+locon` | `geneonly_unstranded_balanced_screen` | 1 | `hda_atac` | 0.7783 | 0.7934 |
| `hda-joint` | `lora+locon` | `geneonly_unstranded_balanced_screen` | 1 | `hda_rna` | 0.4782 | 0.5436 |
| `zemke2023-human` | `lora` | `corrw10` | 4 | `zemke2023_atac` | 0.6994 | 0.7220 |
| `zemke2023-human` | `lora` | `corrw10` | 4 | `zemke2023_rna` | 0.5704 | 0.4399 |
| `zemke2023-human` | `lora+locon` | `corrw10` | 4 | `zemke2023_atac` | 0.6998 | 0.7217 |
| `zemke2023-human` | `lora+locon` | `corrw10` | 4 | `zemke2023_rna` | 0.5298 | 0.4452 |
| `zemke2023_human` | `lora` | `published_gene_corrw10` | 1 | `zemke2023_atac` | 0.6325 | 0.6558 |
| `zemke2023_human` | `lora` | `published_gene_corrw10` | 1 | `zemke2023_rna` | 0.3533 | 0.3774 |
| `zemke2023_human` | `lora+locon` | `published_gene_corrw10` | 1 | `zemke2023_atac` | 0.5988 | 0.6245 |
| `zemke2023_human` | `lora+locon` | `published_gene_corrw10` | 1 | `zemke2023_rna` | 0.2498 | 0.1793 |
| `zemke2024-all` | `lora` | `rna_corrw10` | 1 | `zemke2024_all_atac` | 0.7073 | 0.7117 |
| `zemke2024-all` | `lora` | `rna_corrw10` | 1 | `zemke2024_all_rna` | 0.4916 | 0.2590 |
| `zemke2024-all` | `lora+locon` | `rna_corrw10` | 1 | `zemke2024_all_atac` | 0.7079 | 0.7125 |
| `zemke2024-all` | `lora+locon` | `rna_corrw10` | 1 | `zemke2024_all_rna` | 0.4800 | 0.2510 |

## Independently selected checkpoints

| Dataset | Strategy | Variant | Epoch | Head | Validation R | Test R |
|---|---|---|---:|---|---:|---:|
| `hda-joint` | `lora` | `cosine3e4_screen` | 2 | `hda_atac` | 0.7565 | 0.7688 |
| `hda-joint` | `lora` | `cosine3e4_screen` | 2 | `hda_rna` | 0.4229 | 0.5753 |
| `hda-joint` | `lora` | `geneonly_corrw0_screen` | 1 | `hda_atac` | 0.7724 | 0.7834 |
| `hda-joint` | `lora` | `geneonly_corrw0_screen` | 1 | `hda_rna` | 0.4232 | 0.5714 |
| `hda-joint` | `lora` | `geneonly_corrw0p1_screen` | 1 | `hda_atac` | 0.7710 | 0.7827 |
| `hda-joint` | `lora` | `geneonly_corrw0p1_screen` | 1 | `hda_rna` | 0.4294 | 0.6274 |
| `hda-joint` | `lora` | `geneonly_corrw10_screen` | 1 | `hda_atac` | 0.7486 | 0.7594 |
| `hda-joint` | `lora` | `geneonly_corrw10_screen` | 1 | `hda_rna` | 0.3875 | 0.6053 |
| `hda-joint` | `lora` | `geneonly_corrw1_screen` | 1 | `hda_atac` | 0.7649 | 0.7787 |
| `hda-joint` | `lora` | `geneonly_corrw1_screen` | 1 | `hda_rna` | 0.4253 | 0.6420 |
| `hda-joint` | `lora` | `geneonly_rowcorrw10_screen` | 1 | `hda_atac` | 0.7398 | 0.7512 |
| `hda-joint` | `lora` | `geneonly_rowcorrw10_screen` | 1 | `hda_rna` | 0.3709 | 0.4943 |
| `hda-joint` | `lora` | `geneonly_stranded_balanced_screen` | 1 | `hda_atac` | 0.7679 | 0.7819 |
| `hda-joint` | `lora` | `geneonly_stranded_balanced_screen` | 1 | `hda_rna` | 0.4598 | 0.4899 |
| `hda-joint` | `lora` | `geneonly_unstranded_balanced_screen` | 1 | `hda_atac` | 0.7692 | 0.7835 |
| `hda-joint` | `lora` | `geneonly_unstranded_balanced_screen` | 1 | `hda_rna` | 0.4788 | 0.5537 |
| `hda-joint` | `lora+locon` | `geneonly_corrw0p1_screen` | 1 | `hda_atac` | 0.7755 | 0.7862 |
| `hda-joint` | `lora+locon` | `geneonly_corrw0p1_screen` | 1 | `hda_rna` | 0.4282 | 0.6262 |
| `hda-joint` | `lora+locon` | `geneonly_unstranded_balanced_screen` | 1 | `hda_atac` | 0.7783 | 0.7934 |
| `hda-joint` | `lora+locon` | `geneonly_unstranded_balanced_screen` | 1 | `hda_rna` | 0.4782 | 0.5436 |
| `hda-joint` | `lora` | `rnaw5` | 1 | `hda_atac` | 0.7538 | 0.7645 |
| `hda-joint` | `lora` | `rnaw5` | 1 | `hda_rna` | 0.4310 | 0.5976 |
| `liu-hdma` | `lora` | `legacy_exon_plus_gene` | 1 | `liu_atac` | 0.7648 | 0.7786 |
| `liu-hdma` | `lora` | `legacy_exon_plus_gene` | 1 | `liu_rna` | 0.4053 | 0.4390 |
| `liu-hdma` | `lora` | `exonwindow_screen` | 1 | `liu_atac` | 0.7620 | 0.7795 |
| `liu-hdma` | `lora` | `exonwindow_screen` | 1 | `liu_rna` | 0.2614 | 0.3977 |
| `zemke2023-human` | `lora` | `corrw10` | 4 | `zemke2023_atac` | 0.6994 | 0.7220 |
| `zemke2023-human` | `lora` | `corrw10` | 4 | `zemke2023_rna` | 0.5704 | 0.4399 |
| `zemke2023-human` | `lora+locon` | `corrw10` | 3 | `zemke2023_atac` | 0.6931 | 0.7137 |
| `zemke2023-human` | `lora+locon` | `corrw10` | 3 | `zemke2023_rna` | 0.5430 | 0.4555 |
| `zemke2023-human` | `lora` | `rnaw5` | 1 | `zemke2023_atac` | 0.6637 | 0.6790 |
| `zemke2023-human` | `lora` | `rnaw5` | 1 | `zemke2023_rna` | 0.4306 | 0.3871 |
| `zemke2023_human` | `lora+locon` | `published_gene_corrw10` | 1 | `zemke2023_atac` | 0.5988 | 0.6245 |
| `zemke2023_human` | `lora+locon` | `published_gene_corrw10` | 1 | `zemke2023_rna` | 0.2498 | 0.1793 |
| `zemke2023_human` | `lora` | `published_gene_corrw10` | 1 | `zemke2023_atac` | 0.6325 | 0.6558 |
| `zemke2023_human` | `lora` | `published_gene_corrw10` | 1 | `zemke2023_rna` | 0.3533 | 0.3774 |
| `zemke2023_macaque` | `lora` | `fold_chr10_chr11` | 4 | `zemke2023_atac` | 0.5157 | 0.6356 |
| `zemke2023_macaque` | `lora` | `fold_chr10_chr11` | 4 | `zemke2023_rna` | 0.3597 | 0.4539 |
| `zemke2024-all` | `lora+locon` | `rna_corrw10` | 1 | `zemke2024_all_atac` | 0.7079 | 0.7125 |
| `zemke2024-all` | `lora+locon` | `rna_corrw10` | 1 | `zemke2024_all_rna` | 0.4800 | 0.2510 |
| `zemke2024-all` | `lora` | `rna_corrw10` | 1 | `zemke2024_all_atac` | 0.7073 | 0.7117 |
| `zemke2024-all` | `lora` | `rna_corrw10` | 1 | `zemke2024_all_rna` | 0.4916 | 0.2590 |
| `zemke2024-all` | `lora` | `rnaw5` | 1 | `zemke2024_all_atac` | 0.7149 | 0.7239 |
| `zemke2024-all` | `lora` | `rnaw5` | 1 | `zemke2024_all_rna` | 0.4012 | 0.2481 |
