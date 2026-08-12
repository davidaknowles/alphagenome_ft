# Objective and preprocessing screens

These non-canonical runs test one change at a time. Paired strategies are compared at their highest common epoch. Independently selected checkpoints are reported separately and must not be used for a strategy comparison when epochs differ. Technical smoke tests and gradient diagnostics are excluded.

## Highest matched epochs

| Dataset | Strategy | Variant | Epoch | Head | Validation R | Test R |
|---|---|---|---:|---|---:|---:|
| `zemke2023-human` | `lora` | `corrw10` | 3 | `zemke2023_atac` | 0.6942 | 0.7174 |
| `zemke2023-human` | `lora` | `corrw10` | 3 | `zemke2023_rna` | 0.5414 | 0.4311 |
| `zemke2023-human` | `lora+locon` | `corrw10` | 3 | `zemke2023_atac` | 0.6931 | 0.7137 |
| `zemke2023-human` | `lora+locon` | `corrw10` | 3 | `zemke2023_rna` | 0.5430 | 0.4555 |

## Independently selected checkpoints

| Dataset | Strategy | Variant | Epoch | Head | Validation R | Test R |
|---|---|---|---:|---|---:|---:|
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
| `hda-joint` | `lora` | `rnaw5` | 1 | `hda_atac` | 0.7538 | 0.7645 |
| `hda-joint` | `lora` | `rnaw5` | 1 | `hda_rna` | 0.4310 | 0.5976 |
| `liu-hdma` | `lora` | `legacy_exon_plus_gene` | 1 | `liu_atac` | 0.7648 | 0.7786 |
| `liu-hdma` | `lora` | `legacy_exon_plus_gene` | 1 | `liu_rna` | 0.4053 | 0.4390 |
| `zemke2023-human` | `lora` | `corrw10` | 4 | `zemke2023_atac` | 0.6994 | 0.7220 |
| `zemke2023-human` | `lora` | `corrw10` | 4 | `zemke2023_rna` | 0.5704 | 0.4399 |
| `zemke2023-human` | `lora+locon` | `corrw10` | 3 | `zemke2023_atac` | 0.6931 | 0.7137 |
| `zemke2023-human` | `lora+locon` | `corrw10` | 3 | `zemke2023_rna` | 0.5430 | 0.4555 |
| `zemke2023-human` | `lora` | `rnaw5` | 1 | `zemke2023_atac` | 0.6637 | 0.6790 |
| `zemke2023-human` | `lora` | `rnaw5` | 1 | `zemke2023_rna` | 0.4306 | 0.3871 |
| `zemke2023_macaque` | `lora` | `fold_chr10_chr11` | 4 | `zemke2023_atac` | 0.5157 | 0.6356 |
| `zemke2023_macaque` | `lora` | `fold_chr10_chr11` | 4 | `zemke2023_rna` | 0.3597 | 0.4539 |
| `zemke2024-all` | `lora` | `rnaw5` | 1 | `zemke2024_all_atac` | 0.7149 | 0.7239 |
| `zemke2024-all` | `lora` | `rnaw5` | 1 | `zemke2024_all_rna` | 0.4012 | 0.2481 |
