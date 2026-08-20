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
| `joint_all_nonencode` | `lora+locon` | `lr1e4_reset` | 19 | `allen_atac` | 0.5853 | 0.5409 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_reset` | 19 | `allen_rna` | 0.3407 | 0.4591 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_reset` | 19 | `hda_atac` | 0.8081 | 0.8204 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_reset` | 19 | `hda_rna` | 0.4573 | 0.6556 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_reset` | 19 | `liu_atac` | 0.7687 | 0.7878 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_reset` | 19 | `liu_rna` | 0.6938 | 0.4989 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_reset` | 19 | `zemke2023_atac` | 0.4534 | 0.6756 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_reset` | 19 | `zemke2023_rna` | 0.3354 | 0.2149 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_reset` | 19 | `zemke2024_all_atac` | 0.7507 | 0.7619 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_reset` | 19 | `zemke2024_all_rna` | 0.4110 | 0.2912 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_rnaw2_reset` | 19 | `allen_atac` | 0.5842 | 0.5403 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_rnaw2_reset` | 19 | `allen_rna` | 0.3411 | 0.4601 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_rnaw2_reset` | 19 | `hda_atac` | 0.8074 | 0.8200 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_rnaw2_reset` | 19 | `hda_rna` | 0.4571 | 0.6559 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_rnaw2_reset` | 19 | `liu_atac` | 0.7682 | 0.7876 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_rnaw2_reset` | 19 | `liu_rna` | 0.6973 | 0.5033 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_rnaw2_reset` | 19 | `zemke2023_atac` | 0.4552 | 0.6750 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_rnaw2_reset` | 19 | `zemke2023_rna` | 0.3350 | 0.2152 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_rnaw2_reset` | 19 | `zemke2024_all_atac` | 0.7508 | 0.7617 |
| `joint_all_nonencode` | `lora+locon` | `lr1e4_rnaw2_reset` | 19 | `zemke2024_all_rna` | 0.4125 | 0.2945 |
| `joint_all_nonencode` | `lora+locon` | `lr3e4_reset` | 17 | `allen_atac` | 0.5804 | 0.5371 |
| `joint_all_nonencode` | `lora+locon` | `lr3e4_reset` | 17 | `allen_rna` | 0.3263 | 0.4571 |
| `joint_all_nonencode` | `lora+locon` | `lr3e4_reset` | 17 | `hda_atac` | 0.8052 | 0.8181 |
| `joint_all_nonencode` | `lora+locon` | `lr3e4_reset` | 17 | `hda_rna` | 0.4374 | 0.6580 |
| `joint_all_nonencode` | `lora+locon` | `lr3e4_reset` | 17 | `liu_atac` | 0.7672 | 0.7865 |
| `joint_all_nonencode` | `lora+locon` | `lr3e4_reset` | 17 | `liu_rna` | 0.7121 | 0.5072 |
| `joint_all_nonencode` | `lora+locon` | `lr3e4_reset` | 17 | `zemke2023_atac` | 0.4647 | 0.6725 |
| `joint_all_nonencode` | `lora+locon` | `lr3e4_reset` | 17 | `zemke2023_rna` | 0.3256 | 0.2172 |
| `joint_all_nonencode` | `lora+locon` | `lr3e4_reset` | 17 | `zemke2024_all_atac` | 0.7466 | 0.7578 |
| `joint_all_nonencode` | `lora+locon` | `lr3e4_reset` | 17 | `zemke2024_all_rna` | 0.4084 | 0.2779 |
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
