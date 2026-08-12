# Adapter comparison results

The first table compares strategies at the highest epoch completed by both. The second reports each run's independently selected checkpoint, which maximizes mean validation signed double-centered Pearson correlation across heads and must not be used for a strategy comparison when epochs differ.

## Highest matched epochs

| Dataset | Strategy | Epoch | Head | Validation R | Test R |
|---|---|---:|---|---:|---:|
| `hda` | `lora` | 4 | `hda_atac` | 0.8053 | 0.8159 |
| `hda` | `lora+locon` | 4 | `hda_atac` | 0.8086 | 0.8166 |
| `hda-joint` | `lora` | 3 | `hda_atac` | 0.7947 | 0.8061 |
| `hda-joint` | `lora` | 3 | `hda_rna` | 0.5170 | 0.6181 |
| `hda-joint` | `lora+locon` | 3 | `hda_atac` | 0.8035 | 0.8127 |
| `hda-joint` | `lora+locon` | 3 | `hda_rna` | 0.5190 | 0.6089 |
| `johansen-human` | `lora` | 3 | `allen_atac` | 0.5862 | 0.6181 |
| `johansen-human` | `lora` | 3 | `allen_rna` | 0.4908 | 0.4692 |
| `johansen-human` | `lora+locon` | 3 | `allen_atac` | 0.5886 | 0.6218 |
| `johansen-human` | `lora+locon` | 3 | `allen_rna` | 0.5237 | 0.4588 |
| `liu-hdma` | `lora` | 1 | `liu_atac` | 0.7613 | 0.7791 |
| `liu-hdma` | `lora` | 1 | `liu_rna` | 0.5231 | 0.4445 |
| `liu-hdma` | `lora+locon` | 1 | `liu_atac` | 0.7629 | 0.7795 |
| `liu-hdma` | `lora+locon` | 1 | `liu_rna` | 0.5159 | 0.4437 |
| `zemke2023-human` | `lora` | 4 | `zemke2023_atac` | 0.6957 | 0.7112 |
| `zemke2023-human` | `lora` | 4 | `zemke2023_rna` | 0.4809 | 0.4051 |
| `zemke2023-human` | `lora+locon` | 4 | `zemke2023_atac` | 0.6971 | 0.7151 |
| `zemke2023-human` | `lora+locon` | 4 | `zemke2023_rna` | 0.4987 | 0.4099 |
| `zemke2023_joint` | `lora` | 1 | `zemke2023_atac` | 0.4017 | 0.6478 |
| `zemke2023_joint` | `lora` | 1 | `zemke2023_rna` | 0.2950 | 0.1990 |
| `zemke2023_joint` | `lora+locon` | 1 | `zemke2023_atac` | 0.4333 | 0.6540 |
| `zemke2023_joint` | `lora+locon` | 1 | `zemke2023_rna` | 0.2835 | 0.2152 |
| `zemke2023_macaque` | `lora` | 1 | `zemke2023_atac` | 0.2000 | 0.5819 |
| `zemke2023_macaque` | `lora` | 1 | `zemke2023_rna` | 0.2401 | 0.2863 |
| `zemke2023_macaque` | `lora+locon` | 1 | `zemke2023_atac` | 0.2054 | 0.5871 |
| `zemke2023_macaque` | `lora+locon` | 1 | `zemke2023_rna` | 0.2371 | 0.2839 |
| `zemke2023_marmoset` | `lora` | 1 | `zemke2023_atac` | 0.5754 | 0.5495 |
| `zemke2023_marmoset` | `lora` | 1 | `zemke2023_rna` | 0.2771 | 0.1549 |
| `zemke2023_marmoset` | `lora+locon` | 1 | `zemke2023_atac` | 0.5799 | 0.5527 |
| `zemke2023_marmoset` | `lora+locon` | 1 | `zemke2023_rna` | 0.2800 | 0.1918 |
| `zemke2024-all` | `lora` | 5 | `zemke2024_all_atac` | 0.7436 | 0.7534 |
| `zemke2024-all` | `lora` | 5 | `zemke2024_all_rna` | 0.4759 | 0.2925 |
| `zemke2024-all` | `lora+locon` | 5 | `zemke2024_all_atac` | 0.7462 | 0.7543 |
| `zemke2024-all` | `lora+locon` | 5 | `zemke2024_all_rna` | 0.4708 | 0.2945 |

## Independently selected checkpoints

| Dataset | Strategy | Epoch | Head | Validation R | Test R |
|---|---|---:|---|---:|---:|
| `hda-joint` | `lora` | 3 | `hda_atac` | 0.7947 | 0.8061 |
| `hda-joint` | `lora` | 3 | `hda_rna` | 0.5170 | 0.6181 |
| `hda-joint` | `lora+locon` | 3 | `hda_atac` | 0.8035 | 0.8127 |
| `hda-joint` | `lora+locon` | 3 | `hda_rna` | 0.5190 | 0.6089 |
| `hda` | `lora` | 4 | `hda_atac` | 0.8053 | 0.8159 |
| `hda` | `lora+locon` | 4 | `hda_atac` | 0.8086 | 0.8166 |
| `johansen-human` | `lora+locon` | 3 | `allen_atac` | 0.5886 | 0.6218 |
| `johansen-human` | `lora+locon` | 3 | `allen_rna` | 0.5237 | 0.4588 |
| `johansen-human` | `lora` | 2 | `allen_atac` | 0.5818 | 0.6140 |
| `johansen-human` | `lora` | 2 | `allen_rna` | 0.5144 | 0.4543 |
| `johansen_joint` | `lora` | 1 | `allen_atac` | 0.5270 | 0.4840 |
| `johansen_joint` | `lora` | 1 | `allen_rna` | 0.2406 | 0.3564 |
| `liu-hdma` | `lora` | 1 | `liu_atac` | 0.7613 | 0.7791 |
| `liu-hdma` | `lora` | 1 | `liu_rna` | 0.5231 | 0.4445 |
| `liu-hdma` | `lora+locon` | 1 | `liu_atac` | 0.7629 | 0.7795 |
| `liu-hdma` | `lora+locon` | 1 | `liu_rna` | 0.5159 | 0.4437 |
| `zemke2023-human` | `lora` | 4 | `zemke2023_atac` | 0.6957 | 0.7112 |
| `zemke2023-human` | `lora` | 4 | `zemke2023_rna` | 0.4809 | 0.4051 |
| `zemke2023-human` | `lora+locon` | 4 | `zemke2023_atac` | 0.6971 | 0.7151 |
| `zemke2023-human` | `lora+locon` | 4 | `zemke2023_rna` | 0.4987 | 0.4099 |
| `zemke2023_joint` | `lora` | 1 | `zemke2023_atac` | 0.4017 | 0.6478 |
| `zemke2023_joint` | `lora` | 1 | `zemke2023_rna` | 0.2950 | 0.1990 |
| `zemke2023_joint` | `lora+locon` | 1 | `zemke2023_atac` | 0.4333 | 0.6540 |
| `zemke2023_joint` | `lora+locon` | 1 | `zemke2023_rna` | 0.2835 | 0.2152 |
| `zemke2023_macaque` | `lora` | 1 | `zemke2023_atac` | 0.2000 | 0.5819 |
| `zemke2023_macaque` | `lora` | 1 | `zemke2023_rna` | 0.2401 | 0.2863 |
| `zemke2023_macaque` | `lora+locon` | 1 | `zemke2023_atac` | 0.2054 | 0.5871 |
| `zemke2023_macaque` | `lora+locon` | 1 | `zemke2023_rna` | 0.2371 | 0.2839 |
| `zemke2023_marmoset` | `lora` | 1 | `zemke2023_atac` | 0.5754 | 0.5495 |
| `zemke2023_marmoset` | `lora` | 1 | `zemke2023_rna` | 0.2771 | 0.1549 |
| `zemke2023_marmoset` | `lora+locon` | 1 | `zemke2023_atac` | 0.5799 | 0.5527 |
| `zemke2023_marmoset` | `lora+locon` | 1 | `zemke2023_rna` | 0.2800 | 0.1918 |
| `zemke2023_mouse` | `lora` | 1 | `zemke2023_atac` | 0.6641 | 0.6642 |
| `zemke2023_mouse` | `lora` | 1 | `zemke2023_rna` | 0.2095 | 0.2310 |
| `zemke2024-all` | `lora` | 5 | `zemke2024_all_atac` | 0.7436 | 0.7534 |
| `zemke2024-all` | `lora` | 5 | `zemke2024_all_rna` | 0.4759 | 0.2925 |
| `zemke2024-all` | `lora+locon` | 4 | `zemke2024_all_atac` | 0.7499 | 0.7599 |
| `zemke2024-all` | `lora+locon` | 4 | `zemke2024_all_rna` | 0.4722 | 0.2888 |
