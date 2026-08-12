# Adapter comparison results

Each run is selected by the mean validation signed double-centered Pearson correlation across its heads. Per-head test values come from that same epoch. Runs can have different maximum epochs while training is active, so strategy conclusions require a matched-epoch comparison.

| Dataset | Strategy | Epoch | Head | Validation R | Test R |
|---|---|---:|---|---:|---:|
| `hda-joint` | `lora` | 3 | `hda_atac` | 0.7947 | 0.8061 |
| `hda-joint` | `lora` | 3 | `hda_rna` | 0.5170 | 0.6181 |
| `hda-joint` | `lora+locon` | 2 | `hda_atac` | 0.7957 | 0.8049 |
| `hda-joint` | `lora+locon` | 2 | `hda_rna` | 0.5241 | 0.6222 |
| `hda` | `lora` | 4 | `hda_atac` | 0.8053 | 0.8159 |
| `hda` | `lora+locon` | 4 | `hda_atac` | 0.8086 | 0.8166 |
| `johansen_joint` | `lora+locon` | 1 | `allen_atac` | 0.5107 | 0.4704 |
| `johansen_joint` | `lora+locon` | 1 | `allen_rna` | 0.2721 | 0.3621 |
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
| `zemke2023_mouse` | `lora` | 1 | `zemke2023_atac` | 0.2855 | 0.2911 |
| `zemke2023_mouse` | `lora` | 1 | `zemke2023_rna` | 0.0053 | 0.0157 |
| `zemke2023_mouse` | `lora+locon` | 1 | `zemke2023_atac` | 0.3237 | 0.3135 |
| `zemke2023_mouse` | `lora+locon` | 1 | `zemke2023_rna` | 0.0009 | -0.0014 |
| `zemke2024-all` | `lora` | 5 | `zemke2024_all_atac` | 0.7436 | 0.7534 |
| `zemke2024-all` | `lora` | 5 | `zemke2024_all_rna` | 0.4759 | 0.2925 |
| `zemke2024-all` | `lora+locon` | 4 | `zemke2024_all_atac` | 0.7499 | 0.7599 |
| `zemke2024-all` | `lora+locon` | 4 | `zemke2024_all_rna` | 0.4722 | 0.2888 |
