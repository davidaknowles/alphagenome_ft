# Adapter comparison results

Each run is selected by the mean validation signed double-centered Pearson correlation across its heads. Per-head test values come from that same epoch. Runs can have different maximum epochs while training is active, so strategy conclusions require a matched-epoch comparison.

| Dataset | Strategy | Epoch | Head | Validation R | Test R |
|---|---|---:|---|---:|---:|
| `hda-joint` | `lora` | 2 | `hda_atac` | 0.7892 | 0.7990 |
| `hda-joint` | `lora` | 2 | `hda_rna` | 0.5201 | 0.6182 |
| `hda-joint` | `lora+locon` | 2 | `hda_atac` | 0.7957 | 0.8049 |
| `hda-joint` | `lora+locon` | 2 | `hda_rna` | 0.5241 | 0.6222 |
| `hda` | `lora` | 4 | `hda_atac` | 0.8053 | 0.8159 |
| `hda` | `lora+locon` | 4 | `hda_atac` | 0.8086 | 0.8166 |
| `zemke2023-human` | `lora` | 4 | `zemke2023_atac` | 0.6957 | 0.7112 |
| `zemke2023-human` | `lora` | 4 | `zemke2023_rna` | 0.4809 | 0.4051 |
| `zemke2023-human` | `lora+locon` | 2 | `zemke2023_atac` | 0.6925 | 0.7093 |
| `zemke2023-human` | `lora+locon` | 2 | `zemke2023_rna` | 0.4578 | 0.3873 |
| `zemke2023_marmoset` | `lora` | 1 | `zemke2023_atac` | 0.5754 | 0.5495 |
| `zemke2023_marmoset` | `lora` | 1 | `zemke2023_rna` | 0.2771 | 0.1549 |
| `zemke2023_marmoset` | `lora+locon` | 1 | `zemke2023_atac` | 0.5799 | 0.5527 |
| `zemke2023_marmoset` | `lora+locon` | 1 | `zemke2023_rna` | 0.2800 | 0.1918 |
| `zemke2024-all` | `lora` | 3 | `zemke2024_all_atac` | 0.7439 | 0.7548 |
| `zemke2024-all` | `lora` | 3 | `zemke2024_all_rna` | 0.4713 | 0.2775 |
| `zemke2024-all` | `lora+locon` | 4 | `zemke2024_all_atac` | 0.7499 | 0.7599 |
| `zemke2024-all` | `lora+locon` | 4 | `zemke2024_all_rna` | 0.4722 | 0.2888 |
