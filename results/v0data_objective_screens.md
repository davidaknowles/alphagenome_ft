# Objective and preprocessing screens

These non-canonical runs test one change at a time. Each run is selected by mean validation signed double-centered Pearson correlation; technical smoke tests and gradient diagnostics are excluded.

| Dataset | Strategy | Variant | Epoch | Head | Validation R | Test R |
|---|---|---|---:|---|---:|---:|
| `hda-joint` | `lora` | `geneonly_corrw0_screen` | 1 | `hda_atac` | 0.7724 | 0.7834 |
| `hda-joint` | `lora` | `geneonly_corrw0_screen` | 1 | `hda_rna` | 0.4232 | 0.5714 |
| `hda-joint` | `lora` | `rnaw5` | 1 | `hda_atac` | 0.7538 | 0.7645 |
| `hda-joint` | `lora` | `rnaw5` | 1 | `hda_rna` | 0.4310 | 0.5976 |
| `liu-hdma` | `lora` | `legacy_exon_plus_gene` | 1 | `liu_atac` | 0.7648 | 0.7786 |
| `liu-hdma` | `lora` | `legacy_exon_plus_gene` | 1 | `liu_rna` | 0.4053 | 0.4390 |
| `zemke2023-human` | `lora` | `corrw10` | 1 | `zemke2023_atac` | 0.6831 | 0.7001 |
| `zemke2023-human` | `lora` | `corrw10` | 1 | `zemke2023_rna` | 0.4673 | 0.4079 |
| `zemke2023-human` | `lora` | `rnaw5` | 1 | `zemke2023_atac` | 0.6637 | 0.6790 |
| `zemke2023-human` | `lora` | `rnaw5` | 1 | `zemke2023_rna` | 0.4306 | 0.3871 |
| `zemke2023_macaque` | `lora` | `fold_chr10_chr11` | 4 | `zemke2023_atac` | 0.5157 | 0.6356 |
| `zemke2023_macaque` | `lora` | `fold_chr10_chr11` | 4 | `zemke2023_rna` | 0.3597 | 0.4539 |
| `zemke2024-all` | `lora` | `rnaw5` | 1 | `zemke2024_all_atac` | 0.7149 | 0.7239 |
| `zemke2024-all` | `lora` | `rnaw5` | 1 | `zemke2024_all_rna` | 0.4012 | 0.2481 |
