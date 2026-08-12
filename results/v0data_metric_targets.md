# Canonical metric-target audit

The table reports the stronger of the two strategy-selected canonical checkpoints for each modality. Each strategy checkpoint is selected by mean validation signed double-centered Pearson correlation across its heads. Missing evidence is distinct from a measured value below the target.

| Dataset | Modality | Strategy | Epoch | Validation R | Test R | Gap to 0.8 | Status |
|---|---|---|---:|---:|---:|---:|---|
| `hda` | ATAC | `lora+locon` | 4 | 0.8086 | 0.8166 | 0.0000 | target reached |
| `hda-joint` | ATAC | `lora+locon` | 2 | 0.7957 | 0.8049 | 0.0043 | below target |
| `hda-joint` | RNA | `lora+locon` | 2 | 0.5241 | 0.6222 | 0.2759 | below target |
| `johansen-human` | ATAC | `lora` | 2 | 0.5818 | 0.6140 | 0.2182 | below target |
| `johansen-human` | RNA | `lora` | 2 | 0.5144 | 0.4543 | 0.2856 | below target |
| `johansen_joint` | ATAC |  |  |  |  |  | missing evidence |
| `johansen_joint` | RNA |  |  |  |  |  | missing evidence |
| `liu-hdma` | ATAC | `lora+locon` | 1 | 0.7629 | 0.7795 | 0.0371 | below target |
| `liu-hdma` | RNA | `lora` | 1 | 0.5231 | 0.4445 | 0.2769 | below target |
| `zemke2023-human` | ATAC | `lora+locon` | 4 | 0.6971 | 0.7151 | 0.1029 | below target |
| `zemke2023-human` | RNA | `lora+locon` | 4 | 0.4987 | 0.4099 | 0.3013 | below target |
| `zemke2023_macaque` | ATAC | `lora+locon` | 1 | 0.2054 | 0.5871 | 0.5946 | below target |
| `zemke2023_macaque` | RNA | `lora` | 1 | 0.2401 | 0.2863 | 0.5599 | below target |
| `zemke2023_marmoset` | ATAC | `lora+locon` | 1 | 0.5799 | 0.5527 | 0.2201 | below target |
| `zemke2023_marmoset` | RNA | `lora+locon` | 1 | 0.2800 | 0.1918 | 0.5200 | below target |
| `zemke2023_mouse` | ATAC |  |  |  |  |  | missing evidence |
| `zemke2023_mouse` | RNA |  |  |  |  |  | missing evidence |
| `zemke2023_joint` | ATAC | `lora+locon` | 1 | 0.4333 | 0.6540 | 0.3667 | below target |
| `zemke2023_joint` | RNA | `lora` | 1 | 0.2950 | 0.1990 | 0.5050 | below target |
| `zemke2024-all` | ATAC | `lora+locon` | 4 | 0.7499 | 0.7599 | 0.0501 | below target |
| `zemke2024-all` | RNA | `lora` | 5 | 0.4759 | 0.2925 | 0.3241 | below target |
