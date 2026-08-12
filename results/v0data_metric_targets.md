# Canonical metric-target audit

The table reports the stronger of the two strategy-selected canonical checkpoints for each modality. Each strategy checkpoint is selected by mean validation signed double-centered Pearson correlation across its heads. Missing evidence is distinct from a measured value below the target.

| Dataset | Modality | Strategy | Epoch | Validation R | Test R | Gap to 0.8 | Status |
|---|---|---|---:|---:|---:|---:|---|
| `hda` | ATAC | `lora+locon` | 4 | 0.8086 | 0.8166 | 0.0000 | target reached |
| `hda-joint` | ATAC | `lora+locon` | 3 | 0.8035 | 0.8127 | 0.0000 | target reached |
| `hda-joint` | RNA | `lora+locon` | 3 | 0.5190 | 0.6089 | 0.2810 | below target |
| `johansen-human` | ATAC | `lora+locon` | 3 | 0.5886 | 0.6218 | 0.2114 | below target |
| `johansen-human` | RNA | `lora+locon` | 3 | 0.5237 | 0.4588 | 0.2763 | below target |
| `johansen_joint` | ATAC | `lora+locon` | 1 | 0.5339 | 0.4926 | 0.2661 | below target |
| `johansen_joint` | RNA | `lora+locon` | 1 | 0.2667 | 0.3933 | 0.5333 | below target |
| `liu-hdma` | ATAC | `lora+locon` | 3 | 0.7723 | 0.7895 | 0.0277 | below target |
| `liu-hdma` | RNA | `lora` | 3 | 0.6481 | 0.4886 | 0.1519 | below target |
| `zemke2023-human` | ATAC | `lora+locon` | 4 | 0.6971 | 0.7151 | 0.1029 | below target |
| `zemke2023-human` | RNA | `lora+locon` | 4 | 0.4987 | 0.4099 | 0.3013 | below target |
| `zemke2023_macaque` | ATAC | `lora` | 2 | 0.2603 | 0.6016 | 0.5397 | below target |
| `zemke2023_macaque` | RNA | `lora` | 2 | 0.2851 | 0.3004 | 0.5149 | below target |
| `zemke2023_marmoset` | ATAC | `lora+locon` | 1 | 0.5799 | 0.5527 | 0.2201 | below target |
| `zemke2023_marmoset` | RNA | `lora+locon` | 1 | 0.2800 | 0.1918 | 0.5200 | below target |
| `zemke2023_mouse` | ATAC | `lora` | 2 | 0.6886 | 0.6871 | 0.1114 | below target |
| `zemke2023_mouse` | RNA | `lora` | 2 | 0.2448 | 0.3019 | 0.5552 | below target |
| `zemke2023_joint` | ATAC | `lora+locon` | 1 | 0.4333 | 0.6540 | 0.3667 | below target |
| `zemke2023_joint` | RNA | `lora` | 1 | 0.2950 | 0.1990 | 0.5050 | below target |
| `zemke2024-all` | ATAC | `lora+locon` | 4 | 0.7499 | 0.7599 | 0.0501 | below target |
| `zemke2024-all` | RNA | `lora` | 5 | 0.4759 | 0.2925 | 0.3241 | below target |
