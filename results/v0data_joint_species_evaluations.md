# Joint-checkpoint native-species evaluations

Each joint cross-species checkpoint is evaluated without parameter updates against the native reference and target manifest for each species.

| Dataset | Species | Strategy | Epoch | Split | Head | Signed double-centered R |
|---|---|---|---:|---|---|---:|
| `zemke2023_joint` | `mouse` | `lora` | 1 | `test` | `zemke2023_atac` | 0.6905 |
| `zemke2023_joint` | `mouse` | `lora` | 1 | `test` | `zemke2023_rna` | 0.2734 |
| `zemke2023_joint` | `mouse` | `lora` | 1 | `valid` | `zemke2023_atac` | 0.6909 |
| `zemke2023_joint` | `mouse` | `lora` | 1 | `valid` | `zemke2023_rna` | 0.2185 |
