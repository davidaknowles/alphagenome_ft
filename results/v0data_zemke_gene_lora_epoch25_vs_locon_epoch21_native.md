# Joint native-source evaluation

Each row evaluates one head from the same union model on one native source. R is signed double-centered Pearson correlation.

| Dataset | Source | Strategy | Epoch | Head | Validation R | Test R |
|---|---|---|---:|---|---:|---:|
| `hda` | `human` | `LoRA` | 25 | `hda_atac` | 0.8004 | 0.8133 |
| `hda` | `human` | `LoRA` | 25 | `hda_rna` | 0.5431 | 0.6223 |
| `liu_hdma` | `human` | `LoRA` | 25 | `liu_atac` | 0.7727 | 0.7911 |
| `liu_hdma` | `human` | `LoRA` | 25 | `liu_rna` | 0.6568 | 0.5086 |
| `johansen2025` | `human` | `LoRA` | 25 | `allen_atac` | 0.5641 | 0.6011 |
| `johansen2025` | `human` | `LoRA` | 25 | `allen_rna` | 0.5632 | 0.5391 |
| `johansen2025` | `macaque` | `LoRA` | 25 | `allen_atac` | 0.5762 | 0.4298 |
| `johansen2025` | `macaque` | `LoRA` | 25 | `allen_rna` | 0.3138 | 0.5016 |
| `johansen2025` | `marmoset` | `LoRA` | 25 | `allen_atac` | 0.4715 | 0.5063 |
| `johansen2025` | `marmoset` | `LoRA` | 25 | `allen_rna` | 0.2425 | 0.4094 |
| `zemke2023` | `human` | `LoRA` | 25 | `zemke2023_atac` | 0.6848 | 0.7132 |
| `zemke2023` | `human` | `LoRA` | 25 | `zemke2023_rna` | 0.5459 | 0.5003 |
| `zemke2023` | `macaque` | `LoRA` | 25 | `zemke2023_atac` | 0.3159 | 0.6215 |
| `zemke2023` | `macaque` | `LoRA` | 25 | `zemke2023_rna` | 0.4365 | 0.5590 |
| `zemke2023` | `marmoset` | `LoRA` | 25 | `zemke2023_atac` | 0.6156 | 0.5858 |
| `zemke2023` | `marmoset` | `LoRA` | 25 | `zemke2023_rna` | 0.4385 | 0.5527 |
| `zemke2023` | `mouse` | `LoRA` | 25 | `zemke2023_atac` | 0.7151 | 0.7131 |
| `zemke2023` | `mouse` | `LoRA` | 25 | `zemke2023_rna` | 0.5278 | 0.5030 |
| `zemke2024` | `human` | `LoRA` | 25 | `zemke2024_all_atac` | 0.7436 | 0.7564 |
| `zemke2024` | `human` | `LoRA` | 25 | `zemke2024_all_rna` | 0.5721 | 0.3047 |
| `hda` | `human` | `LoRA+LoCon` | 21 | `hda_atac` | 0.7977 | 0.8118 |
| `hda` | `human` | `LoRA+LoCon` | 21 | `hda_rna` | 0.5280 | 0.6787 |
| `liu_hdma` | `human` | `LoRA+LoCon` | 21 | `liu_atac` | 0.7728 | 0.7904 |
| `liu_hdma` | `human` | `LoRA+LoCon` | 21 | `liu_rna` | 0.5903 | 0.5167 |
| `johansen2025` | `human` | `LoRA+LoCon` | 21 | `allen_atac` | 0.5593 | 0.5966 |
| `johansen2025` | `human` | `LoRA+LoCon` | 21 | `allen_rna` | 0.5639 | 0.5528 |
| `johansen2025` | `macaque` | `LoRA+LoCon` | 21 | `allen_atac` | 0.5723 | 0.4221 |
| `johansen2025` | `macaque` | `LoRA+LoCon` | 21 | `allen_rna` | 0.3311 | 0.4967 |
| `johansen2025` | `marmoset` | `LoRA+LoCon` | 21 | `allen_atac` | 0.4664 | 0.4985 |
| `johansen2025` | `marmoset` | `LoRA+LoCon` | 21 | `allen_rna` | 0.2244 | 0.4060 |
| `zemke2023` | `human` | `LoRA+LoCon` | 21 | `zemke2023_atac` | 0.6773 | 0.7070 |
| `zemke2023` | `human` | `LoRA+LoCon` | 21 | `zemke2023_rna` | 0.4951 | 0.5192 |
| `zemke2023` | `macaque` | `LoRA+LoCon` | 21 | `zemke2023_atac` | 0.2562 | 0.6136 |
| `zemke2023` | `macaque` | `LoRA+LoCon` | 21 | `zemke2023_rna` | 0.4334 | 0.5683 |
| `zemke2023` | `marmoset` | `LoRA+LoCon` | 21 | `zemke2023_atac` | 0.6095 | 0.5795 |
| `zemke2023` | `marmoset` | `LoRA+LoCon` | 21 | `zemke2023_rna` | 0.4389 | 0.5672 |
| `zemke2023` | `mouse` | `LoRA+LoCon` | 21 | `zemke2023_atac` | 0.7064 | 0.7052 |
| `zemke2023` | `mouse` | `LoRA+LoCon` | 21 | `zemke2023_rna` | 0.4788 | 0.4914 |
| `zemke2024` | `human` | `LoRA+LoCon` | 21 | `zemke2024_all_atac` | 0.7400 | 0.7503 |
| `zemke2024` | `human` | `LoRA+LoCon` | 21 | `zemke2024_all_rna` | 0.4891 | 0.2949 |

| Strategy | Native sources | Heads | Mean validation R | Mean test R | ATAC validation R | ATAC test R | RNA validation R | RNA test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `LoRA` | 10 | 20 | 0.5550 | 0.5766 | 0.6260 | 0.6532 | 0.4840 | 0.5001 |
| `LoRA+LoCon` | 10 | 20 | 0.5365 | 0.5783 | 0.6158 | 0.6475 | 0.4573 | 0.5092 |
