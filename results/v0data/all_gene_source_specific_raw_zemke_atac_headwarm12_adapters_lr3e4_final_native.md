# Joint native-source evaluation

Each row evaluates one head from the same union model on one native source. R is signed double-centered Pearson correlation.

| Dataset | Source | Strategy | Epoch | Head | Validation R | Test R |
|---|---|---|---:|---|---:|---:|
| `hda` | `human` | `lora` | 20 | `hda_atac` | 0.7986 | 0.8128 |
| `hda` | `human` | `lora` | 20 | `hda_rna` | 0.5338 | 0.6736 |
| `liu_hdma` | `human` | `lora` | 20 | `liu_atac` | 0.7680 | 0.7855 |
| `liu_hdma` | `human` | `lora` | 20 | `liu_rna` | 0.6787 | 0.5091 |
| `johansen2025` | `human` | `lora` | 20 | `allen_atac_human` | 0.6024 | 0.6368 |
| `johansen2025` | `human` | `lora` | 20 | `allen_rna_human` | 0.5407 | 0.5758 |
| `johansen2025` | `macaque` | `lora` | 20 | `allen_atac_macaque` | 0.5887 | 0.4537 |
| `johansen2025` | `macaque` | `lora` | 20 | `allen_rna_macaque` | 0.3287 | 0.5208 |
| `johansen2025` | `marmoset` | `lora` | 20 | `allen_atac_marmoset` | 0.5499 | 0.5888 |
| `johansen2025` | `marmoset` | `lora` | 20 | `allen_rna_marmoset` | 0.2278 | 0.4264 |
| `zemke2023` | `human` | `lora` | 20 | `zemke2023_atac_human` | 0.7092 | 0.7341 |
| `zemke2023` | `human` | `lora` | 20 | `zemke2023_rna_human` | 0.5921 | 0.5906 |
| `zemke2023` | `macaque` | `lora` | 20 | `zemke2023_atac_macaque` | 0.3998 | 0.6494 |
| `zemke2023` | `macaque` | `lora` | 20 | `zemke2023_rna_macaque` | 0.5772 | 0.6482 |
| `zemke2023` | `marmoset` | `lora` | 20 | `zemke2023_atac_marmoset` | 0.6379 | 0.6037 |
| `zemke2023` | `marmoset` | `lora` | 20 | `zemke2023_rna_marmoset` | 0.5039 | 0.6590 |
| `zemke2023` | `mouse` | `lora` | 20 | `zemke2023_atac_mouse` | 0.7300 | 0.7287 |
| `zemke2023` | `mouse` | `lora` | 20 | `zemke2023_rna_mouse` | 0.5262 | 0.6026 |
| `zemke2024` | `human` | `lora` | 20 | `zemke2024_all_atac` | 0.7412 | 0.7642 |
| `zemke2024` | `human` | `lora` | 20 | `zemke2024_all_rna` | 0.5565 | 0.4765 |
| `hda` | `human` | `lora+locon` | 21 | `hda_atac` | 0.8028 | 0.8172 |
| `hda` | `human` | `lora+locon` | 21 | `hda_rna` | 0.5200 | 0.6597 |
| `liu_hdma` | `human` | `lora+locon` | 21 | `liu_atac` | 0.7649 | 0.7837 |
| `liu_hdma` | `human` | `lora+locon` | 21 | `liu_rna` | 0.5247 | 0.4847 |
| `johansen2025` | `human` | `lora+locon` | 21 | `allen_atac_human` | 0.6107 | 0.6441 |
| `johansen2025` | `human` | `lora+locon` | 21 | `allen_rna_human` | 0.5942 | 0.5835 |
| `johansen2025` | `macaque` | `lora+locon` | 21 | `allen_atac_macaque` | 0.5948 | 0.4629 |
| `johansen2025` | `macaque` | `lora+locon` | 21 | `allen_rna_macaque` | 0.3280 | 0.5236 |
| `johansen2025` | `marmoset` | `lora+locon` | 21 | `allen_atac_marmoset` | 0.5606 | 0.6000 |
| `johansen2025` | `marmoset` | `lora+locon` | 21 | `allen_rna_marmoset` | 0.2339 | 0.4533 |
| `zemke2023` | `human` | `lora+locon` | 21 | `zemke2023_atac_human` | 0.7182 | 0.7427 |
| `zemke2023` | `human` | `lora+locon` | 21 | `zemke2023_rna_human` | 0.6509 | 0.5719 |
| `zemke2023` | `macaque` | `lora+locon` | 21 | `zemke2023_atac_macaque` | 0.4553 | 0.6628 |
| `zemke2023` | `macaque` | `lora+locon` | 21 | `zemke2023_rna_macaque` | 0.5490 | 0.6786 |
| `zemke2023` | `marmoset` | `lora+locon` | 21 | `zemke2023_atac_marmoset` | 0.6538 | 0.6146 |
| `zemke2023` | `marmoset` | `lora+locon` | 21 | `zemke2023_rna_marmoset` | 0.4925 | 0.6486 |
| `zemke2023` | `mouse` | `lora+locon` | 21 | `zemke2023_atac_mouse` | 0.7408 | 0.7391 |
| `zemke2023` | `mouse` | `lora+locon` | 21 | `zemke2023_rna_mouse` | 0.5394 | 0.6103 |
| `zemke2024` | `human` | `lora+locon` | 21 | `zemke2024_all_atac` | 0.7462 | 0.7694 |
| `zemke2024` | `human` | `lora+locon` | 21 | `zemke2024_all_rna` | 0.6102 | 0.4838 |

| Strategy | Native sources | Heads | Mean validation R | Mean test R | ATAC validation R | ATAC test R | RNA validation R | RNA test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora` | 10 | 20 | 0.5796 | 0.6220 | 0.6526 | 0.6758 | 0.5066 | 0.5683 |
| `lora+locon` | 10 | 20 | 0.5846 | 0.6267 | 0.6648 | 0.6836 | 0.5043 | 0.5698 |
