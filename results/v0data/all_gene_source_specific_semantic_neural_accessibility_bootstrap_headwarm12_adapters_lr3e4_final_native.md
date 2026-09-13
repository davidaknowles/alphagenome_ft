# Joint native-source evaluation

Each row evaluates one head from the same union model on one native source. R is signed double-centered Pearson correlation.

| Dataset | Source | Strategy | Epoch | Head | Validation R | Test R |
|---|---|---|---:|---|---:|---:|
| `hda` | `human` | `lora` | 17 | `hda_atac` | 0.7987 | 0.8127 |
| `hda` | `human` | `lora` | 17 | `hda_rna` | 0.5391 | 0.6643 |
| `liu_hdma` | `human` | `lora` | 17 | `liu_atac` | 0.7693 | 0.7875 |
| `liu_hdma` | `human` | `lora` | 17 | `liu_rna` | 0.6036 | 0.4772 |
| `johansen2025` | `human` | `lora` | 17 | `allen_atac_human` | 0.6015 | 0.6365 |
| `johansen2025` | `human` | `lora` | 17 | `allen_rna_human` | 0.5123 | 0.5631 |
| `johansen2025` | `macaque` | `lora` | 17 | `allen_atac_macaque` | 0.5725 | 0.4717 |
| `johansen2025` | `macaque` | `lora` | 17 | `allen_rna_macaque` | 0.3055 | 0.5151 |
| `johansen2025` | `marmoset` | `lora` | 17 | `allen_atac_marmoset` | 0.5468 | 0.5858 |
| `johansen2025` | `marmoset` | `lora` | 17 | `allen_rna_marmoset` | 0.2182 | 0.4055 |
| `zemke2023` | `human` | `lora` | 17 | `zemke2023_atac_human` | 0.7088 | 0.7346 |
| `zemke2023` | `human` | `lora` | 17 | `zemke2023_rna_human` | 0.5985 | 0.6292 |
| `zemke2023` | `macaque` | `lora` | 17 | `zemke2023_atac_macaque` | 0.4699 | 0.6493 |
| `zemke2023` | `macaque` | `lora` | 17 | `zemke2023_rna_macaque` | 0.6488 | 0.6593 |
| `zemke2023` | `marmoset` | `lora` | 17 | `zemke2023_atac_marmoset` | 0.6371 | 0.6017 |
| `zemke2023` | `marmoset` | `lora` | 17 | `zemke2023_rna_marmoset` | 0.4961 | 0.6359 |
| `zemke2023` | `mouse` | `lora` | 17 | `zemke2023_atac_mouse` | 0.7295 | 0.7278 |
| `zemke2023` | `mouse` | `lora` | 17 | `zemke2023_rna_mouse` | 0.5470 | 0.3113 |
| `zemke2024` | `human` | `lora` | 17 | `zemke2024_all_atac` | 0.7454 | 0.7595 |
| `zemke2024` | `human` | `lora` | 17 | `zemke2024_all_rna` | 0.5423 | 0.4767 |
| `hda` | `human` | `lora+locon` | 15 | `hda_atac` | 0.8027 | 0.8161 |
| `hda` | `human` | `lora+locon` | 15 | `hda_rna` | 0.5547 | 0.6508 |
| `liu_hdma` | `human` | `lora+locon` | 15 | `liu_atac` | 0.7681 | 0.7874 |
| `liu_hdma` | `human` | `lora+locon` | 15 | `liu_rna` | 0.6400 | 0.4725 |
| `johansen2025` | `human` | `lora+locon` | 15 | `allen_atac_human` | 0.6058 | 0.6392 |
| `johansen2025` | `human` | `lora+locon` | 15 | `allen_rna_human` | 0.5861 | 0.5220 |
| `johansen2025` | `macaque` | `lora+locon` | 15 | `allen_atac_macaque` | 0.6004 | 0.4556 |
| `johansen2025` | `macaque` | `lora+locon` | 15 | `allen_rna_macaque` | 0.3047 | 0.5113 |
| `johansen2025` | `marmoset` | `lora+locon` | 15 | `allen_atac_marmoset` | 0.5530 | 0.5919 |
| `johansen2025` | `marmoset` | `lora+locon` | 15 | `allen_rna_marmoset` | 0.2025 | 0.4128 |
| `zemke2023` | `human` | `lora+locon` | 15 | `zemke2023_atac_human` | 0.7155 | 0.7367 |
| `zemke2023` | `human` | `lora+locon` | 15 | `zemke2023_rna_human` | 0.6173 | 0.6145 |
| `zemke2023` | `macaque` | `lora+locon` | 15 | `zemke2023_atac_macaque` | 0.3565 | 0.6558 |
| `zemke2023` | `macaque` | `lora+locon` | 15 | `zemke2023_rna_macaque` | 0.6246 | 0.6528 |
| `zemke2023` | `marmoset` | `lora+locon` | 15 | `zemke2023_atac_marmoset` | 0.6444 | 0.6095 |
| `zemke2023` | `marmoset` | `lora+locon` | 15 | `zemke2023_rna_marmoset` | 0.4535 | 0.5945 |
| `zemke2023` | `mouse` | `lora+locon` | 15 | `zemke2023_atac_mouse` | 0.7358 | 0.7376 |
| `zemke2023` | `mouse` | `lora+locon` | 15 | `zemke2023_rna_mouse` | 0.5068 | 0.4108 |
| `zemke2024` | `human` | `lora+locon` | 15 | `zemke2024_all_atac` | 0.7512 | 0.7613 |
| `zemke2024` | `human` | `lora+locon` | 15 | `zemke2024_all_rna` | 0.5951 | 0.4299 |

| Strategy | Native sources | Heads | Mean validation R | Mean test R | ATAC validation R | ATAC test R | RNA validation R | RNA test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora` | 10 | 20 | 0.5795 | 0.6052 | 0.6579 | 0.6767 | 0.5011 | 0.5338 |
| `lora+locon` | 10 | 20 | 0.5809 | 0.6032 | 0.6533 | 0.6791 | 0.5085 | 0.5272 |
