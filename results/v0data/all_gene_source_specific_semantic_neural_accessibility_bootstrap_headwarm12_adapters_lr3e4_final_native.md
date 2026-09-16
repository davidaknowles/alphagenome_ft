# Joint native-source evaluation

Each row evaluates both modality heads from the same union model on one native source. R is signed double-centered Pearson correlation.

| Dataset | Source | Strategy | Epoch | ATAC valid R | ATAC test R | RNA valid R | RNA test R |
|---|---|---|---:|---:|---:|---:|---:|
| `hda` | `human` | `lora` | 17 | 0.7987 | 0.8127 | 0.5391 | 0.6643 |
| `liu_hdma` | `human` | `lora` | 17 | 0.7693 | 0.7875 | 0.6036 | 0.4772 |
| `johansen2025` | `human` | `lora` | 17 | 0.6015 | 0.6365 | 0.5123 | 0.5631 |
| `johansen2025` | `macaque` | `lora` | 17 | 0.5725 | 0.4717 | 0.3055 | 0.5151 |
| `johansen2025` | `marmoset` | `lora` | 17 | 0.5468 | 0.5858 | 0.2182 | 0.4055 |
| `zemke2023` | `human` | `lora` | 17 | 0.7088 | 0.7346 | 0.5985 | 0.6292 |
| `zemke2023` | `macaque` | `lora` | 17 | 0.4699 | 0.6493 | 0.6488 | 0.6593 |
| `zemke2023` | `marmoset` | `lora` | 17 | 0.6371 | 0.6017 | 0.4961 | 0.6359 |
| `zemke2023` | `mouse` | `lora` | 17 | 0.7295 | 0.7278 | 0.5470 | 0.3113 |
| `zemke2024` | `human` | `lora` | 17 | 0.7454 | 0.7595 | 0.5423 | 0.4767 |
| `hda` | `human` | `lora+locon` | 15 | 0.8027 | 0.8161 | 0.5547 | 0.6508 |
| `liu_hdma` | `human` | `lora+locon` | 15 | 0.7681 | 0.7874 | 0.6400 | 0.4725 |
| `johansen2025` | `human` | `lora+locon` | 15 | 0.6058 | 0.6392 | 0.5861 | 0.5220 |
| `johansen2025` | `macaque` | `lora+locon` | 15 | 0.6004 | 0.4556 | 0.3047 | 0.5113 |
| `johansen2025` | `marmoset` | `lora+locon` | 15 | 0.5530 | 0.5919 | 0.2025 | 0.4128 |
| `zemke2023` | `human` | `lora+locon` | 15 | 0.7155 | 0.7367 | 0.6173 | 0.6145 |
| `zemke2023` | `macaque` | `lora+locon` | 15 | 0.3565 | 0.6558 | 0.6246 | 0.6528 |
| `zemke2023` | `marmoset` | `lora+locon` | 15 | 0.6444 | 0.6095 | 0.4535 | 0.5945 |
| `zemke2023` | `mouse` | `lora+locon` | 15 | 0.7358 | 0.7376 | 0.5068 | 0.4108 |
| `zemke2024` | `human` | `lora+locon` | 15 | 0.7512 | 0.7613 | 0.5951 | 0.4299 |

| Strategy | Native sources | Heads | Mean validation R | Mean test R | ATAC validation R | ATAC test R | RNA validation R | RNA test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora` | 10 | 20 | 0.5795 | 0.6052 | 0.6579 | 0.6767 | 0.5011 | 0.5338 |
| `lora+locon` | 10 | 20 | 0.5809 | 0.6032 | 0.6533 | 0.6791 | 0.5085 | 0.5272 |
