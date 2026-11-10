# Joint native-source evaluation

Each row evaluates both modality heads from the same union model on one native source. R is signed double-centered Pearson correlation.

| Dataset | Source | Strategy | Epoch | ATAC valid R | ATAC test R | RNA valid R | RNA test R |
|---|---|---|---:|---:|---:|---:|---:|
| `hda` | `human` | `lora` | 18 | 0.7646 | 0.7689 | 0.3965 | 0.4123 |
| `liu_hdma` | `human` | `lora` | 18 | 0.7293 | 0.7524 | 0.4326 | 0.2352 |
| `johansen2025` | `human` | `lora` | 18 | 0.5722 | 0.5881 | 0.2522 | 0.4255 |
| `johansen2025` | `macaque` | `lora` | 18 | 0.5849 | 0.6073 | 0.4630 | 0.3674 |
| `johansen2025` | `marmoset` | `lora` | 18 | 0.5394 | 0.5414 | 0.2914 | 0.2292 |
| `zemke2023` | `human` | `lora` | 18 | 0.6721 | 0.6819 | 0.3874 | 0.3193 |
| `zemke2023` | `macaque` | `lora` | 18 | 0.2829 | 0.4956 | 0.5416 | 0.3980 |
| `zemke2023` | `marmoset` | `lora` | 18 | 0.6586 | 0.2788 | 0.5999 | 0.4311 |
| `zemke2023` | `mouse` | `lora` | 18 | 0.6842 | 0.6865 | 0.0969 | 0.3468 |
| `zemke2024` | `human` | `lora` | 18 | 0.6925 | 0.7104 | 0.3540 | 0.3615 |
| `hda` | `human` | `lora+locon` | 15 | 0.7683 | 0.7722 | 0.3658 | 0.1954 |
| `liu_hdma` | `human` | `lora+locon` | 15 | 0.7255 | 0.7500 | 0.3325 | 0.2925 |
| `johansen2025` | `human` | `lora+locon` | 15 | 0.5761 | 0.5912 | 0.3290 | 0.4046 |
| `johansen2025` | `macaque` | `lora+locon` | 15 | 0.5922 | 0.6125 | 0.5312 | 0.3641 |
| `johansen2025` | `marmoset` | `lora+locon` | 15 | 0.5489 | 0.5524 | 0.2954 | 0.2209 |
| `zemke2023` | `human` | `lora+locon` | 15 | 0.6787 | 0.6886 | 0.3809 | 0.3072 |
| `zemke2023` | `macaque` | `lora+locon` | 15 | 0.2921 | 0.4661 | 0.5734 | 0.3724 |
| `zemke2023` | `marmoset` | `lora+locon` | 15 | 0.6702 | 0.4750 | 0.5703 | 0.4624 |
| `zemke2023` | `mouse` | `lora+locon` | 15 | 0.6912 | 0.7010 | 0.1193 | 0.3716 |
| `zemke2024` | `human` | `lora+locon` | 15 | 0.6964 | 0.7153 | 0.3674 | 0.3362 |

| Strategy | Native sources | Heads | Mean validation R | Mean test R | ATAC validation R | ATAC test R | RNA validation R | RNA test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora` | 10 | 20 | 0.4998 | 0.4819 | 0.6181 | 0.6111 | 0.3816 | 0.3526 |
| `lora+locon` | 10 | 20 | 0.5052 | 0.4826 | 0.6240 | 0.6324 | 0.3865 | 0.3327 |
