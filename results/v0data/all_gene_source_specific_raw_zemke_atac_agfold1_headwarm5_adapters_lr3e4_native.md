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
| `hda` | `human` | `lora+locon` | 10 | 0.7657 | 0.7703 | 0.3538 | 0.2257 |
| `liu_hdma` | `human` | `lora+locon` | 10 | 0.7265 | 0.7514 | 0.2601 | 0.2821 |
| `johansen2025` | `human` | `lora+locon` | 10 | 0.5744 | 0.5903 | 0.3033 | 0.4354 |
| `johansen2025` | `macaque` | `lora+locon` | 10 | 0.5905 | 0.6100 | 0.5246 | 0.3744 |
| `johansen2025` | `marmoset` | `lora+locon` | 10 | 0.5448 | 0.5486 | 0.2553 | 0.2286 |
| `zemke2023` | `human` | `lora+locon` | 10 | 0.6785 | 0.6885 | 0.3771 | 0.3197 |
| `zemke2023` | `macaque` | `lora+locon` | 10 | 0.3021 | 0.4732 | 0.5841 | 0.3882 |
| `zemke2023` | `marmoset` | `lora+locon` | 10 | 0.6677 | 0.3395 | 0.5240 | 0.4524 |
| `zemke2023` | `mouse` | `lora+locon` | 10 | 0.6950 | 0.6998 | 0.1125 | 0.3727 |
| `zemke2024` | `human` | `lora+locon` | 10 | 0.6961 | 0.7152 | 0.3581 | 0.3626 |

| Strategy | Native sources | Heads | Mean validation R | Mean test R | ATAC validation R | ATAC test R | RNA validation R | RNA test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora` | 10 | 20 | 0.4998 | 0.4819 | 0.6181 | 0.6111 | 0.3816 | 0.3526 |
| `lora+locon` | 10 | 20 | 0.4947 | 0.4814 | 0.6241 | 0.6187 | 0.3653 | 0.3442 |
