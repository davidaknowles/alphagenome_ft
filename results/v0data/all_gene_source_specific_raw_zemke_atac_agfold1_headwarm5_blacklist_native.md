# Joint native-source evaluation

Each row evaluates both modality heads from the same union model on one native source. R is signed double-centered Pearson correlation.

| Dataset | Source | Strategy | Epoch | ATAC valid R | ATAC test R | RNA valid R | RNA test R |
|---|---|---|---:|---:|---:|---:|---:|
| `hda` | `human` | `head_only` | 5 | 0.7428 | 0.7499 | 0.3917 | 0.1524 |
| `liu_hdma` | `human` | `head_only` | 5 | 0.7228 | 0.7450 | 0.2955 | 0.2180 |
| `johansen2025` | `human` | `head_only` | 5 | 0.5392 | 0.5586 | 0.2710 | 0.3163 |
| `johansen2025` | `macaque` | `head_only` | 5 | 0.5428 | 0.5713 | 0.3191 | 0.2823 |
| `johansen2025` | `marmoset` | `head_only` | 5 | 0.4818 | 0.4834 | 0.2094 | 0.2319 |
| `zemke2023` | `human` | `head_only` | 5 | 0.6391 | 0.6558 | 0.3373 | 0.2727 |
| `zemke2023` | `macaque` | `head_only` | 5 | 0.5847 | 0.5870 | 0.4515 | 0.3200 |
| `zemke2023` | `marmoset` | `head_only` | 5 | 0.6082 | 0.5890 | 0.5054 | 0.4230 |
| `zemke2023` | `mouse` | `head_only` | 5 | 0.6348 | 0.6470 | 0.1008 | 0.2709 |
| `zemke2024` | `human` | `head_only` | 5 | 0.6612 | 0.6854 | 0.3358 | 0.3325 |
| `hda` | `human` | `lora` | 18 | 0.7646 | 0.7689 | 0.3965 | 0.4123 |
| `liu_hdma` | `human` | `lora` | 18 | 0.7293 | 0.7524 | 0.4326 | 0.2352 |
| `johansen2025` | `human` | `lora` | 18 | 0.5722 | 0.5881 | 0.2522 | 0.4255 |
| `johansen2025` | `macaque` | `lora` | 18 | 0.5849 | 0.6073 | 0.4630 | 0.3674 |
| `johansen2025` | `marmoset` | `lora` | 18 | 0.5394 | 0.5414 | 0.2914 | 0.2292 |
| `zemke2023` | `human` | `lora` | 18 | 0.6721 | 0.6819 | 0.3874 | 0.3193 |
| `zemke2023` | `macaque` | `lora` | 18 | 0.6267 | 0.6287 | 0.5414 | 0.3979 |
| `zemke2023` | `marmoset` | `lora` | 18 | 0.6586 | 0.6408 | 0.5999 | 0.4339 |
| `zemke2023` | `mouse` | `lora` | 18 | 0.6842 | 0.6865 | 0.0969 | 0.3468 |
| `zemke2024` | `human` | `lora` | 18 | 0.6925 | 0.7104 | 0.3540 | 0.3615 |
| `hda` | `human` | `lora+locon` | 15 | 0.7671 | 0.7713 | 0.3670 | 0.1932 |
| `liu_hdma` | `human` | `lora+locon` | 15 | 0.7255 | 0.7500 | 0.3325 | 0.2925 |
| `johansen2025` | `human` | `lora+locon` | 15 | 0.5761 | 0.5912 | 0.3290 | 0.4046 |
| `johansen2025` | `macaque` | `lora+locon` | 15 | 0.5922 | 0.6125 | 0.5312 | 0.3641 |
| `johansen2025` | `marmoset` | `lora+locon` | 15 | 0.5481 | 0.5517 | 0.2953 | 0.2230 |
| `zemke2023` | `human` | `lora+locon` | 15 | 0.6786 | 0.6885 | 0.3821 | 0.3095 |
| `zemke2023` | `macaque` | `lora+locon` | 15 | 0.6383 | 0.6376 | 0.5734 | 0.3723 |
| `zemke2023` | `marmoset` | `lora+locon` | 15 | 0.6702 | 0.6520 | 0.5723 | 0.4639 |
| `zemke2023` | `mouse` | `lora+locon` | 15 | 0.6912 | 0.7010 | 0.1193 | 0.3716 |
| `zemke2024` | `human` | `lora+locon` | 15 | 0.6960 | 0.7149 | 0.3681 | 0.3363 |

| Strategy | Native sources | Heads | Mean validation R | Mean test R | ATAC validation R | ATAC test R | RNA validation R | RNA test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `head_only` | 10 | 20 | 0.4687 | 0.4546 | 0.6157 | 0.6272 | 0.3217 | 0.2820 |
| `lora` | 10 | 20 | 0.5170 | 0.5068 | 0.6524 | 0.6606 | 0.3815 | 0.3529 |
| `lora+locon` | 10 | 20 | 0.5227 | 0.5001 | 0.6583 | 0.6671 | 0.3870 | 0.3331 |
