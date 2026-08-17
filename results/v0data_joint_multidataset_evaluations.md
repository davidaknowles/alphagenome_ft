# Joint native-source evaluation

Each row evaluates one head from the same union model on one native source. R is signed double-centered Pearson correlation.

| Dataset | Source | Strategy | Epoch | Head | Validation R | Test R |
|---|---|---|---:|---|---:|---:|
| `hda` | `human` | `lora` | 9 | `hda_atac` | 0.7913 | 0.8026 |
| `hda` | `human` | `lora` | 9 | `hda_rna` | 0.4679 | 0.6240 |
| `liu_hdma` | `human` | `lora` | 9 | `liu_atac` | 0.7644 | 0.7837 |
| `liu_hdma` | `human` | `lora` | 9 | `liu_rna` | 0.7755 | 0.4845 |
| `johansen2025` | `human` | `lora` | 9 | `allen_atac` | 0.5707 | 0.6040 |
| `johansen2025` | `human` | `lora` | 9 | `allen_rna` | 0.5393 | 0.5219 |
| `johansen2025` | `macaque` | `lora` | 9 | `allen_atac` | 0.5791 | 0.4268 |
| `johansen2025` | `macaque` | `lora` | 9 | `allen_rna` | 0.3301 | 0.4835 |
| `johansen2025` | `marmoset` | `lora` | 9 | `allen_atac` | 0.5158 | 0.5524 |
| `johansen2025` | `marmoset` | `lora` | 9 | `allen_rna` | 0.2290 | 0.4305 |
| `zemke2023` | `human` | `lora` | 9 | `zemke2023_atac` | 0.6853 | 0.7081 |
| `zemke2023` | `human` | `lora` | 9 | `zemke2023_rna` | 0.3557 | 0.3782 |
| `zemke2023` | `macaque` | `lora` | 9 | `zemke2023_atac` | 0.2311 | 0.6064 |
| `zemke2023` | `macaque` | `lora` | 9 | `zemke2023_rna` | 0.2550 | 0.3636 |
| `zemke2023` | `marmoset` | `lora` | 9 | `zemke2023_atac` | 0.6057 | 0.5723 |
| `zemke2023` | `marmoset` | `lora` | 9 | `zemke2023_rna` | 0.2929 | 0.1424 |
| `zemke2023` | `mouse` | `lora` | 9 | `zemke2023_atac` | 0.6944 | 0.6937 |
| `zemke2023` | `mouse` | `lora` | 9 | `zemke2023_rna` | 0.2868 | 0.2800 |
| `zemke2024` | `human` | `lora` | 9 | `zemke2024_all_atac` | 0.7388 | 0.7485 |
| `zemke2024` | `human` | `lora` | 9 | `zemke2024_all_rna` | 0.3997 | 0.2848 |
| `hda` | `human` | `lora+locon` | 6 | `hda_atac` | 0.7938 | 0.8072 |
| `hda` | `human` | `lora+locon` | 6 | `hda_rna` | 0.4574 | 0.6339 |
| `liu_hdma` | `human` | `lora+locon` | 6 | `liu_atac` | 0.7636 | 0.7816 |
| `liu_hdma` | `human` | `lora+locon` | 6 | `liu_rna` | 0.6735 | 0.4759 |
| `johansen2025` | `human` | `lora+locon` | 6 | `allen_atac` | 0.5676 | 0.6030 |
| `johansen2025` | `human` | `lora+locon` | 6 | `allen_rna` | 0.5600 | 0.4810 |
| `johansen2025` | `macaque` | `lora+locon` | 6 | `allen_atac` | 0.5734 | 0.4265 |
| `johansen2025` | `macaque` | `lora+locon` | 6 | `allen_rna` | 0.2964 | 0.4214 |
| `johansen2025` | `marmoset` | `lora+locon` | 6 | `allen_atac` | 0.4957 | 0.5292 |
| `johansen2025` | `marmoset` | `lora+locon` | 6 | `allen_rna` | 0.1987 | 0.4119 |
| `zemke2023` | `human` | `lora+locon` | 6 | `zemke2023_atac` | 0.6817 | 0.7070 |
| `zemke2023` | `human` | `lora+locon` | 6 | `zemke2023_rna` | 0.3244 | 0.3479 |
| `zemke2023` | `macaque` | `lora+locon` | 6 | `zemke2023_atac` | 0.2183 | 0.6103 |
| `zemke2023` | `macaque` | `lora+locon` | 6 | `zemke2023_rna` | 0.2420 | 0.3180 |
| `zemke2023` | `marmoset` | `lora+locon` | 6 | `zemke2023_atac` | 0.5957 | 0.5644 |
| `zemke2023` | `marmoset` | `lora+locon` | 6 | `zemke2023_rna` | 0.2656 | 0.1688 |
| `zemke2023` | `mouse` | `lora+locon` | 6 | `zemke2023_atac` | 0.6902 | 0.6876 |
| `zemke2023` | `mouse` | `lora+locon` | 6 | `zemke2023_rna` | 0.2486 | 0.2590 |
| `zemke2024` | `human` | `lora+locon` | 6 | `zemke2024_all_atac` | 0.7355 | 0.7499 |
| `zemke2024` | `human` | `lora+locon` | 6 | `zemke2024_all_rna` | 0.3539 | 0.2630 |

| Strategy | Native sources | Heads | Mean validation R | Mean test R | ATAC validation R | ATAC test R | RNA validation R | RNA test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora` | 10 | 20 | 0.5054 | 0.5246 | 0.6177 | 0.6498 | 0.3932 | 0.3993 |
| `lora+locon` | 10 | 20 | 0.4868 | 0.5124 | 0.6116 | 0.6467 | 0.3620 | 0.3781 |
