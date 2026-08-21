# Joint native-source evaluation

Each row evaluates one head from the same union model on one native source. R is signed double-centered Pearson correlation.

| Dataset | Source | Strategy | Epoch | Head | Validation R | Test R |
|---|---|---|---:|---|---:|---:|
| `hda` | `human` | `LoRA, lr 3e-4` | 9 | `hda_atac` | 0.7913 | 0.8026 |
| `hda` | `human` | `LoRA, lr 3e-4` | 9 | `hda_rna` | 0.4679 | 0.6240 |
| `liu_hdma` | `human` | `LoRA, lr 3e-4` | 9 | `liu_atac` | 0.7644 | 0.7837 |
| `liu_hdma` | `human` | `LoRA, lr 3e-4` | 9 | `liu_rna` | 0.7755 | 0.4845 |
| `johansen2025` | `human` | `LoRA, lr 3e-4` | 9 | `allen_atac` | 0.5707 | 0.6040 |
| `johansen2025` | `human` | `LoRA, lr 3e-4` | 9 | `allen_rna` | 0.5393 | 0.5219 |
| `johansen2025` | `macaque` | `LoRA, lr 3e-4` | 9 | `allen_atac` | 0.5791 | 0.4268 |
| `johansen2025` | `macaque` | `LoRA, lr 3e-4` | 9 | `allen_rna` | 0.3301 | 0.4835 |
| `johansen2025` | `marmoset` | `LoRA, lr 3e-4` | 9 | `allen_atac` | 0.5158 | 0.5524 |
| `johansen2025` | `marmoset` | `LoRA, lr 3e-4` | 9 | `allen_rna` | 0.2290 | 0.4305 |
| `zemke2023` | `human` | `LoRA, lr 3e-4` | 9 | `zemke2023_atac` | 0.6853 | 0.7081 |
| `zemke2023` | `human` | `LoRA, lr 3e-4` | 9 | `zemke2023_rna` | 0.3557 | 0.3782 |
| `zemke2023` | `macaque` | `LoRA, lr 3e-4` | 9 | `zemke2023_atac` | 0.2311 | 0.6064 |
| `zemke2023` | `macaque` | `LoRA, lr 3e-4` | 9 | `zemke2023_rna` | 0.2550 | 0.3636 |
| `zemke2023` | `marmoset` | `LoRA, lr 3e-4` | 9 | `zemke2023_atac` | 0.6057 | 0.5723 |
| `zemke2023` | `marmoset` | `LoRA, lr 3e-4` | 9 | `zemke2023_rna` | 0.2929 | 0.1424 |
| `zemke2023` | `mouse` | `LoRA, lr 3e-4` | 9 | `zemke2023_atac` | 0.6944 | 0.6937 |
| `zemke2023` | `mouse` | `LoRA, lr 3e-4` | 9 | `zemke2023_rna` | 0.2868 | 0.2800 |
| `zemke2024` | `human` | `LoRA, lr 3e-4` | 9 | `zemke2024_all_atac` | 0.7388 | 0.7485 |
| `zemke2024` | `human` | `LoRA, lr 3e-4` | 9 | `zemke2024_all_rna` | 0.3997 | 0.2848 |
| `hda` | `human` | `LoRA+LoCon, lr 1e-4` | 23 | `hda_atac` | 0.8084 | 0.8192 |
| `hda` | `human` | `LoRA+LoCon, lr 1e-4` | 23 | `hda_rna` | 0.4655 | 0.6553 |
| `liu_hdma` | `human` | `LoRA+LoCon, lr 1e-4` | 23 | `liu_atac` | 0.7686 | 0.7883 |
| `liu_hdma` | `human` | `LoRA+LoCon, lr 1e-4` | 23 | `liu_rna` | 0.7474 | 0.4831 |
| `johansen2025` | `human` | `LoRA+LoCon, lr 1e-4` | 23 | `allen_atac` | 0.5952 | 0.6297 |
| `johansen2025` | `human` | `LoRA+LoCon, lr 1e-4` | 23 | `allen_rna` | 0.6046 | 0.5418 |
| `johansen2025` | `macaque` | `LoRA+LoCon, lr 1e-4` | 23 | `allen_atac` | 0.6065 | 0.4503 |
| `johansen2025` | `macaque` | `LoRA+LoCon, lr 1e-4` | 23 | `allen_rna` | 0.3711 | 0.5155 |
| `johansen2025` | `marmoset` | `LoRA+LoCon, lr 1e-4` | 23 | `allen_atac` | 0.5444 | 0.5799 |
| `johansen2025` | `marmoset` | `LoRA+LoCon, lr 1e-4` | 23 | `allen_rna` | 0.2522 | 0.4278 |
| `zemke2023` | `human` | `LoRA+LoCon, lr 1e-4` | 23 | `zemke2023_atac` | 0.6982 | 0.7199 |
| `zemke2023` | `human` | `LoRA+LoCon, lr 1e-4` | 23 | `zemke2023_rna` | 0.3914 | 0.3957 |
| `zemke2023` | `macaque` | `LoRA+LoCon, lr 1e-4` | 23 | `zemke2023_atac` | 0.3414 | 0.6355 |
| `zemke2023` | `macaque` | `LoRA+LoCon, lr 1e-4` | 23 | `zemke2023_rna` | 0.2889 | 0.3705 |
| `zemke2023` | `marmoset` | `LoRA+LoCon, lr 1e-4` | 23 | `zemke2023_atac` | 0.6248 | 0.5896 |
| `zemke2023` | `marmoset` | `LoRA+LoCon, lr 1e-4` | 23 | `zemke2023_rna` | 0.3249 | 0.1698 |
| `zemke2023` | `mouse` | `LoRA+LoCon, lr 1e-4` | 23 | `zemke2023_atac` | 0.7145 | 0.7131 |
| `zemke2023` | `mouse` | `LoRA+LoCon, lr 1e-4` | 23 | `zemke2023_rna` | 0.3136 | 0.3192 |
| `zemke2024` | `human` | `LoRA+LoCon, lr 1e-4` | 23 | `zemke2024_all_atac` | 0.7492 | 0.7596 |
| `zemke2024` | `human` | `LoRA+LoCon, lr 1e-4` | 23 | `zemke2024_all_rna` | 0.4092 | 0.2792 |
| `hda` | `human` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `hda_atac` | 0.8065 | 0.8184 |
| `hda` | `human` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `hda_rna` | 0.4829 | 0.6529 |
| `liu_hdma` | `human` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `liu_atac` | 0.7674 | 0.7878 |
| `liu_hdma` | `human` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `liu_rna` | 0.7155 | 0.4977 |
| `johansen2025` | `human` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `allen_atac` | 0.5933 | 0.6292 |
| `johansen2025` | `human` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `allen_rna` | 0.5950 | 0.5571 |
| `johansen2025` | `macaque` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `allen_atac` | 0.6045 | 0.4475 |
| `johansen2025` | `macaque` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `allen_rna` | 0.3712 | 0.5146 |
| `johansen2025` | `marmoset` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `allen_atac` | 0.5418 | 0.5771 |
| `johansen2025` | `marmoset` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `allen_rna` | 0.2456 | 0.4225 |
| `zemke2023` | `human` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `zemke2023_atac` | 0.6963 | 0.7196 |
| `zemke2023` | `human` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `zemke2023_rna` | 0.3927 | 0.4006 |
| `zemke2023` | `macaque` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `zemke2023_atac` | 0.3047 | 0.6298 |
| `zemke2023` | `macaque` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `zemke2023_rna` | 0.2847 | 0.3775 |
| `zemke2023` | `marmoset` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `zemke2023_atac` | 0.6227 | 0.5874 |
| `zemke2023` | `marmoset` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `zemke2023_rna` | 0.3278 | 0.1524 |
| `zemke2023` | `mouse` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `zemke2023_atac` | 0.7132 | 0.7118 |
| `zemke2023` | `mouse` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `zemke2023_rna` | 0.3107 | 0.3130 |
| `zemke2024` | `human` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `zemke2024_all_atac` | 0.7477 | 0.7612 |
| `zemke2024` | `human` | `LoRA+LoCon, lr 1e-4, RNA weight 2` | 22 | `zemke2024_all_rna` | 0.4207 | 0.2859 |

| Strategy | Native sources | Heads | Mean validation R | Mean test R | ATAC validation R | ATAC test R | RNA validation R | RNA test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `LoRA, lr 3e-4` | 10 | 20 | 0.5054 | 0.5246 | 0.6177 | 0.6498 | 0.3932 | 0.3993 |
| `LoRA+LoCon, lr 1e-4` | 10 | 20 | 0.5310 | 0.5422 | 0.6451 | 0.6685 | 0.4169 | 0.4158 |
| `LoRA+LoCon, lr 1e-4, RNA weight 2` | 10 | 20 | 0.5272 | 0.5422 | 0.6398 | 0.6670 | 0.4147 | 0.4174 |
