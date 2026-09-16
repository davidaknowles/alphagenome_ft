# Joint native-source evaluation

Each row evaluates both modality heads from the same union model on one native source. R is signed double-centered Pearson correlation.

| Dataset | Source | Strategy | Epoch | ATAC valid R | ATAC test R | RNA valid R | RNA test R |
|---|---|---|---:|---:|---:|---:|---:|
| `hda` | `human` | `lora` | 17 | 0.8001 | 0.8143 | 0.5277 | 0.6613 |
| `liu_hdma` | `human` | `lora` | 17 | 0.7729 | 0.7914 | 0.6898 | 0.5099 |
| `johansen2025` | `human` | `lora` | 17 | 0.5644 | 0.6030 | 0.5296 | 0.5933 |
| `johansen2025` | `macaque` | `lora` | 17 | 0.5761 | 0.4338 | 0.3696 | 0.5166 |
| `johansen2025` | `marmoset` | `lora` | 17 | 0.4736 | 0.5062 | 0.2323 | 0.4338 |
| `zemke2023` | `human` | `lora` | 17 | 0.6877 | 0.7149 | 0.5331 | 0.6342 |
| `zemke2023` | `macaque` | `lora` | 17 | 0.3879 | 0.6259 | 0.5239 | 0.5385 |
| `zemke2023` | `marmoset` | `lora` | 17 | 0.6169 | 0.5878 | 0.4787 | 0.5738 |
| `zemke2023` | `mouse` | `lora` | 17 | 0.7183 | 0.7176 | 0.4854 | 0.1506 |
| `zemke2024` | `human` | `lora` | 17 | 0.7462 | 0.7588 | 0.4992 | 0.5121 |
| `hda` | `human` | `lora+locon` | 25 | 0.8113 | 0.8240 | 0.5421 | 0.6747 |
| `liu_hdma` | `human` | `lora+locon` | 25 | 0.7719 | 0.7914 | 0.7076 | 0.4841 |
| `johansen2025` | `human` | `lora+locon` | 25 | 0.5792 | 0.6166 | 0.5762 | 0.5870 |
| `johansen2025` | `macaque` | `lora+locon` | 25 | 0.5909 | 0.4343 | 0.3391 | 0.5302 |
| `johansen2025` | `marmoset` | `lora+locon` | 25 | 0.5106 | 0.5460 | 0.2531 | 0.4648 |
| `zemke2023` | `human` | `lora+locon` | 25 | 0.7047 | 0.7292 | 0.6483 | 0.6269 |
| `zemke2023` | `macaque` | `lora+locon` | 25 | 0.4083 | 0.6479 | 0.5132 | 0.5962 |
| `zemke2023` | `marmoset` | `lora+locon` | 25 | 0.6358 | 0.6056 | 0.4930 | 0.6044 |
| `zemke2023` | `mouse` | `lora+locon` | 25 | 0.7335 | 0.7331 | 0.5119 | 0.2130 |
| `zemke2024` | `human` | `lora+locon` | 25 | 0.7548 | 0.7652 | 0.5883 | 0.5026 |

| Strategy | Native sources | Heads | Mean validation R | Mean test R | ATAC validation R | ATAC test R | RNA validation R | RNA test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora` | 10 | 20 | 0.5607 | 0.5839 | 0.6344 | 0.6554 | 0.4869 | 0.5124 |
| `lora+locon` | 10 | 20 | 0.5837 | 0.5989 | 0.6501 | 0.6693 | 0.5173 | 0.5284 |
