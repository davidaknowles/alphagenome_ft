# Joint native-source evaluation

Each row evaluates both modality heads from the same union model on one native source. R is signed double-centered Pearson correlation.

| Dataset | Source | Strategy | Epoch | ATAC valid R | ATAC test R | RNA valid R | RNA test R |
|---|---|---|---:|---:|---:|---:|---:|
| `hda` | `human` | `lora` | 37 | 0.8048 | 0.8156 | 0.5519 | 0.6419 |
| `liu_hdma` | `human` | `lora` | 37 | 0.7723 | 0.7915 | 0.6761 | 0.5139 |
| `johansen2025` | `human` | `lora` | 37 | 0.5765 | 0.6111 | 0.5179 | 0.5942 |
| `johansen2025` | `macaque` | `lora` | 37 | 0.5824 | 0.4405 | 0.3334 | 0.5621 |
| `johansen2025` | `marmoset` | `lora` | 37 | 0.4943 | 0.5296 | 0.2309 | 0.4169 |
| `zemke2023` | `human` | `lora` | 37 | 0.6950 | 0.7206 | 0.5963 | 0.5126 |
| `zemke2023` | `macaque` | `lora` | 37 | 0.4687 | 0.6395 | 0.4846 | 0.6128 |
| `zemke2023` | `marmoset` | `lora` | 37 | 0.6281 | 0.5958 | 0.4753 | 0.6483 |
| `zemke2023` | `mouse` | `lora` | 37 | 0.7250 | 0.7241 | 0.5573 | 0.4879 |
| `zemke2024` | `human` | `lora` | 37 | 0.7453 | 0.7576 | 0.6014 | 0.3507 |
| `hda` | `human` | `lora+locon` | 37 | 0.8104 | 0.8219 | 0.5707 | 0.6593 |
| `liu_hdma` | `human` | `lora+locon` | 37 | 0.7716 | 0.7901 | 0.6526 | 0.5006 |
| `johansen2025` | `human` | `lora+locon` | 37 | 0.5831 | 0.6183 | 0.6239 | 0.5913 |
| `johansen2025` | `macaque` | `lora+locon` | 37 | 0.5889 | 0.4404 | 0.3423 | 0.5806 |
| `johansen2025` | `marmoset` | `lora+locon` | 37 | 0.5010 | 0.5371 | 0.2477 | 0.4435 |
| `zemke2023` | `human` | `lora+locon` | 37 | 0.7022 | 0.7276 | 0.6770 | 0.5354 |
| `zemke2023` | `macaque` | `lora+locon` | 37 | 0.4390 | 0.6486 | 0.5155 | 0.6383 |
| `zemke2023` | `marmoset` | `lora+locon` | 37 | 0.6351 | 0.6013 | 0.5213 | 0.6188 |
| `zemke2023` | `mouse` | `lora+locon` | 37 | 0.7332 | 0.7329 | 0.5645 | 0.5313 |
| `zemke2024` | `human` | `lora+locon` | 37 | 0.7506 | 0.7635 | 0.6973 | 0.3402 |

| Strategy | Native sources | Heads | Mean validation R | Mean test R | ATAC validation R | ATAC test R | RNA validation R | RNA test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora` | 10 | 20 | 0.5759 | 0.5984 | 0.6493 | 0.6626 | 0.5025 | 0.5341 |
| `lora+locon` | 10 | 20 | 0.5964 | 0.6060 | 0.6515 | 0.6682 | 0.5413 | 0.5439 |
