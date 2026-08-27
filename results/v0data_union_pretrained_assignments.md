# Joint pretrained-head assignment audit

This audit reports the exact `neural_accessibility_bootstrap` initialization map. The baseline variant deterministically shuffles within each eligible assay, strand, and neural-status pool without matching target labels to source biosamples.

| Dataset | Native source | Head | Source assay | Targets | Neural targets | Unique sources | Maximum reuse |
|---|---|---|---|---:|---:|---:|---:|
| `hda` | `human` | `hda_atac` | `dnase` | 134 | 98 | 52 | 7 |
| `hda` | `human` | `hda_rna` | `rna_seq` | 134 | 98 | 46 | 8 |
| `liu_hdma` | `human` | `liu_atac` | `dnase` | 186 | 18 | 178 | 2 |
| `liu_hdma` | `human` | `liu_rna` | `rna_seq` | 372 | 36 | 352 | 3 |
| `johansen2025` | `human` | `allen_atac` | `dnase` | 47 | 41 | 24 | 3 |
| `johansen2025` | `human` | `allen_rna` | `rna_seq` | 94 | 82 | 36 | 5 |
| `johansen2025` | `macaque` | `allen_atac` | `dnase` | 47 | 41 | 24 | 3 |
| `johansen2025` | `macaque` | `allen_rna` | `rna_seq` | 94 | 82 | 36 | 5 |
| `johansen2025` | `marmoset` | `allen_atac` | `dnase` | 47 | 41 | 24 | 3 |
| `johansen2025` | `marmoset` | `allen_rna` | `rna_seq` | 94 | 82 | 36 | 5 |
| `zemke2023` | `human` | `zemke2023_atac` | `dnase` | 20 | 17 | 20 | 1 |
| `zemke2023` | `human` | `zemke2023_rna` | `rna_seq` | 20 | 17 | 18 | 2 |
| `zemke2023` | `macaque` | `zemke2023_atac` | `dnase` | 20 | 17 | 20 | 1 |
| `zemke2023` | `macaque` | `zemke2023_rna` | `rna_seq` | 20 | 17 | 18 | 2 |
| `zemke2023` | `marmoset` | `zemke2023_atac` | `dnase` | 20 | 17 | 20 | 1 |
| `zemke2023` | `marmoset` | `zemke2023_rna` | `rna_seq` | 20 | 17 | 18 | 2 |
| `zemke2023` | `mouse` | `zemke2023_atac` | `dnase` | 20 | 17 | 12 | 2 |
| `zemke2023` | `mouse` | `zemke2023_rna` | `rna_seq` | 20 | 17 | 7 | 4 |
| `zemke2024` | `human` | `zemke2024_all_atac` | `dnase` | 22 | 15 | 22 | 1 |
| `zemke2024` | `human` | `zemke2024_all_rna` | `rna_seq` | 22 | 15 | 22 | 1 |
