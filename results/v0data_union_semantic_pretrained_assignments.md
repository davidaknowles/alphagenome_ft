# Joint pretrained-head assignment audit

This audit reports the exact `semantic_neural_accessibility_bootstrap` initialization map. The semantic variant prefers matching anatomy and cell-class concepts while preserving assay, strand, and neural-status eligibility. Unmatched targets retain the deterministic shuffled assignment.

| Dataset | Native source | Head | Source assay | Targets | Neural targets | Unique sources | Maximum reuse |
|---|---|---|---|---:|---:|---:|---:|
| `hda` | `human` | `hda_atac` | `dnase` | 134 | 98 | 40 | 17 |
| `hda` | `human` | `hda_rna` | `rna_seq` | 134 | 98 | 41 | 21 |
| `liu_hdma` | `human` | `liu_atac` | `dnase` | 186 | 18 | 178 | 2 |
| `liu_hdma` | `human` | `liu_rna` | `rna_seq` | 372 | 36 | 352 | 3 |
| `johansen2025` | `human` | `allen_atac` | `dnase` | 47 | 41 | 18 | 11 |
| `johansen2025` | `human` | `allen_rna` | `rna_seq` | 94 | 82 | 34 | 12 |
| `johansen2025` | `macaque` | `allen_atac` | `dnase` | 47 | 41 | 18 | 11 |
| `johansen2025` | `macaque` | `allen_rna` | `rna_seq` | 94 | 82 | 34 | 12 |
| `johansen2025` | `marmoset` | `allen_atac` | `dnase` | 47 | 41 | 18 | 11 |
| `johansen2025` | `marmoset` | `allen_rna` | `rna_seq` | 94 | 82 | 34 | 12 |
| `zemke2023` | `human` | `zemke2023_atac` | `dnase` | 20 | 17 | 9 | 6 |
| `zemke2023` | `human` | `zemke2023_rna` | `rna_seq` | 20 | 17 | 9 | 8 |
| `zemke2023` | `macaque` | `zemke2023_atac` | `dnase` | 20 | 17 | 9 | 6 |
| `zemke2023` | `macaque` | `zemke2023_rna` | `rna_seq` | 20 | 17 | 9 | 8 |
| `zemke2023` | `marmoset` | `zemke2023_atac` | `dnase` | 20 | 17 | 9 | 6 |
| `zemke2023` | `marmoset` | `zemke2023_rna` | `rna_seq` | 20 | 17 | 9 | 8 |
| `zemke2023` | `mouse` | `zemke2023_atac` | `dnase` | 20 | 17 | 8 | 6 |
| `zemke2023` | `mouse` | `zemke2023_rna` | `rna_seq` | 20 | 17 | 5 | 14 |
| `zemke2024` | `human` | `zemke2024_all_atac` | `dnase` | 22 | 15 | 17 | 4 |
| `zemke2024` | `human` | `zemke2024_all_rna` | `rna_seq` | 22 | 15 | 15 | 4 |
