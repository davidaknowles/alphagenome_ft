# Johansen RNA normalization audit

Legacy pseudobulks summed expression after per-cell normalization. Corrected pseudobulks sum raw unique molecular identifier, UMI, counts within each cell group and then normalize the group total to counts per million, CPM. Agreement is measured after putting both matrices in CPM units.

| Species | Groups | Genes | Raw CPM double-centered R | log1p CPM double-centered R | Median per-group R |
|---|---:|---:|---:|---:|---:|
| human | 60 | 36601 | 0.4645 | 0.9224 | 0.4942 |
| macaque | 58 | 35219 | 0.5023 | 0.9418 | 0.5714 |
| marmoset | 56 | 35787 | 0.5430 | 0.9537 | 0.5968 |

The raw-CPM discrepancy changes the cell-group-specific target structure, not only its scale. Johansen RNA training results produced from the legacy matrices are therefore superseded by raw-count-derived targets.
