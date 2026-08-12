# Pretrained neural-head source audit

The neural bootstrap uses a neural source pool only when every target strand has at least two eligible pretrained channels. Otherwise it retains the complete assay-wide pool.

| Organism | Assay | Valid channels | Neural channels | Valid by strand | Neural by strand |
|---|---|---:|---:|---|---|
| `HOMO_SAPIENS` | `atac` | 167 | 1 | .=167 | .=1 |
| `HOMO_SAPIENS` | `dnase` | 305 | 18 | .=305 | .=18 |
| `HOMO_SAPIENS` | `rna_seq` | 667 | 41 | +=271, -=271, .=125 | +=13, -=13, .=15 |
| `MUS_MUSCULUS` | `atac` | 18 | 4 | .=18 | .=4 |
| `MUS_MUSCULUS` | `dnase` | 67 | 10 | .=67 | .=10 |
| `MUS_MUSCULUS` | `rna_seq` | 173 | 22 | +=71, -=71, .=31 | +=8, -=8, .=6 |
