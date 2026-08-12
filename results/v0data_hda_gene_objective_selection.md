# HDA gene-objective selection

A nonzero correlation weight advances to LoRA plus LoCon only when its LoRA epoch-one mean validation signed double-centered Pearson correlation exceeds the gene-only weight-zero baseline.

| Weight | ATAC validation R | RNA validation R | Mean validation R |
|---:|---:|---:|---:|
| 0 | 0.7724 | 0.4232 | 0.5978 |
| 0.1 | 0.7710 | 0.4294 | 0.6002 |
| 1 | 0.7649 | 0.4253 | 0.5951 |
| 10 | 0.7486 | 0.3875 | 0.5680 |

Status, selected nonzero objective.
Improvement over weight zero, 0.0023.
Selected weight, 0.1.
