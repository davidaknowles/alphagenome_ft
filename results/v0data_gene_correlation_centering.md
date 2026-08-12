# Gene-correlation centering audit

Gene targets are counts per million, CPM. `Row/global cosine` compares targets centered only across cell groups within each gene with targets also centered across all training genes. `Local/global cosine` compares minibatch double centering with split-wide double centering after randomizing genomic windows. A low local value indicates noise from estimating cell-group means using the few genes in one sequence batch.

| Dataset | Groups | Training genes | Median genes/batch | Row/global cosine | Global DC variance retained | Median local/global cosine | Local/global p10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HDA | 134 | 44,840 | 24 | 0.99975 | 0.99950 | 0.94781 | 0.55230 |
| Liu | 186 | 25,367 | 18 | 0.99986 | 0.99973 | 0.96727 | 0.93162 |
| Johansen-human | 47 | 7,512 | 13 | 0.99312 | 0.98629 | 0.95906 | 0.91830 |
| Johansen-macaque | 47 | 7,642 | 13 | 0.99691 | 0.99384 | 0.95849 | 0.92085 |
| Johansen-marmoset | 47 | 7,574 | 13 | 0.99492 | 0.98987 | 0.95859 | 0.92043 |

For CPM targets, total expression is nearly equal across cell-group tracks. Row-centering therefore closely approximates split-wide double centering without estimating a noisy cell-group mean from each small minibatch. This is an objective-alignment diagnostic, not a prediction result.
