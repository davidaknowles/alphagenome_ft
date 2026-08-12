# Gene-correlation centering audit

Gene targets are counts per million, CPM. `Row/global cosine` compares targets centered only across cell groups within each gene with targets also centered across all training genes. `Local/global cosine` compares minibatch double centering with split-wide double centering after randomizing genomic windows. A low local value indicates noise from estimating cell-group means using the few genes in one sequence batch.

| Dataset | Groups | Training genes | Gene-bearing windows | Random median genes | Balanced median genes | Random empty | Balanced empty | Random no local DC variance | Balanced no local DC variance | Row/global cosine | Global DC variance retained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HDA | 134 | 44,840 | 68.7% | 16 | 17 | 0.0% | 0.0% | 0.1% | 0.2% | 0.99975 | 0.99950 |
| Liu | 186 | 25,367 | 49.4% | 9 | 9 | 0.4% | 0.0% | 1.4% | 0.0% | 0.99986 | 0.99973 |
| Johansen-human | 47 | 7,512 | 21.1% | 1 | 1 | 38.9% | 15.8% | 26.0% | 53.0% | 0.99312 | 0.98629 |
| Johansen-macaque | 47 | 7,642 | 23.1% | 1 | 1 | 34.9% | 7.6% | 26.7% | 58.8% | 0.99691 | 0.99384 |
| Johansen-marmoset | 47 | 7,574 | 22.9% | 1 | 1 | 35.0% | 8.3% | 26.9% | 58.1% | 0.99492 | 0.98987 |
| Zemke2023-published-human | 20 | 15,033 | 33.8% | 5 | 5 | 3.6% | 0.0% | 8.4% | 0.3% | 0.99993 | 0.99986 |

For CPM targets, total expression is nearly equal across cell-group tracks. Row-centering therefore closely approximates split-wide double centering without estimating a noisy cell-group mean from each small minibatch. Gene-balanced ordering is intended only for the row-centered objective. In sparse Johansen data it reduces empty batches but increases batches containing only one gene, which have no local double-centered variance. This is an objective-alignment diagnostic, not a prediction result.
