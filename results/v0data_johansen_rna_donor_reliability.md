# Johansen RNA donor reliability

Raw unique molecular identifier counts are aggregated separately by donor and retained cell group. Donors are assigned within each group to library-depth-balanced halves, and each half is normalized to counts per million, CPM. Full reliability uses the Spearman-Brown correction. The model correlation ceiling is the square root of reliability under a classical independent measurement-error assumption; it is not an observed model result.

| Species | Donors | Estimable groups | Genes | Split-half raw CPM R | Full reliability | Estimated model R ceiling | Split-half log1p CPM R |
|---|---:|---:|---:|---:|---:|---:|---:|
| human | 10 | 47 / 47 | 13509 | 0.9749 | 0.9873 | 0.9936 | 0.9331 |
| macaque | 8 | 47 / 47 | 13495 | 0.9714 | 0.9855 | 0.9927 | 0.8865 |
| marmoset | 4 | 47 / 47 | 13506 | 0.9661 | 0.9828 | 0.9914 | 0.8409 |
