# Zemke 2023 RNA donor reliability

Raw unique molecular identifier counts are aggregated separately by donor and retained cell group. Donors are assigned within each group to library-depth-balanced halves, and each half is normalized to counts per million, CPM. Full reliability uses the Spearman-Brown correction. The model correlation ceiling is the square root of reliability under a classical independent measurement-error assumption; it is not an observed model result.

| Species | Donors | Estimable groups | Genes | Split-half raw CPM R | Full reliability | Estimated model R ceiling | Split-half log1p CPM R |
|---|---:|---:|---:|---:|---:|---:|---:|
| human | 7 | 19 / 19 | 23264 | 0.9211 | 0.9589 | 0.9793 | 0.8425 |
| macaque | 4 | 19 / 19 | 15841 | 0.9375 | 0.9678 | 0.9837 | 0.7627 |
| marmoset | 4 | 19 / 19 | 22046 | 0.9774 | 0.9886 | 0.9943 | 0.8238 |
| mouse | 8 | 19 / 19 | 26291 | 0.9708 | 0.9852 | 0.9926 | 0.8030 |
