# Zemke 2023 RNA target agreement

Published reads-per-kilobase-per-million, RPKM, tracks are integrated over sampled union exons and divided by 1,000 to recover counts-per-million scale. Correlation is computed after centering across genes and cell subclasses.

| Species | Sampled genes | Groups | Raw CPM R | log1p R | Raw within-gene R | log1p within-gene R | Raw nonzero | Integrated nonzero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| human | 512 | 19 | 0.5882 | 0.8462 | 0.7706 | 0.7752 | 0.7816 | 0.7391 |
| macaque | 512 | 19 | 0.3777 | 0.8347 | 0.7167 | 0.7121 | 0.8800 | 0.8792 |
| marmoset | 512 | 19 | 0.4814 | 0.8507 | 0.7326 | 0.7305 | 0.8191 | 0.8413 |
| mouse | 512 | 19 | 0.4377 | 0.8705 | 0.7313 | 0.7281 | 0.6986 | 0.6740 |
