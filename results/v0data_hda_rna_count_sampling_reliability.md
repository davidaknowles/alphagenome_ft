# Mannens RNA technical sampling reliability

Each published cluster-level raw-count pseudobulk was divided into two equal-probability molecule samples, independently for every gene. Each half was normalized to counts per million, CPM, and compared using signed double-centered correlation. Repeated binomial splits measure technical molecule-sampling repeatability only. The released matrix has no donor-level counts, so this audit does not measure biological donor or specimen variability.

| Quantity | Value |
|---|---:|
| Modeled cell groups | 134 |
| Modeled genes | 59,310 |
| Raw molecules | 1,224,754,432 |
| Repeated splits | 20 |
| Raw CPM split-half double-centered R, mean | 0.9993 |
| Raw CPM split-half double-centered R, standard deviation | 0.0000 |
| Raw CPM full technical reliability, Spearman-Brown | 0.9997 |
| Raw CPM assumption-based model R ceiling | 0.9998 |
| log1p CPM split-half double-centered R, mean | 0.8498 |
| log1p CPM split-half double-centered R, standard deviation | 0.0002 |
| log1p CPM full technical reliability, Spearman-Brown | 0.9188 |
