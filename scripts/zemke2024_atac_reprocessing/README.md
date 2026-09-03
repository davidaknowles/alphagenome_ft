# Zemke 2024 ATAC reprocessing

This directory reconstructs broad-subclass ATAC coverage targets from the indexed donor fragment resources. The metadata labels 18 broad subclasses, whereas the released coordinate tracks have 22 channels because Astro and microglia subclasses are split further. The reconstruction therefore starts as a separate 18-channel target family rather than silently substituting an incompatible signal for the released 22-channel head.

Each retained metadata cell contributes every recorded fragment and its multiplicity. Fragment coverage is accumulated in 100 bp bins after standard Tn5 end shifts, then normalized to signal per million fragments, SPMR, using each broad subclass's whole-genome fragment total. The one-donor chromosome 9 smoke run verifies barcode matching and target scale before an all-donor reaggregation and released-track comparison.

`audit_metadata_coverage.py` first measures retained-cell and retained-fragment overlap for every donor. It gates a full reaggregation: every donor must have complete metadata-cell coverage and the retained-fragment fraction must be reported rather than treated as an implicit downsampling rate.
