# Johansen RNA reprocessing

The legacy group pseudobulks sum per-cell normalized expression and then renormalize each group. This directory rebuilds targets by summing raw unique molecular identifier, UMI, counts for all cells in each group before counts-per-million, CPM, normalization.

`aggregate_corrected_pseudobulk.py` reads only the raw sparse matrix, group labels, and gene metadata from a single-cell H5AD file. A sparse cells-by-groups assignment matrix aggregates the cells in one matrix multiplication. `replace_gene_supervision.py` aligns corrected expression to an existing genomic supervision artifact, preserving gene coordinates, strands, and union-exon annotations. Expression is renormalized over the retained modeled genes, matching the existing joint-target convention. `rewrite_species_config.py` selects these corrected artifacts, disables derived exon-coverage loss, and enables signed double-centered correlation on direct gene expression.

`audit_normalization.py` compares legacy and corrected matrices in raw and log1p CPM space. The checked-in audit shows that the correction materially changes cell-group-specific raw expression and is not equivalent to scalar rescaling.
