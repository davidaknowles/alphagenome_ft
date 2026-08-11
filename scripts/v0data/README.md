# v0data fine-tuning experiments

This directory contains dataset-specific preparation and experiment launchers for comparing low-rank adaptation, LoRA, with LoRA plus low-rank convolution adaptation, LoCon. All comparisons use the same 131,072 bp chromosome holdouts, initialization, optimizer settings, and signed double-centered Pearson correlation for model selection.

`prepare_study_targets.py` creates deterministic manifests from published BigWigs. RNA heads receive the exact base-weighted mean over finite positive values for each track because the AlphaGenome RNA output transform uses this quantity for channel scaling. ATAC tracks remain in their native published units because earlier scalar and quantile normalization screens did not improve Allen performance.

The initial published-track comparison includes the Mannens fetal-brain ATAC panel, the human Zemke 2023 motor-cortex ATAC/RNA panel, and the all-age Zemke 2024 hippocampus ATAC/RNA panel. The all-age Zemke panel avoids mixing biological age effects into the first adapter comparison. Age-stratified channels will be evaluated after the shared cell-type baseline.

Johansen targets use the existing fragment-derived ATAC and exon-plus-gene RNA preparation. Liu HDMA requires a separate reconstruction from cell-assigned fragments and expression matrices because its released observed BigWigs are significance scores rather than normalized accessibility coverage. Those reconstructions are not replaced by model predictions or attribution tracks.
