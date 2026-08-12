# Zemke 2024 RNA reprocessing

This workflow aggregates raw unique molecular identifier, UMI, counts from the released donor-level 10x Genomics matrices using the filtered-cell metadata from GSE278576. Counts are summed across donors by the 18 released broad subclasses and normalized to counts per million, CPM. Per-donor aggregate molecule totals must exactly equal the metadata `nCount_RNA` totals.

The published all-age target has 22 RNA channels. The released cell metadata identifies 18 broad subclasses but does not provide cell assignments for `Astro1_all`, `Astro2_all`, `Micro1_all`, or `Micro2_all`. Direct gene losses and metrics mask these four channels rather than treating missing labels as zero or inferring them from unrelated clustering columns. Their published coordinate tracks remain active.

`prepare_gene_supervision.py` builds the direct targets and a modified target manifest. `audit_gene_track_agreement.py` integrates the corresponding published reads-per-kilobase-per-million, RPKM, tracks over sampled union exons, converts the integral to CPM scale, and gates subsequent training on signed double-centered agreement over the 18 valid groups.

`smoke_donor_aggregation.py` validates one donor independently. It requires exact recovery of retained cell counts and raw RNA molecule totals from the release metadata before the longer all-donor build is trusted.

After the audit passes, `../submit_zemke2024_direct_gene_screen.sh` can submit the matched one-epoch LoRA and LoRA plus LoCon screen. The launcher is intentionally dormant until the agreement artifact exists and validates all expected supported and masked groups.
