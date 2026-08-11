# Liu HDMA target reconstruction

The Liu Human Development Multiomic Atlas, HDMA, targets are reconstructed from measured cell-level data. The released observed accessibility tracks are Model-based Analysis of ChIP-Seq version 2 significance scores, while the other released tracks are model predictions and attributions. None of those transformed tracks is used as a training target.

The paired panel contains the 186 clusters retained by the source study for its published ChromBPNet analysis. The remaining 17 annotated clusters are recorded in the audit manifest and excluded from both modalities. This gives ATAC and RNA heads identical cell-cluster support.

ATAC targets retain all fragments from metadata-selected cells, without downsampling. Fragment-covered bases are averaged in 100 bp bins and divided by the cluster's complete fragment count in millions. This follows signal per million reads, SPMR, normalization while treating each paired-end fragment as the library unit. RNA unique molecular identifier, UMI, counts are summed by cluster, normalized to counts per million, CPM, and represented as strand-specific union-exon density with direct gene-level supervision.

The workflow is split into auditable stages, cell and cluster validation, per-sample RNA aggregation, RNA reduction, whole-genome ATAC library-size calculation, chromosome-sharded ATAC aggregation, BigWig materialization, and joint target-manifest generation.
