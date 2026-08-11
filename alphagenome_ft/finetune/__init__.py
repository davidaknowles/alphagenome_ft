"""Finetuning utilities for AlphaGenome."""

from alphagenome_ft.finetune.config import (
    TrackInfo,
    HeadSpec,
    load_targets_config,
    prepare_head_specs,
    validate_head_specs,
)
from alphagenome_ft.finetune.data import (
    get_fold_split,
    BigWigDataModule,
    MultiSpeciesDataModule,
    WindowedTargetCache,
    build_interval,
    load_intervals_from_bed,
    load_intervals_from_dataframe,
    prepare_intervals_from_fold,
    prepare_intervals_from_split,
    build_fasta_index,
    prepare_batch,
)
from alphagenome_ft.finetune.train import (
    register_predefined_heads,
    create_optimizer,
    train,
)
from alphagenome_ft.finetune.metrics import (
    select_prediction_for_targets,
    r2_metrics,
)
from alphagenome_ft.finetune.target_transforms import (
    PiecewiseLinearTargetTransform,
    SpatialRebinTargetTransform,
    load_target_transform,
)
from alphagenome_ft.finetune.reprocessing import (
    BinnedAtacAccumulator,
    fragment_totals_by_group,
    read_cell_groups,
)
from alphagenome_ft.finetune.rna_tracks import (
    GeneBody,
    GeneExons,
    PseudobulkExpression,
    read_gene_bodies,
    read_gene_exons,
    read_pseudobulk_expression,
    remap_expression_gene_ids,
    write_stranded_gene_body_bigwigs,
    write_gene_expression_supervision,
    write_stranded_exon_bigwigs,
)
from alphagenome_ft.finetune.backends import (
    BackendName,
    PreparedRun,
    TorchBackendConfig,
    TorchSubprocessBackend,
)

__all__ = [
    # config
    "TrackInfo",
    "HeadSpec",
    "load_targets_config",
    "prepare_head_specs",
    "validate_head_specs",
    # data
    "get_fold_split",
    "BigWigDataModule",
    "MultiSpeciesDataModule",
    "WindowedTargetCache",
    "build_interval",
    "load_intervals_from_bed",
    "load_intervals_from_dataframe",
    "prepare_intervals_from_fold",
    "prepare_intervals_from_split",
    "build_fasta_index",
    "prepare_batch",
    # train
    "register_predefined_heads",
    "create_optimizer",
    "train",
    # metrics
    "select_prediction_for_targets",
    "r2_metrics",
    # target transforms
    "PiecewiseLinearTargetTransform",
    "SpatialRebinTargetTransform",
    "load_target_transform",
    # ATAC pseudobulk tracks
    "BinnedAtacAccumulator",
    "fragment_totals_by_group",
    "read_cell_groups",
    # RNA pseudobulk tracks
    "GeneBody",
    "GeneExons",
    "PseudobulkExpression",
    "read_gene_bodies",
    "read_gene_exons",
    "read_pseudobulk_expression",
    "remap_expression_gene_ids",
    "write_stranded_gene_body_bigwigs",
    "write_gene_expression_supervision",
    "write_stranded_exon_bigwigs",
    # backends
    "BackendName",
    "PreparedRun",
    "TorchBackendConfig",
    "TorchSubprocessBackend",
]
