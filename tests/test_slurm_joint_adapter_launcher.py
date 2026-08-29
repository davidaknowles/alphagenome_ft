from pathlib import Path


LAUNCHERS = (
    "scripts/v0data/slurm_joint_adapter_comparison.sbatch",
    "scripts/v0data/slurm_johansen_joint_adapters.sbatch",
    "scripts/v0data/slurm_study_adapter_comparison.sbatch",
    "scripts/v0data/slurm_zemke2023_adapter_matrix.sbatch",
    "scripts/v0data/slurm_zemke2023_joint_adapters.sbatch",
)


def test_multi_gpu_drop_last_is_applied_after_smoke_arguments() -> None:
    script = Path("scripts/v0data/slurm_joint_adapter_comparison.sbatch").read_text()

    smoke_position = script.index('if [[ "${SMOKE:-0}" == "1" ]]')
    drop_last_position = script.index("if (( num_devices > 1 ))")

    assert drop_last_position > smoke_position


def test_all_study_lower_rate_continuation_snapshots_selected_checkpoint() -> None:
    launcher = Path("scripts/v0data/slurm_joint_multidataset_adapters.sbatch").read_text()
    submitter = Path(
        "scripts/v0data/submit_joint_locon_lower_rate_continuation.sh"
    ).read_text()

    assert 'extra_args+=(--resume-from "$RESUME_FROM")' in launcher
    assert "extra_args+=(--reset-optimizer)" in launcher
    assert 'source_epoch="${SOURCE_EPOCH:-6}"' in submitter
    assert 'cp -a "$source_checkpoint" "$snapshot"' in submitter
    assert "Refusing to replace mismatched metric history" in submitter
    assert "RESET_OPTIMIZER=1" in submitter
    assert '"source_epoch": int(sys.argv[2])' in submitter
    assert '"reset_optimizer": True' in submitter
    assert 'run_dir/continuation.json' in submitter
    assert 'learning_rate="${LEARNING_RATE:-3e-4}"' in submitter
    assert "LEARNING_RATE=${learning_rate}" in submitter
    assert "--array=1" in submitter


def test_all_study_launcher_preallocates_gpu_memory_and_aborts_failed_collectives() -> None:
    launcher = Path("scripts/v0data/slurm_joint_multidataset_adapters.sbatch").read_text()

    assert 'XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"' in launcher
    assert 'XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.97}"' in launcher
    assert 'XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE="${XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE:-1}"' in launcher


def test_all_study_launcher_allows_an_explicit_checkpoint_selection_metric() -> None:
    launcher = Path("scripts/v0data/slurm_joint_multidataset_adapters.sbatch").read_text()

    assert '--best-metric "${BEST_METRIC:-valid/mean/differential_pearson_r}"' in launcher
    assert '--best-metric-mode "${BEST_METRIC_MODE:-max}"' in launcher


def test_all_study_resumed_smoke_honors_epoch_horizon() -> None:
    launcher = Path("scripts/v0data/slurm_joint_multidataset_adapters.sbatch").read_text()

    assert '--num-epochs "${NUM_EPOCHS:-1}"' in launcher


def test_zemke_direct_gene_smoke_covers_early_mouse_genes() -> None:
    launcher = Path("scripts/v0data/slurm_joint_multidataset_adapters.sbatch").read_text()
    submitter = Path(
        "scripts/v0data/submit_joint_zemke_gene_warmup_then_adapters.sh"
    ).read_text()

    for split in ("TRAIN", "VALID", "TEST"):
        assert f'--limit-{split.lower()} "${{SMOKE_LIMIT_{split}:-8}}"' in launcher
        assert f"SMOKE_LIMIT_{split}=${{SMOKE_LIMIT_{split}:-40}}" in submitter
    assert 'warmup_max_epochs="${WARMUP_MAX_EPOCHS:-20}"' in submitter
    assert 'warmup_patience="${WARMUP_PATIENCE:-5}"' in submitter


def test_parallel_joint_continuations_are_smoke_gated_and_matched() -> None:
    script = Path("scripts/v0data/submit_joint_parallel_continuations.sh").read_text()
    library = Path("scripts/v0data/joint_continuation_lib.sh").read_text()

    assert "lora_epoch=9" in script
    assert "locon_epoch=17" in script
    assert "_lr3e4_reset" in script
    assert "_lr1e4_reset" in script
    assert "_lr1e4_rnaw2_reset" in script
    assert "prepare_joint_rna_objective_config.py" in script
    assert "--loss-weight 2" in script
    assert "joint_continuation_lib.sh" in script
    assert 'dependency="afterok:${smoke}_${task}"' in library
    assert "RESET_OPTIMIZER=1" in library


def test_metric_aligned_pair_uses_one_snapshot_and_matched_reset_control() -> None:
    script = Path("scripts/v0data/submit_joint_metric_aligned_pair.sh").read_text()

    assert "prepare_joint_metric_aligned_config.py" in script
    assert 'source_epoch="${SOURCE_EPOCH:-22}"' in script
    assert "snapshot_checkpoint" in script
    assert "_lr1e4_epoch22_reset_control" in script
    assert "_lr1e4_epoch22_metric_aligned_reset" in script
    assert script.count("submit_continuation 1") == 2


def test_joint_head_refit_freezes_adapters_and_screens_matched_rates() -> None:
    script = Path("scripts/v0data/submit_joint_head_refit_triplet.sh").read_text()
    launcher = Path("scripts/v0data/slurm_joint_multidataset_adapters.sbatch").read_text()
    library = Path("scripts/v0data/joint_continuation_lib.sh").read_text()
    entrypoint = Path("scripts/run_humanbraindev_finetune.py").read_text()

    assert "freeze_backbone_adapters=1" in script
    assert 'source_epoch="${SOURCE_EPOCH:-32}"' in script
    assert script.count("submit_continuation 1") == 3
    for rate in ("1e-4", "3e-4", "1e-3"):
        assert rate in script
    assert "FREEZE_BACKBONE_ADAPTERS=${freeze_backbone_adapters:-0}" in library
    assert '"freeze_backbone_adapters": bool(int(sys.argv[7]))' in library
    assert "extra_args+=(--freeze-backbone-adapters)" in launcher
    assert "args.backbone_lora and not args.freeze_backbone_adapters" in entrypoint


def test_joint_adapter_expansion_is_function_preserving_and_smoke_gated() -> None:
    script = Path("scripts/v0data/submit_joint_adapter_expansion_pair.sh").read_text()
    launcher = Path("scripts/v0data/slurm_joint_multidataset_adapters.sbatch").read_text()
    library = Path("scripts/v0data/joint_continuation_lib.sh").read_text()

    assert "expand_backbone_adapters=1" in script
    assert 'source_epoch="${SOURCE_EPOCH:-32}"' in script
    assert script.count("submit_continuation 1") == 2
    assert "downres_block_2;downres_block_3;downres_block_4;downres_block_5" in script
    assert "lora_rank=32" in script
    assert "locon_rank=8" in script
    assert "EXPAND_BACKBONE_ADAPTERS=${expand_backbone_adapters:-0}" in library
    assert 'dependency="afterok:${smoke}_${task}"' in library
    assert "extra_args+=(--expand-backbone-adapters)" in launcher
    assert 'locon_targets="${locon_targets//;/,}"' in launcher


def test_source_balanced_retry_can_wait_for_an_experiment_slot() -> None:
    script = Path(
        "scripts/v0data/submit_joint_source_balanced_continuation.sh"
    ).read_text()
    library = Path("scripts/v0data/joint_continuation_lib.sh").read_text()

    assert 'run_suffix="${RUN_SUFFIX:-_source_balanced_epoch32_reset}"' in script
    assert 'initial_dependency="${INITIAL_DEPENDENCY:-}"' in script
    assert 'smoke_args+=(--dependency="$initial_dependency")' in library


def test_joint_head_warmup_branches_one_checkpoint_into_both_strategies() -> None:
    script = Path("scripts/v0data/submit_joint_head_warmup_then_adapters.sh").read_text()
    branch = Path(
        "scripts/v0data/slurm_submit_joint_adapters_from_head_warmup.sbatch"
    ).read_text()
    launcher = Path("scripts/v0data/slurm_joint_multidataset_adapters.sbatch").read_text()

    assert "BACKBONE_LORA=0" in script
    assert "RUN_BASENAME=${run_basename}" in script
    assert 'warmup_max_epochs="${WARMUP_MAX_EPOCHS:-20}"' in script
    assert 'warmup_patience="${WARMUP_PATIENCE:-5}"' in script
    assert 'warmup_time_limit="${WARMUP_TIME_LIMIT:-6-00:00:00}"' in script
    assert "NUM_EPOCHS=${warmup_max_epochs}" in script
    assert "EARLY_STOPPING_PATIENCE=${warmup_patience}" in script
    assert '--time="$warmup_time_limit"' in script
    assert 'dependency="afterok:${smoke}_0"' in script
    assert 'dependency="afterok:${warmup}_0"' in script
    assert 'source_checkpoint="${source_run}/best"' in branch
    assert 'sbatch_bin="${SBATCH_BIN:-sbatch}"' in branch
    assert "learning_rate=3e-4" in branch
    assert "expand_backbone_adapters=1" in branch
    assert (
        'locon_targets="${LOCON_TARGETS:-downres_block_2;downres_block_3;'
        'downres_block_4;downres_block_5}"' in branch
    )
    assert "LOCON_TARGETS=${locon_targets}" in script
    assert branch.count("submit_continuation") == 2
    assert 'submit_continuation 0 "$source_run"' in branch
    assert 'submit_continuation 1 "$source_run"' in branch
    assert 'if [[ "${BACKBONE_LORA:-1}" == "1" ]]' in launcher
    assert 'RUN_BASENAME:-joint_all_nonencode_${strategy//+/_}' in launcher


def test_joint_head_warmup_exposes_isolated_pretrained_initialization() -> None:
    script = Path("scripts/v0data/submit_joint_head_warmup_then_adapters.sh").read_text()
    launcher = Path("scripts/v0data/slurm_joint_multidataset_adapters.sbatch").read_text()

    assert '--pretrained-head-initialization "${PRETRAINED_HEAD_INITIALIZATION:-none}"' in launcher
    assert 'initializer="${PRETRAINED_HEAD_INITIALIZATION:-none}"' in script
    assert 'initializer_suffix="_${initializer}"' in script
    assert "PRETRAINED_HEAD_INITIALIZATION=${initializer}" in script
    assert "BRANCH_TAG=${branch_tag}" in script
    assert 'smoke_args+=(--dependency="${INITIAL_DEPENDENCY}")' in script
    assert "semantic_neural_accessibility_bootstrap" in script
    assert 'DATASET_CONFIG:-$default_dataset_config' in script
    assert 'run_tag="${RUN_TAG:-}"' in script
    assert 'run_suffix="_head_warmup_tempered${tag_suffix}${initializer_suffix}"' in script


def test_all_study_native_evaluation_uses_and_validates_provisional_runs() -> None:
    submitter = Path(
        "scripts/v0data/submit_joint_multidataset_evaluations.sh"
    ).read_text()
    worker = Path("scripts/v0data/slurm_joint_multidataset_evaluate.sbatch").read_text()

    assert 'run_suffix="${RUN_SUFFIX:-_provisional}"' in submitter
    assert "for strategy in lora lora_locon" in submitter
    assert "Missing evaluation checkpoint metadata" in submitter
    assert "CHECKPOINT_ROOT=${checkpoint_root}" in submitter
    assert "Set both LORA_JOB and LOCON_JOB, or neither" in submitter
    assert 'if [[ -n "$lora_job" ]]' in submitter
    assert "Missing source checkpoint metadata" in worker


def test_single_checkpoint_native_evaluation_is_explicit_and_ten_tasks() -> None:
    submitter = Path(
        "scripts/v0data/submit_joint_checkpoint_evaluations.sh"
    ).read_text()
    worker = Path("scripts/v0data/slurm_joint_multidataset_evaluate.sbatch").read_text()

    assert 'strategy="${STRATEGY:?' in submitter
    assert 'source_checkpoint="${SOURCE_CHECKPOINT:?' in submitter
    assert 'evaluation_tag="${EVALUATION_TAG:?' in submitter
    assert "--array=0-9%4" in submitter
    assert 'dependency="afterok:${source_job}"' in submitter
    assert '"${EVALUATION_STRATEGY:-}"' in worker
    assert 'source_checkpoint="${SOURCE_CHECKPOINT:?' in worker
    assert 'evaluation_tag="${EVALUATION_TAG:?' in worker


def test_joint_launcher_exposes_evaluate_only_with_checkpoint() -> None:
    script = Path("scripts/v0data/slurm_joint_adapter_comparison.sbatch").read_text()

    assert '"${EVALUATE_ONLY:-0}" == "1"' in script
    assert "EVALUATE_ONLY requires RESUME_FROM" in script
    assert "EXTRA_ARGS+=(--evaluate-only)" in script
    assert '"${DEFER_TEST_EVALUATION:-0}" == "1"' in script
    assert "EXTRA_ARGS+=(--defer-test-evaluation)" in script


def test_smoke_runs_use_isolated_checkpoint_names() -> None:
    for path in LAUNCHERS:
        script = Path(path).read_text()

        assert '"${SMOKE:-0}" == "1"' in script
        assert '!= *_smoke' in script
        assert '_smoke"' in script


def test_gene_supervision_launchers_expose_balanced_window_ordering() -> None:
    for path in LAUNCHERS[:2]:
        script = Path(path).read_text()

        assert '"${BALANCE_GENE_WINDOWS:-0}" == "1"' in script
        assert "EXTRA_ARGS+=(--balance-gene-windows)" in script


def test_multidataset_gene_balancing_is_opt_in_and_propagates_after_warmup() -> None:
    launcher = Path("scripts/v0data/slurm_joint_multidataset_adapters.sbatch").read_text()
    submitter = Path("scripts/v0data/submit_joint_head_warmup_then_adapters.sh").read_text()
    branch = Path(
        "scripts/v0data/slurm_submit_joint_adapters_from_head_warmup.sbatch"
    ).read_text()
    continuation = Path("scripts/v0data/joint_continuation_lib.sh").read_text()

    assert '"${BALANCE_GENE_WINDOWS:-0}" == "1"' in launcher
    assert "extra_args+=(--balance-gene-windows)" in launcher
    assert launcher.count("--balance-gene-windows") == 1
    assert 'balance_gene_windows="${BALANCE_GENE_WINDOWS:-0}"' in submitter
    assert "BALANCE_GENE_WINDOWS=${balance_gene_windows}" in submitter
    assert 'balance_gene_windows="${BALANCE_GENE_WINDOWS:-0}"' in branch
    assert "BALANCE_GENE_WINDOWS=${balance_gene_windows:-0}" in continuation


def test_hda_gene_window_repeat_screen_has_ordering_matched_control() -> None:
    launcher = Path("scripts/v0data/slurm_joint_adapter_comparison.sbatch").read_text()
    script = Path("scripts/v0data/submit_hda_gene_window_repeat_screen.sh").read_text()

    assert '--gene-window-repeats "${GENE_WINDOW_REPEATS}"' in launcher
    assert "for repeats in 0 1" in script
    assert "BALANCE_GENE_WINDOWS=1" in script
    assert "GENE_WINDOW_REPEATS=${repeats}" in script
    assert "--array=0" in script


def test_zemke_joint_evaluator_covers_each_species_and_strategy() -> None:
    script = Path("scripts/v0data/submit_zemke_joint_native_evaluations.sh").read_text()

    assert "for species in human macaque marmoset mouse" in script
    assert "submit_evaluation 0 lora" in script
    assert "submit_evaluation 1 lora_locon" in script
    assert "RUN_SUFFIX=_joint_epoch${epoch}_eval" in script
    assert "EVALUATE_ONLY=1" in script
    assert "EVALUATE_SPECIES=${species}" in script
    assert "slurm_zemke2023_joint_adapters.sbatch" in script
    assert 'sbatch_bin="${SBATCH_BIN:-sbatch}"' in script

    launcher = Path("scripts/v0data/slurm_zemke2023_joint_adapters.sbatch").read_text()
    assert 'EXTRA_ARGS+=(--evaluate-only)' in launcher
    assert '--evaluate-species "${EVALUATE_SPECIES}"' in launcher
    assert 'run_name="zemke2023_${EVALUATE_SPECIES}_${strategy//+/_}${RUN_SUFFIX:-}"' in launcher


def test_johansen_joint_evaluator_uses_corrected_checkpoints() -> None:
    script = Path("scripts/v0data/submit_johansen_joint_native_evaluations.sh").read_text()

    assert "for species in human macaque marmoset" in script
    assert "submit_evaluation 0 lora" in script
    assert "submit_evaluation 1 lora_locon" in script
    assert "johansen_joint_${strategy}_rawcount_geneonly_corrw1/best" in script
    assert 'if [[ -n "$source_job" ]]' in script
    assert 'sbatch_args+=(--dependency="afterok:${source_job}_${task}")' in script
    assert "EVALUATE_SPECIES=${species}" in script


def test_zemke_species_continuation_requires_optimizer_state_by_default() -> None:
    script = Path("scripts/v0data/submit_zemke_species_continuations.sh").read_text()

    assert 'tasks="${TASKS:-2;3}"' in script
    assert '"${REQUIRE_OPTIMIZER_STATE:-1}" == "1"' in script
    assert 'test -d "$source/optimizer_state"' in script
    assert "RESUME_FROM=${source}" in script
    assert 'sbatch_args+=(--dependency="$DEPENDENCY")' in script


def test_johansen_joint_continuation_requires_corrected_state() -> None:
    script = Path("scripts/v0data/submit_johansen_joint_continuations.sh").read_text()

    assert 'tasks="${TASKS:-0;1}"' in script
    assert "johansen-rna-corrected/geneonly-corrw1/species.json" in script
    assert "johansen_joint_${strategy}_rawcount_geneonly_corrw1/last" in script
    assert '"${REQUIRE_OPTIMIZER_STATE:-1}" == "1"' in script
    assert 'test -d "$source/optimizer_state"' in script
    assert "RESUME_FROM=${source}" in script
    assert "RUN_SUFFIX=_rawcount_geneonly_corrw1" in script
    assert 'sbatch_args+=(--dependency="$DEPENDENCY")' in script


def test_hda_joint_epoch_three_matches_the_optimizer_reset_boundary() -> None:
    script = Path("scripts/v0data/submit_hda_joint_matched_epoch3.sh").read_text()

    assert "hda-joint_lora/last/metrics.json" in script
    assert "hda-joint_lora_locon/last" in script
    assert 'test "$(jq -er \'.epoch\' "$lora_metrics")" -eq 3' in script
    assert 'test "$(jq -er \'.epoch\' "$locon_source/metrics.json")" -eq 2' in script
    assert "DATASET=hda-joint" in script
    assert "NUM_EPOCHS=3" in script


def test_zemke2024_screen_changes_only_the_rna_objective() -> None:
    script = Path("scripts/v0data/submit_zemke2024_rna_correlation_screen.sh").read_text()

    assert "prepare_head_objective_config.py" in script
    assert "--head zemke2024_all_rna" in script
    assert '--correlation-loss-weight "$weight"' in script
    assert "--array=4-5%2" in script
    assert 'dependency="afterok:${smoke}_*"' in script


def test_zemke2024_all_head_correlation_screen_uses_global_override() -> None:
    script = Path("scripts/v0data/submit_zemke2024_all_correlation_screen.sh").read_text()

    assert "prepare_head_objective_config.py" not in script
    assert "CORRELATION_LOSS_WEIGHT=${weight}" in script
    assert "RUN_SUFFIX=_all_corrw${suffix}" in script
    assert "--array=4-5%2" in script
    assert 'dependency="afterok:${smoke}_*"' in script


def test_pretrained_head_screen_supports_neural_accessibility_bootstrap() -> None:
    script = Path("scripts/v0data/submit_pretrained_head_bootstrap_screens.sh").read_text()

    assert "neural_accessibility_bootstrap)" in script
    assert "_neural_accessibility_bootstrap_screen" in script
    assert 'sbatch_bin="${SBATCH_BIN:-sbatch}"' in script
    assert '"${SMOKE_GATE:-0}" == "1"' in script
    assert '--nice="${NICE:-50}"' in script
    assert 'dependency="afterok:${smoke}_*"' in script


def test_hda_rna_only_screen_is_deterministically_initialized() -> None:
    script = Path("scripts/v0data/submit_hda_rna_only_screen.sh").read_text()

    assert "prepare_single_head_config.py" in script
    assert "--head hda_rna" in script
    assert "PRETRAINED_HEAD_INITIALIZATION=neural_accessibility_bootstrap" in script
    assert "RUN_SUFFIX=_rna_only_neural_accessibility_bootstrap_screen" in script
    assert 'dependency="afterok:${smoke}_*"' in script


def test_johansen_row_objective_uses_corrected_raw_count_targets() -> None:
    script = Path("scripts/v0data/submit_johansen_row_correlation_screen.sh").read_text()

    assert "johansen-rna-corrected/geneonly-corrw1/species.json" in script
    assert "johansen-rna-corrected/geneonly-rowcorrw" in script
    assert "johansen-fragment-joint-depth-filtered/species.json" not in script
    assert 'sbatch_bin="${SBATCH_BIN:-sbatch}"' in script
    assert 'smoke=$("$sbatch_bin"' in script


def test_hda_row_objective_supports_scheduler_dry_run() -> None:
    script = Path("scripts/v0data/submit_hda_row_correlation_screen.sh").read_text()

    assert 'sbatch_bin="${SBATCH_BIN:-sbatch}"' in script
    assert 'smoke=$("$sbatch_bin"' in script
    assert 'tasks="${TASKS:-0-1%2}"' in script
    assert "BALANCE_GENE_WINDOWS=1" in script


def test_liu_row_objective_uses_corrected_gene_only_targets() -> None:
    script = Path("scripts/v0data/submit_liu_row_correlation_screen.sh").read_text()

    assert "liu-hdma/joint/targets_geneonly_corrw1.json" in script
    assert "--head liu_rna" in script
    assert '--row-correlation-loss-weight "$weight"' in script
    assert "BALANCE_GENE_WINDOWS=1" in script
    assert 'sbatch_bin="${SBATCH_BIN:-sbatch}"' in script


def test_reconstructed_row_sweep_has_matched_ordering_controls() -> None:
    script = Path(
        "scripts/v0data/submit_reconstructed_row_correlation_sweeps.sh"
    ).read_text()

    assert "weights=(0 0.1 1 10)" in script
    assert "suffixes=(0 0p1 1 10)" in script
    assert "prepare_reconstructed_row_sweeps.py" in script
    assert script.count("BALANCE_GENE_WINDOWS=1") == 2
    assert script.count("--array=0") == 4
    assert script.count('dependency="afterok:${') == 2


def test_hda_factorized_rna_screen_changes_only_the_rna_projection() -> None:
    script = Path("scripts/v0data/submit_hda_factorized_rna_screen.sh").read_text()

    assert "prepare_factorized_head_config.py" in script
    assert "--head hda_rna" in script
    assert 'rank="${RNA_OUTPUT_RANK:-16}"' in script
    assert '--nice="${NICE:-100}"' in script
    assert 'dependency="afterok:${smoke}_*"' in script


def test_hda_cosine_screen_is_staged_and_lower_rate() -> None:
    script = Path("scripts/v0data/submit_hda_cosine_lr_screen.sh").read_text()

    assert "LEARNING_RATE=3e-4" in script
    assert "LEARNING_RATE_SCHEDULE=warmup_cosine" in script
    assert "WARMUP_STEPS=262" in script
    assert "NUM_EPOCHS=8" in script
    assert "DEFER_TEST_EVALUATION=1" in script
    assert "--array=0" in script
    assert 'dependency="afterok:${smoke}_*"' in script


def test_liu_cosine_screen_uses_corrected_targets_and_is_staged() -> None:
    script = Path("scripts/v0data/submit_liu_cosine_lr_screen.sh").read_text()

    assert "DATASET=liu-hdma" in script
    assert "outputs/v0data/liu-hdma/joint/targets.json" in script
    assert "LEARNING_RATE=3e-4" in script
    assert "LEARNING_RATE_SCHEDULE=warmup_cosine" in script
    assert "WARMUP_STEPS=262" in script
    assert "DEFER_TEST_EVALUATION=1" in script
    assert "--array=0" in script
    assert 'dependency="afterok:${smoke}_*"' in script


def test_liu_exon_window_screen_is_lora_only_and_staged() -> None:
    script = Path("scripts/v0data/submit_liu_exon_window_screen.sh").read_text()

    assert "prepare_gene_window_assignment.py" in script
    assert "max_exon_overlap_scaled" in script
    assert "targets_geneonly_corrw1.json" in script
    assert script.count("--array=0") == 4
    assert "NUM_EPOCHS=1" in script
    assert 'dependency="afterok:${smoke}_*"' in script
    assert script.count('dependency="afterok:${full}_*"') == 2
    assert "_canonical_exonwindow_eval" in script
    assert "_exonwindow_train_fullspan_eval" in script
    assert script.count("EVALUATE_ONLY=1") == 2


def test_zemke_published_row_screen_has_ordering_control_and_coordinate_eval() -> None:
    script = Path(
        "scripts/v0data/submit_zemke2023_published_row_correlation_screen.sh"
    ).read_text()

    assert 'weights_string="${ROW_WEIGHTS:-0;1;10}"' in script
    assert "--double-centered-weight 0" in script
    assert "BALANCE_GENE_WINDOWS=1" in script
    assert script.count("--array=0") == 3
    assert "zemke2023-species/human/targets.json" in script
    assert "EVALUATE_ONLY=1" in script
    assert 'dependency="afterok:${full}_0"' in script


def test_jax_launchers_expose_learning_rate_schedule() -> None:
    for path in LAUNCHERS:
        script = Path(path).read_text()
        assert '"${LEARNING_RATE_SCHEDULE:-constant}"' in script
        assert '"${WARMUP_STEPS:-0}"' in script
        assert '"${MINIMUM_LEARNING_RATE_RATIO:-0.1}"' in script


def test_zemke_direct_gene_screen_requires_complete_target_agreement() -> None:
    script = Path("scripts/v0data/submit_zemke2023_direct_gene_screen.sh").read_text()

    assert "v0data_zemke2023_gene_target_agreement.json" in script
    assert "(.species | length) == 4" in script
    assert ".raw_cpm_double_centered_r >= $minimum" in script
    assert "zemke2023-gene-supervision/human/targets.json" in script
    assert "CORRELATION_LOSS_WEIGHT=${weight}" in script
    assert 'dependency="afterok:${smoke}_*"' in script


def test_zemke_published_gene_screen_reports_direct_and_coordinate_metrics() -> None:
    script = Path("scripts/v0data/submit_zemke2023_published_gene_screen.sh").read_text()

    assert "zemke2023-published-gene-supervision/human/targets.json" in script
    assert 'dependency="afterok:${source_job}_0"' in script
    assert 'dependency="afterok:${smoke}_*"' in script
    assert "zemke2023-species/human/targets.json" in script
    assert "EVALUATE_ONLY=1" in script
    assert 'dependency="afterok:${full}_${task}"' in script


def test_zemke2024_direct_gene_screen_requires_supported_group_agreement() -> None:
    script = Path("scripts/v0data/submit_zemke2024_direct_gene_screen.sh").read_text()

    assert "v0data_zemke2024_gene_track_agreement.json" in script
    assert ".direct_gene_groups == 18" in script
    assert "Astro1_all" in script and "Micro2_all" in script
    assert ".raw_cpm_double_centered_r >= .minimum_r" in script
    assert "zemke2024-gene-supervision/targets.json" in script
    assert "BALANCE_GENE_WINDOWS=1" in script
    assert 'dependency="afterok:${smoke}_*"' in script


def test_joint_all_gene_warmup_uses_direct_gene_rna_for_both_zemke_datasets() -> None:
    script = Path(
        "scripts/v0data/submit_joint_all_gene_warmup_then_adapters.sh"
    ).read_text()

    assert "prepare_gene_only_species.py" in script
    assert "zemke2024_rna_reprocessing/prepare_gene_only_target.py" in script
    assert '--zemke2024-targets "$zemke2024_target"' in script
    assert "--zemke-rna-weight 1" in script
    assert "--zemke2024-rna-weight 1" in script
    assert 'source_specific_heads="${SOURCE_SPECIFIC_HEADS:-0}"' in script
    assert 'separate_head_updates="${SEPARATE_HEAD_UPDATES:-0}"' in script
    assert "metric_args+=(--sampling-strategy equal_sources)" in script
    assert "metric_args+=(--head-update-strategy separate_heads)" in script
    assert "prepare_source_specific_joint_heads.py" in script
    assert 'run_tag="all_gene_source_specific"' in script
    assert 'run_tag="${run_tag}_separate_heads"' in script
    assert 'RUN_TAG="$run_tag"' in script
    for split in ("TRAIN", "VALID", "TEST"):
        assert f'SMOKE_LIMIT_{split}="${{SMOKE_LIMIT_{split}:-40}}"' in script


def test_zemke_gene_warmup_branches_with_expanded_locon_coverage() -> None:
    script = Path(
        "scripts/v0data/submit_joint_zemke_gene_warmup_then_adapters.sh"
    ).read_text()

    assert (
        'locon_targets="${LOCON_TARGETS:-downres_block_2;downres_block_3;'
        'downres_block_4;downres_block_5}"' in script
    )
    assert "LOCON_TARGETS=${locon_targets}" in script
