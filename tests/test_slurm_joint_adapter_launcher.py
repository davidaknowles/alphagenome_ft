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
    assert 'dependency="afterok:${source_job}_${task}"' in script
    assert "EVALUATE_SPECIES=${species}" in script


def test_zemke_species_continuation_requires_optimizer_state_by_default() -> None:
    script = Path("scripts/v0data/submit_zemke_species_continuations.sh").read_text()

    assert 'tasks="${TASKS:-2;3}"' in script
    assert '"${REQUIRE_OPTIMIZER_STATE:-1}" == "1"' in script
    assert 'test -d "$source/optimizer_state"' in script
    assert "RESUME_FROM=${source}" in script


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


def test_liu_row_objective_uses_corrected_gene_only_targets() -> None:
    script = Path("scripts/v0data/submit_liu_row_correlation_screen.sh").read_text()

    assert "liu-hdma/joint/targets_geneonly_corrw1.json" in script
    assert "--head liu_rna" in script
    assert '--row-correlation-loss-weight "$weight"' in script
    assert "BALANCE_GENE_WINDOWS=1" in script
    assert 'sbatch_bin="${SBATCH_BIN:-sbatch}"' in script


def test_hda_factorized_rna_screen_changes_only_the_rna_projection() -> None:
    script = Path("scripts/v0data/submit_hda_factorized_rna_screen.sh").read_text()

    assert "prepare_factorized_head_config.py" in script
    assert "--head hda_rna" in script
    assert 'rank="${RNA_OUTPUT_RANK:-16}"' in script
    assert '--nice="${NICE:-100}"' in script
    assert 'dependency="afterok:${smoke}_*"' in script


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
