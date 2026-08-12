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

    assert "submit_strategy lora 0,2,4,6" in script
    assert "submit_strategy lora_locon 1,3,5,7" in script
    assert "RUN_SUFFIX=_joint_epoch${epoch}_eval" in script
    assert "EVALUATE_ONLY=1" in script
    assert 'sbatch_bin="${SBATCH_BIN:-sbatch}"' in script


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
