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
