import json
from pathlib import Path

import numpy as np

from scripts.v0data.zemke2023_rna_reprocessing.prepare_marmoset_ano6_audit import (
    OUTLIER_END,
    OUTLIER_START,
    prepare_audit,
)


def test_prepare_ano6_audit_changes_only_marmoset_interval_exclusion(tmp_path):
    source_config = {
        "datasets": [
            {
                "name": "zemke2023",
                "sources": [
                    {"name": "human", "fasta": "human.fa"},
                    {"name": "marmoset", "fasta": "marmoset.fa"},
                ],
            }
        ]
    }
    config_path = tmp_path / "source.json"
    config_path.write_text(json.dumps(source_config))
    supervision_path = tmp_path / "supervision.npz"
    np.savez(
        supervision_path,
        gene_ids=np.asarray(["OTHER", "ANO6"]),
        chromosomes=np.asarray(["chr1", "chr9"]),
        starts=np.asarray([10, OUTLIER_START - 100]),
        ends=np.asarray([20, OUTLIER_END + 100]),
        groups=np.asarray(["Endo", "VLMC"]),
        cpm=np.asarray([[1.0, 9.0], [2.0, 5.0]], dtype=np.float32),
    )

    result = prepare_audit(
        dataset_config_path=config_path,
        supervision_path=supervision_path,
        output_dir=tmp_path / "audit",
        gene_id="ANO6",
    )

    generated = json.loads((tmp_path / "audit" / "datasets.json").read_text())
    human, marmoset = generated["datasets"][0]["sources"]
    assert "exclude_intervals_bed" not in human
    assert marmoset["exclude_intervals_bed"].endswith("ssu_rrna_hsa.bed")
    assert result["maximum_cpm"] == 9.0
    assert result["excluded_repeat"]["label"] == "SSU-rRNA_Hsa"
    assert (tmp_path / "audit" / "ssu_rrna_hsa.bed").read_text() == (
        f"chr9\t{OUTLIER_START}\t{OUTLIER_END}\tSSU-rRNA_Hsa\n"
    )


def test_ano6_audit_launcher_prepares_repeat_exclusion_and_uses_all_windows():
    script = Path("scripts/v0data/slurm_zemke2023_marmoset_ano6_audit.sbatch").read_text()

    assert "prepare_marmoset_ano6_audit.py" in script
    assert "variants=(baseline exclude_rrna_locus)" in script
    assert "--num-devices 1" in script
    assert "--drop-last" not in script
