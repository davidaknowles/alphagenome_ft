import json
from pathlib import Path

import numpy as np
import pyBigWig

from scripts.v0data.audit_target_differential_signal import (
    _canonical_chromosomes,
    audit_manifest,
)


def _write_bigwig(path: Path, values: np.ndarray) -> None:
    starts = np.arange(values.size, dtype=np.int64) * 10
    with pyBigWig.open(str(path), "w") as handle:
        handle.addHeader([("chr1", int(values.size * 10))])
        handle.addEntries(
            ["chr1"] * values.size,
            starts.tolist(),
            ends=(starts + 10).tolist(),
            values=values.astype(float).tolist(),
        )


def test_audit_manifest_reports_finite_differential_signal(tmp_path: Path) -> None:
    positions = np.arange(100, dtype=np.float64)
    paths = [tmp_path / f"track_{index}.bw" for index in range(3)]
    _write_bigwig(paths[0], 1.0 + positions % 7)
    _write_bigwig(paths[1], 2.0 + positions % 5)
    _write_bigwig(paths[2], 3.0 + positions % 3)
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "heads": [
                    {
                        "id": "atac",
                        "kind": "atac",
                        "targets": [{"path": str(path)} for path in paths],
                    }
                ],
            }
        )
    )

    serial = audit_manifest(
        manifest_path,
        num_windows=2,
        window_size=100,
        num_bins=10,
        excluded_chromosomes=set(),
        seed=3,
    )
    parallel = audit_manifest(
        manifest_path,
        num_windows=2,
        window_size=100,
        num_bins=10,
        excluded_chromosomes=set(),
        seed=3,
        workers=3,
    )

    assert parallel == serial
    head = serial["heads"][0]
    assert head["num_tracks"] == 3
    assert head["num_observations"] == 20
    assert 0 < head["double_centered_variance_fraction"] <= 1
    assert np.isfinite(head["median_pairwise_track_correlation"])


def test_canonical_chromosomes_accepts_refseq_accessions() -> None:
    chromosomes, probabilities = _canonical_chromosomes(
        {
            "NC_041754.1": 1_000,
            "NC_041755.1": 900,
            "NW_012345.1": 800,
            "chr1_random": 700,
        },
        window_size=100,
        excluded={"NC_041755.1"},
    )

    assert chromosomes == ["NC_041754.1"]
    np.testing.assert_array_equal(probabilities, np.asarray([1.0]))


def test_audit_counts_uncovered_bases_as_zero(tmp_path: Path) -> None:
    paths = [tmp_path / f"partial_{index}.bw" for index in range(2)]
    for path in paths:
        with pyBigWig.open(str(path), "w") as handle:
            handle.addHeader([("chr1", 100)])
            handle.addEntries(["chr1"], [0], ends=[5], values=[10.0])
    manifest_path = tmp_path / "targets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "partial",
                "heads": [
                    {
                        "id": "atac",
                        "kind": "atac",
                        "targets": [{"path": str(path)} for path in paths],
                    }
                ],
            }
        )
    )

    result = audit_manifest(
        manifest_path,
        num_windows=1,
        window_size=100,
        num_bins=10,
        excluded_chromosomes=set(),
        seed=3,
    )

    head = result["heads"][0]
    assert head["nonzero_fraction"] == 0.1
    assert head["standard_deviation"] == 1.5
