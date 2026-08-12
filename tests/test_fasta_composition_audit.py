from pathlib import Path

from pyfaidx import Fasta

from scripts.v0data.audit_fasta_composition import audit_chromosome


def test_audit_chromosome_reports_gaps_and_softmasking(tmp_path: Path) -> None:
    fasta_path = tmp_path / "reference.fa"
    fasta_path.write_text(">chr1\nACgtNNRX\n")
    reference = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=False)
    try:
        result = audit_chromosome(reference, "chr1", chunk_size=3)
    finally:
        reference.close()

    assert result["length"] == 8
    assert result["base_counts"] == {"A": 1, "C": 1, "G": 1, "N": 2, "R": 1, "T": 1, "X": 1}
    assert result["canonical_fraction"] == 0.5
    assert result["gc_fraction_of_canonical"] == 0.5
    assert result["n_fraction"] == 0.25
    assert result["softmasked_fraction"] == 0.25
    assert result["unexpected_fraction"] == 0.25
