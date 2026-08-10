from pathlib import Path

from alphagenome_ft.finetune.data import build_fasta_index


def test_existing_fasta_index_returns_chromosome_sizes(tmp_path: Path):
    fasta = tmp_path / "tiny.fa"
    fasta.write_text(">chr1\nACGT\n")
    Path(f"{fasta}.fai").write_text("chr1\t4\t6\t4\t5\n")

    assert build_fasta_index(fasta) == {"chr1": 4}
