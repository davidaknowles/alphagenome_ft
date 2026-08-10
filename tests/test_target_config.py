from pathlib import Path

from alphagenome.models import dna_client, dna_output

from alphagenome_ft.finetune.config import TrackInfo, _build_track_metadata


def test_track_metadata_preserves_strands(tmp_path: Path):
    tracks = [
        TrackInfo(name="cell (+)", path=tmp_path / "plus.bw", strand="+"),
        TrackInfo(name="cell (-)", path=tmp_path / "minus.bw", strand="-"),
    ]

    metadata = _build_track_metadata(
        tracks,
        dna_client.Organism.HOMO_SAPIENS,
        dna_output.OutputType.RNA_SEQ,
    )[dna_client.Organism.HOMO_SAPIENS]

    assert metadata.rna_seq["strand"].tolist() == ["+", "-"]
    assert metadata.strand_reindexing[dna_output.OutputType.RNA_SEQ].tolist() == [1, 0]
