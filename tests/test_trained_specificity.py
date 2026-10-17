from pathlib import Path

import numpy as np
import pandas as pd

from scripts.eqtl.analyze_trained_specificity import analyze, diagonal_test
from scripts.eqtl.benchmark import GROUP_TRACKS

TRACKS = [
    "ASC", "Endo", "L2_3_IT", "L4_5_IT", "L5_6_NP", "L5_ET", "L5_IT", "L6_CT", "L6_IT", "L6_IT_CAR3",
    "L6b", "LAMP5", "MGC", "ODC", "OPC", "PVALB", "SNCG", "SST", "VIP", "VLMC",
]


def _write_model(root: Path, model: str, specific: bool, seed: int) -> None:
    rng = np.random.default_rng(seed)
    groups = list(GROUP_TRACKS)
    rows, tss = [], []
    for group in groups:
        for label in (0, 1) * 150:
            effect = rng.gamma(2.0, 1.0, size=len(TRACKS))
            if label and specific:
                for track in GROUP_TRACKS[group]:
                    effect[TRACKS.index(track)] += 1.5
            rows.append({"panel": group, "pos": len(rows), "label": label, "cell_group": group})
            tss.append(effect)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / f"scores_{model}.tsv", sep="\t", index=False)
    tss = np.asarray(tss, dtype=np.float32)
    np.savez(root / f"track_effects_{model}.npz", tss=tss, gene=tss, tracks=np.asarray(TRACKS),
             panel=np.asarray([r["panel"] for r in rows]), pos=np.asarray([r["pos"] for r in rows]))


def test_diagonal_test_is_exact_for_identity():
    matrix = pd.DataFrame(np.eye(4) * 0.2 + 0.5)
    observed, p_value = diagonal_test(matrix)
    assert np.isclose(observed, 0.2)
    assert np.isclose(p_value, 1 / 24)


def test_specific_tracks_are_detected(tmp_path: Path):
    _write_model(tmp_path / "data" / "models_specific", "specific", specific=True, seed=0)
    _write_model(tmp_path / "data" / "models_shared", "shared", specific=False, seed=1)
    analyze({"toy": tmp_path / "data"}, tmp_path / "out")
    tests = pd.read_csv(tmp_path / "out" / "trained_specificity_tests.tsv", sep="\t").set_index("model")
    assert tests.loc["specific", "permutation_p"].max() < 0.01
    assert tests.loc["shared", "permutation_p"].min() > 0.01
    ranks = pd.read_csv(tmp_path / "out" / "trained_specificity_ranks.tsv", sep="\t")
    assert (ranks[ranks.model == "specific"].matched_group_rank == 1).all()
    assert (tmp_path / "out" / "trained_specificity_toy.png").exists()
