#!/usr/bin/env python
"""Prepare references and a balanced four-species Zemke 2023 configuration."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import sys
import urllib.request

import pyBigWig

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import build_fasta_index

UCSC_FASTA_URLS = {
    "marmoset": "https://hgdownload.soe.ucsc.edu/goldenPath/calJac4/bigZips/calJac4.fa.gz",
    "mouse": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
}
SPECIES = ("human", "macaque", "marmoset", "mouse")
CANONICAL_CHROMOSOME_PATTERN = re.compile(r"^chr(?:[1-9][0-9]*|X)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-fasta", required=True, type=Path)
    parser.add_argument("--macaque-fasta", required=True, type=Path)
    parser.add_argument("--targets-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _open_binary(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def materialize_fasta(
    source: Path,
    destination: Path,
    *,
    rename_ncbi_chromosomes: bool = False,
) -> Path:
    if destination.exists():
        build_fasta_index(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    chromosome_pattern = re.compile(rb"\bchromosome\s+([^,\s]+),")
    seen: set[bytes] = set()
    with _open_binary(source) as input_handle, temporary.open("wb") as output_handle:
        for line in input_handle:
            if line.startswith(b">") and rename_ncbi_chromosomes:
                match = chromosome_pattern.search(line)
                if b"mitochondrion, complete genome" in line:
                    name = b"chrM"
                    if name in seen:
                        raise ValueError(f"Duplicate renamed FASTA sequence {name.decode()}.")
                    seen.add(name)
                    line = b">" + name + b"\n"
                elif match is not None:
                    chromosome = match.group(1)
                    chromosome = b"M" if chromosome == b"MT" else chromosome
                    name = b"chr" + chromosome
                    if name in seen:
                        raise ValueError(f"Duplicate renamed FASTA sequence {name.decode()}.")
                    seen.add(name)
                    line = b">" + name + b"\n"
                else:
                    seen.add(line[1:].split(None, 1)[0])
            output_handle.write(line)
    temporary.replace(destination)
    build_fasta_index(destination)
    return destination


def download_fasta(url: str, destination: Path) -> Path:
    if destination.exists():
        build_fasta_index(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    compressed = destination.with_suffix(".fa.gz")
    if not compressed.exists():
        temporary_download = compressed.with_name(f"{compressed.name}.tmp-{os.getpid()}")
        with (
            urllib.request.urlopen(url) as response,
            temporary_download.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=16 * 1024 * 1024)
        temporary_download.replace(compressed)
    return materialize_fasta(compressed, destination)


def verify_bigwig_reference(targets_path: Path, fasta_path: Path) -> list[str]:
    payload = json.loads(targets_path.read_text())
    fasta_chromosomes = build_fasta_index(fasta_path)
    modeled_chromosomes = sorted(
        (
            chromosome
            for chromosome in fasta_chromosomes
            if CANONICAL_CHROMOSOME_PATTERN.match(chromosome)
        ),
        key=lambda chromosome: (
            chromosome == "chrX",
            int(chromosome[3:]) if chromosome[3:].isdigit() else 0,
        ),
    )
    if not {"chr8", "chr9"}.issubset(modeled_chromosomes):
        raise ValueError(f"{fasta_path} lacks the required chr8 and chr9 holdouts.")
    for head in payload["heads"]:
        for target in head["targets"]:
            with pyBigWig.open(target["path"]) as bigwig:
                target_chromosomes = bigwig.chroms()
            missing = sorted(set(modeled_chromosomes) - set(target_chromosomes))
            mismatched = sorted(
                chromosome
                for chromosome in set(modeled_chromosomes) & set(target_chromosomes)
                if target_chromosomes[chromosome] != fasta_chromosomes[chromosome]
            )
            if missing or mismatched:
                raise ValueError(
                    f"{target['path']} does not match {fasta_path} on modeled chromosomes; "
                    f"missing={missing[:10]}, size mismatches={mismatched[:10]}."
                )
    return modeled_chromosomes


def pooled_target_configs(targets_root: Path, output_dir: Path) -> dict[str, Path]:
    payloads = {
        species: json.loads((targets_root / f"zemke2023-{species}" / "targets.json").read_text())
        for species in SPECIES
    }
    signatures = {
        species: [
            (head["id"], tuple(target["label"] for target in head["targets"]))
            for head in payload["heads"]
        ]
        for species, payload in payloads.items()
    }
    if len({repr(signature) for signature in signatures.values()}) != 1:
        raise ValueError(f"Species target layouts differ: {signatures}")
    rna_heads = [payloads[species]["heads"][1] for species in SPECIES]
    pooled_means = [
        sum(float(head["targets"][index]["nonzero_mean"]) for head in rna_heads) / len(rna_heads)
        for index in range(len(rna_heads[0]["targets"]))
    ]
    paths = {}
    for species, payload in payloads.items():
        for target, pooled_mean in zip(payload["heads"][1]["targets"], pooled_means, strict=True):
            target["nonzero_mean"] = pooled_mean
        species_dir = output_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)
        path = species_dir / "targets.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        paths[species] = path.resolve()
    return paths


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    references = output_dir / "references"
    fasta_paths = {
        "human": args.human_fasta.expanduser().resolve(),
        "macaque": materialize_fasta(
            args.macaque_fasta.expanduser().resolve(),
            references / "rheMac10.fa",
            rename_ncbi_chromosomes=True,
        ),
        "marmoset": download_fasta(UCSC_FASTA_URLS["marmoset"], references / "calJac4.fa"),
        "mouse": download_fasta(UCSC_FASTA_URLS["mouse"], references / "mm10.fa"),
    }
    target_paths = pooled_target_configs(args.targets_root.expanduser().resolve(), output_dir)
    modeled_chromosomes = {}
    for species in SPECIES:
        modeled_chromosomes[species] = verify_bigwig_reference(
            target_paths[species], fasta_paths[species]
        )
    payload = {
        "species": [
            {
                "name": species,
                "organism": "MUS_MUSCULUS" if species == "mouse" else "HOMO_SAPIENS",
                "fasta": str(fasta_paths[species]),
                "targets_config": str(target_paths[species]),
                "valid_chroms": "chr8",
                "test_chroms": "chr9",
                "exclude_chroms": "chrM,chrY",
                "include_chroms": ",".join(modeled_chromosomes[species]),
            }
            for species in SPECIES
        ],
        "sampling": "round-robin with equal train batches per species",
        "track_scaling": "channel-wise RNA nonzero means pooled across species",
        "target_channels": {"atac": 20, "rna": 20},
    }
    (output_dir / "species.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
