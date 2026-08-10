#!/usr/bin/env python
"""Prepare aligned human, macaque, and marmoset Allen Multiome targets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphagenome_ft.finetune import (
    PseudobulkExpression,
    build_fasta_index,
    read_gene_exons,
    read_pseudobulk_expression,
    remap_expression_gene_ids,
    write_gene_expression_supervision,
    write_stranded_exon_bigwigs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--human-fasta", type=Path, required=True)
    parser.add_argument("--human-gtf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def materialize_fasta(source: Path, destination: Path) -> Path:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        opener = gzip.open if source.suffix == ".gz" else open
        with opener(source, "rb") as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=16 * 1024 * 1024)
        temporary.replace(destination)
    build_fasta_index(destination)
    return destination


def fasta_chromosome_aliases(path: Path) -> dict[str, str]:
    """Map chromosome labels in NCBI FASTA descriptions to accessions."""
    aliases: dict[str, str] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    pattern = re.compile(r"\bchromosome\s+([^,\s]+)")
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            accession = line[1:].split(None, 1)[0]
            match = pattern.search(line)
            if match is not None:
                aliases.setdefault(match.group(1), accession)
    return aliases


def read_ortholog_map(path: Path, destination_column: str) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            row["human_ensembl"].split(".", 1)[0]: row[destination_column].split(".", 1)[0]
            for row in rows
            if row["human_ensembl"] and row[destination_column]
        }


def build_targets(
    *,
    species: str,
    expression_path: Path,
    atac_dir: Path,
    gtf: Path,
    fasta: Path,
    output_dir: Path,
    ortholog_map: dict[str, str] | None,
    gene_attribute: str,
    chromosome_aliases: dict[str, str] | None,
    shared_group_stems: set[str],
    overwrite: bool,
) -> tuple[Path, dict[str, object]]:
    expression = read_pseudobulk_expression(
        expression_path,
        normalize_cpm=True,
        gene_id_column="_index",
    )
    group_indices = [
        idx for idx, group in enumerate(expression.groups) if safe_name(group) in shared_group_stems
    ]
    expression = PseudobulkExpression(
        groups=tuple(expression.groups[idx] for idx in group_indices),
        gene_ids=expression.gene_ids,
        cpm=expression.cpm[group_indices],
    )
    if ortholog_map is not None:
        expression = remap_expression_gene_ids(expression, ortholog_map)
    chromosome_sizes = build_fasta_index(fasta)
    genes = read_gene_exons(
        gtf,
        gene_ids=expression.gene_ids,
        chromosome_sizes=chromosome_sizes,
        gene_attribute=gene_attribute,
        chromosome_aliases=chromosome_aliases,
    )
    coverage = len(genes) / len(expression.gene_ids)
    if coverage < 0.75:
        raise ValueError(
            f"{species} matched only {len(genes)}/{len(expression.gene_ids)} genes ({coverage:.1%})."
        )

    atac_by_group = {path.stem: path.resolve() for path in atac_dir.glob("*.bw")}
    missing_atac = [group for group in expression.groups if safe_name(group) not in atac_by_group]
    if missing_atac:
        raise ValueError(f"{species} is missing ATAC groups: {missing_atac}")

    species_dir = output_dir / species
    rna_targets = write_stranded_exon_bigwigs(
        expression,
        genes=genes,
        chromosome_sizes=chromosome_sizes,
        output_dir=species_dir / "rna_exon_cpm",
        overwrite=overwrite,
    )
    supervision_path = species_dir / "gene_expression_supervision.npz"
    matched_genes = write_gene_expression_supervision(
        supervision_path,
        expression,
        genes=genes,
    )
    atac_targets = [
        {
            "path": str(atac_by_group[safe_name(group)]),
            "label": group,
            "strand": ".",
        }
        for group in expression.groups
    ]
    targets = {
        "heads": [
            {
                "id": "allen_atac",
                "source": "predefined",
                "kind": "atac",
                "resolutions": [1, 128],
                "apply_squashing": False,
                "targets": atac_targets,
            },
            {
                "id": "allen_rna",
                "source": "predefined",
                "kind": "rna_seq",
                "resolutions": [1, 128],
                "apply_squashing": True,
                "gene_supervision": {
                    "path": str(supervision_path.resolve()),
                    "loss_weight": 1.0,
                    "coverage_loss_weight": 0.1,
                },
                "targets": rna_targets,
            },
        ]
    }
    species_dir.mkdir(parents=True, exist_ok=True)
    targets_path = species_dir / "targets.json"
    targets_path.write_text(json.dumps(targets, indent=2) + "\n")
    return targets_path, {
        "species": species,
        "groups": len(expression.groups),
        "aligned_genes": len(expression.gene_ids),
        "matched_genes": matched_genes,
        "match_rate": coverage,
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    genomes = data_root / "genomes"
    references = output_dir / "references"
    macaque_source = genomes / "GCF_003339765.1_Mmul_10_genomic.fna.gz"
    marmoset_source = genomes / "GCA_011100555.2_mCalJa1.2.pat.X_mitos2.fasta.gz"
    fasta_paths = {
        "human": args.human_fasta.expanduser().resolve(),
        "macaque": materialize_fasta(macaque_source, references / "macaque.fa"),
        "marmoset": materialize_fasta(marmoset_source, references / "marmoset.fa"),
    }
    orthologs = data_root / "RNA" / "orthologs_1to1_final.csv"
    species_settings = {
        "human": {
            "gtf": args.human_gtf.expanduser().resolve(),
            "map": None,
            "gene_attribute": "gene_id",
            "chromosome_aliases": None,
            "valid_chroms": "chr8",
            "test_chroms": "chr9",
            "exclude_chroms": "chrM,chrY",
            "include_chroms": "",
        },
        "macaque": {
            "gtf": genomes / "Macaca_mulatta.Mmul_10.115.gtf.gz",
            "map": read_ortholog_map(orthologs, "macaque_ensembl"),
            "gene_attribute": "gene_id",
            "chromosome_aliases": fasta_chromosome_aliases(macaque_source),
            "valid_chroms": fasta_chromosome_aliases(macaque_source)["8"],
            "test_chroms": fasta_chromosome_aliases(macaque_source)["9"],
            "exclude_chroms": "NC_005943.1,NC_027914.1",
            "include_chroms": ",".join(
                accession
                for chromosome, accession in fasta_chromosome_aliases(macaque_source).items()
                if chromosome not in {"Y", "MT"}
            ),
        },
        "marmoset": {
            "gtf": genomes / "GCF_011100555.1_mCalJa1.2.pat.X_mitos2.gtf.gz",
            "map": read_ortholog_map(orthologs, "marmoset_gene"),
            "gene_attribute": "gene",
            "chromosome_aliases": None,
            "valid_chroms": "chr8",
            "test_chroms": "chr9",
            "exclude_chroms": "chrM,chrY",
            "include_chroms": "",
        },
    }

    shared_group_stems = set.intersection(
        *(
            {path.stem for path in (data_root / "bigwigs" / species).glob("*.bw")}
            for species in species_settings
        )
    )
    if not shared_group_stems:
        raise ValueError("No ATAC groups are shared by all species.")

    species_entries = []
    summaries = []
    targets_paths = []
    for species, settings in species_settings.items():
        title = species.capitalize()
        targets_path, summary = build_targets(
            species=species,
            expression_path=data_root
            / "RNA"
            / f"{title}_HMBA_basalganglia_pseudobulk_aligned.h5ad",
            atac_dir=data_root / "bigwigs" / species,
            gtf=settings["gtf"],
            fasta=fasta_paths[species],
            output_dir=output_dir,
            ortholog_map=settings["map"],
            gene_attribute=settings["gene_attribute"],
            chromosome_aliases=settings["chromosome_aliases"],
            shared_group_stems=shared_group_stems,
            overwrite=args.overwrite,
        )
        species_entries.append(
            {
                "name": species,
                "fasta": str(fasta_paths[species]),
                "targets_config": str(targets_path),
                "valid_chroms": settings["valid_chroms"],
                "test_chroms": settings["test_chroms"],
                "exclude_chroms": settings["exclude_chroms"],
                "include_chroms": settings["include_chroms"],
            }
        )
        targets_paths.append(targets_path)
        summaries.append(summary)

    target_payloads = [json.loads(path.read_text()) for path in targets_paths]
    rna_targets_by_species = [payload["heads"][1]["targets"] for payload in target_payloads]
    pooled_nonzero_means = [
        sum(float(targets[channel_idx]["nonzero_mean"]) for targets in rna_targets_by_species)
        / len(rna_targets_by_species)
        for channel_idx in range(len(rna_targets_by_species[0]))
    ]
    for path, target_payload in zip(targets_paths, target_payloads, strict=True):
        for target, pooled_mean in zip(
            target_payload["heads"][1]["targets"], pooled_nonzero_means, strict=True
        ):
            target["nonzero_mean"] = pooled_mean
        path.write_text(json.dumps(target_payload, indent=2) + "\n")

    payload = {
        "species": species_entries,
        "shared_groups": len(shared_group_stems),
        "sampling": "round-robin with equal train batches per species",
        "organism_embedding": "HOMO_SAPIENS for all primates",
        "track_scaling": "channel-wise nonzero means pooled across species",
        "summary": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "species.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Species config written to {config_path}")


if __name__ == "__main__":
    main()
