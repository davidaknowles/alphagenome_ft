"""Convert gene-level pseudobulk RNA expression into genomic targets."""

from __future__ import annotations

import dataclasses
import gzip
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import pyBigWig


@dataclasses.dataclass(frozen=True)
class GeneExons:
    """Union-exon annotation for one gene, using zero-based half-open intervals."""

    gene_id: str
    chromosome: str
    start: int
    end: int
    strand: str
    exons: tuple[tuple[int, int], ...]

    @property
    def exon_length(self) -> int:
        return sum(end - start for start, end in self.exons)


# Compatibility name for callers written against the initial gene-body representation.
GeneBody = GeneExons


@dataclasses.dataclass(frozen=True)
class PseudobulkExpression:
    groups: tuple[str, ...]
    gene_ids: tuple[str, ...]
    cpm: np.ndarray


def _decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values
    )


def _read_h5ad_column(group: h5py.Group, name: str) -> np.ndarray:
    item = group[name]
    if isinstance(item, h5py.Dataset):
        return item[:]
    if isinstance(item, h5py.Group) and {"categories", "codes"} <= set(item):
        categories = item["categories"][:]
        codes = item["codes"][:]
        return np.asarray([categories[code] if code >= 0 else b"" for code in codes])
    raise ValueError(f"Unsupported h5ad encoding for {group.name}/{name}.")


def read_pseudobulk_expression(
    path: Path,
    *,
    normalize_cpm: bool = True,
    gene_id_column: str = "gene_id",
) -> PseudobulkExpression:
    """Read a dense group-by-gene h5ad matrix and return CPM values."""
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        matrix_node = handle["X"]
        if not isinstance(matrix_node, h5py.Dataset):
            raise ValueError(f"{path} must store a dense matrix in X.")
        values = np.asarray(matrix_node[:], dtype=np.float32)
        groups = _decode(_read_h5ad_column(handle["obs"], "Group"))
        gene_ids = _decode(_read_h5ad_column(handle["var"], gene_id_column))

    if values.shape != (len(groups), len(gene_ids)):
        raise ValueError(
            f"Expression shape {values.shape} does not match "
            f"{len(groups)} groups and {len(gene_ids)} genes."
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Expression values must be finite and non-negative.")
    if normalize_cpm:
        totals = values.sum(axis=1, keepdims=True, dtype=np.float64)
        if np.any(totals <= 0):
            raise ValueError("Every pseudobulk group must have positive total expression.")
        values = (values / totals * 1_000_000.0).astype(np.float32)
    return PseudobulkExpression(groups=groups, gene_ids=gene_ids, cpm=values)


def remap_expression_gene_ids(
    expression: PseudobulkExpression,
    gene_id_map: Mapping[str, str],
) -> PseudobulkExpression:
    """Replace expression identifiers and retain columns with a unique mapping."""
    mapped_ids: list[str] = []
    column_indices: list[int] = []
    seen: set[str] = set()
    for column_idx, gene_id in enumerate(expression.gene_ids):
        mapped = gene_id_map.get(gene_id.split(".", 1)[0], "")
        if mapped and mapped not in seen:
            mapped_ids.append(mapped)
            column_indices.append(column_idx)
            seen.add(mapped)
    if not mapped_ids:
        raise ValueError("No expression genes matched the supplied identifier map.")
    return PseudobulkExpression(
        groups=expression.groups,
        gene_ids=tuple(mapped_ids),
        cpm=expression.cpm[:, column_indices],
    )


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


def read_gene_exons(
    gtf_path: Path,
    *,
    gene_ids: Sequence[str],
    chromosome_sizes: Mapping[str, int],
    gene_attribute: str = "gene_id",
    chromosome_aliases: Mapping[str, str] | None = None,
) -> dict[str, GeneExons]:
    """Read and merge GTF exon records for requested stable Ensembl identifiers."""
    wanted = {gene_id.split(".", 1)[0] for gene_id in gene_ids}
    gene_pattern = re.compile(rf'(?:^|;\s*){re.escape(gene_attribute)}\s+"([^"]+)"')
    chromosome_aliases = chromosome_aliases or {}
    raw: dict[str, tuple[str, str, list[tuple[int, int]]]] = {}
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "exon":
                continue
            match = gene_pattern.search(fields[8])
            if match is None:
                continue
            gene_id = match.group(1).split(".", 1)[0]
            chromosome = chromosome_aliases.get(fields[0], fields[0])
            strand = fields[6]
            if (
                gene_id not in wanted
                or chromosome not in chromosome_sizes
                or strand not in {"+", "-"}
            ):
                continue
            start = max(0, int(fields[3]) - 1)
            end = min(int(fields[4]), int(chromosome_sizes[chromosome]))
            if end <= start:
                continue
            if gene_id not in raw:
                raw[gene_id] = (chromosome, strand, [])
            old_chromosome, old_strand, exons = raw[gene_id]
            if old_chromosome != chromosome:
                # GENCODE repeats pseudoautosomal identifiers on chrY. Keep the
                # first locus so each expression row has one genomic target.
                continue
            if old_strand != strand:
                raise ValueError(f"Inconsistent chromosome or strand for gene {gene_id}.")
            exons.append((start, end))

    records: dict[str, GeneExons] = {}
    for gene_id, (chromosome, strand, exons) in raw.items():
        merged = _merge_intervals(exons)
        if merged:
            records[gene_id] = GeneExons(
                gene_id=gene_id,
                chromosome=chromosome,
                start=merged[0][0],
                end=merged[-1][1],
                strand=strand,
                exons=merged,
            )
    return records


def read_gene_bodies(
    gtf_path: Path,
    *,
    gene_ids: Sequence[str],
    chromosome_sizes: Mapping[str, int],
    gene_attribute: str = "gene_id",
    chromosome_aliases: Mapping[str, str] | None = None,
) -> dict[str, GeneExons]:
    """Compatibility wrapper; RNA annotations now contain union exons."""
    return read_gene_exons(
        gtf_path,
        gene_ids=gene_ids,
        chromosome_sizes=chromosome_sizes,
        gene_attribute=gene_attribute,
        chromosome_aliases=chromosome_aliases,
    )


def _safe_track_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _write_group_strand_track(
    path: Path,
    *,
    expression: np.ndarray,
    gene_ids: Sequence[str],
    genes: Mapping[str, GeneExons],
    strand: str,
    chromosome_sizes: Mapping[str, int],
) -> float:
    events: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for gene_id_raw, cpm in zip(gene_ids, expression, strict=True):
        if cpm <= 0:
            continue
        gene = genes.get(gene_id_raw.split(".", 1)[0])
        if gene is None or gene.strand != strand:
            continue
        density = float(cpm) / float(gene.exon_length)
        for start, end in gene.exons:
            events[gene.chromosome].append((start, density))
            events[gene.chromosome].append((end, -density))

    nonzero_sum = 0.0
    nonzero_bases = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with pyBigWig.open(str(temporary_path), "w") as output:
        output.addHeader([(chrom, int(size)) for chrom, size in chromosome_sizes.items()])
        for chromosome in chromosome_sizes:
            chrom_events = sorted(events.get(chromosome, ()))
            if not chrom_events:
                continue
            starts: list[int] = []
            ends: list[int] = []
            values: list[float] = []
            current = 0.0
            previous = chrom_events[0][0]
            idx = 0
            while idx < len(chrom_events):
                position = chrom_events[idx][0]
                if position > previous and current > 0:
                    starts.append(previous)
                    ends.append(position)
                    values.append(current)
                    length = position - previous
                    nonzero_sum += current * length
                    nonzero_bases += length
                while idx < len(chrom_events) and chrom_events[idx][0] == position:
                    current += chrom_events[idx][1]
                    idx += 1
                if abs(current) < 1e-12:
                    current = 0.0
                previous = position
            if starts:
                output.addEntries([chromosome] * len(starts), starts, ends=ends, values=values)
    temporary_path.replace(path)
    return nonzero_sum / max(nonzero_bases, 1)


def _bigwig_nonzero_mean(path: Path) -> float:
    weighted_sum = 0.0
    covered_bases = 0
    with pyBigWig.open(str(path)) as handle:
        for chromosome in handle.chroms():
            for start, end, value in handle.intervals(chromosome) or ():
                if value > 0:
                    length = end - start
                    weighted_sum += float(value) * length
                    covered_bases += length
    return weighted_sum / max(covered_bases, 1)


def write_stranded_exon_bigwigs(
    expression: PseudobulkExpression,
    *,
    genes: Mapping[str, GeneExons],
    chromosome_sizes: Mapping[str, int],
    output_dir: Path,
    overwrite: bool = False,
) -> list[dict[str, str | float]]:
    """Write paired exon-density tracks, preserving each gene's CPM integral."""
    output_dir = Path(output_dir).expanduser().resolve()
    targets: list[dict[str, str | float]] = []
    for group_idx, group in enumerate(expression.groups):
        stem = _safe_track_name(group)
        for strand, suffix in (("+", "plus"), ("-", "minus")):
            path = output_dir / f"{stem}.{suffix}.bw"
            if overwrite or not path.exists():
                nonzero_mean = _write_group_strand_track(
                    path,
                    expression=expression.cpm[group_idx],
                    gene_ids=expression.gene_ids,
                    genes=genes,
                    strand=strand,
                    chromosome_sizes=chromosome_sizes,
                )
            else:
                nonzero_mean = _bigwig_nonzero_mean(path)
            targets.append(
                {
                    "path": str(path),
                    "label": f"{group} ({strand})",
                    "strand": strand,
                    "nonzero_mean": float(nonzero_mean),
                }
            )
    return targets


def write_stranded_gene_body_bigwigs(
    expression: PseudobulkExpression,
    *,
    gene_bodies: Mapping[str, GeneExons],
    chromosome_sizes: Mapping[str, int],
    output_dir: Path,
    overwrite: bool = False,
) -> list[dict[str, str | float]]:
    """Compatibility wrapper for exon-density target generation."""
    return write_stranded_exon_bigwigs(
        expression,
        genes=gene_bodies,
        chromosome_sizes=chromosome_sizes,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def write_gene_expression_supervision(
    path: Path,
    expression: PseudobulkExpression,
    *,
    genes: Mapping[str, GeneExons],
) -> int:
    """Write exon geometry and matched group-by-gene CPM to a compact NPZ."""
    expression_index = {
        gene_id.split(".", 1)[0]: idx for idx, gene_id in enumerate(expression.gene_ids)
    }
    ordered = sorted(
        genes.values(), key=lambda gene: (gene.chromosome, gene.start, gene.end, gene.gene_id)
    )
    exon_starts: list[int] = []
    exon_ends: list[int] = []
    exon_offsets = [0]
    cpm_columns = []
    for gene in ordered:
        exon_starts.extend(start for start, _ in gene.exons)
        exon_ends.extend(end for _, end in gene.exons)
        exon_offsets.append(len(exon_starts))
        cpm_columns.append(expression.cpm[:, expression_index[gene.gene_id]])

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gene_ids=np.asarray([gene.gene_id for gene in ordered]),
        chromosomes=np.asarray([gene.chromosome for gene in ordered]),
        starts=np.asarray([gene.start for gene in ordered], dtype=np.int64),
        ends=np.asarray([gene.end for gene in ordered], dtype=np.int64),
        strands=np.asarray([gene.strand for gene in ordered]),
        exon_offsets=np.asarray(exon_offsets, dtype=np.int64),
        exon_starts=np.asarray(exon_starts, dtype=np.int64),
        exon_ends=np.asarray(exon_ends, dtype=np.int64),
        groups=np.asarray(expression.groups),
        cpm=np.stack(cpm_columns, axis=1).astype(np.float32),
    )
    return len(ordered)


__all__ = [
    "GeneBody",
    "GeneExons",
    "PseudobulkExpression",
    "read_gene_bodies",
    "read_gene_exons",
    "read_pseudobulk_expression",
    "remap_expression_gene_ids",
    "write_gene_expression_supervision",
    "write_stranded_exon_bigwigs",
    "write_stranded_gene_body_bigwigs",
]
