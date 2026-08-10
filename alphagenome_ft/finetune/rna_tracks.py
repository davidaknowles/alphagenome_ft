"""Convert gene-level pseudobulk RNA expression into genomic BigWig targets."""

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
class GeneBody:
    gene_id: str
    chromosome: str
    start: int
    end: int
    strand: str


@dataclasses.dataclass(frozen=True)
class PseudobulkExpression:
    groups: tuple[str, ...]
    gene_ids: tuple[str, ...]
    cpm: np.ndarray


def _decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    )


def _read_h5ad_column(group: h5py.Group, name: str) -> np.ndarray:
    item = group[name]
    if isinstance(item, h5py.Dataset):
        return item[:]
    if isinstance(item, h5py.Group) and {'categories', 'codes'} <= set(item):
        categories = item['categories'][:]
        codes = item['codes'][:]
        return np.asarray([categories[code] if code >= 0 else b'' for code in codes])
    raise ValueError(f'Unsupported h5ad encoding for {group.name}/{name}.')


def read_pseudobulk_expression(path: Path, *, normalize_cpm: bool = True) -> PseudobulkExpression:
    """Read a dense group-by-gene h5ad matrix and return CPM values."""
    path = Path(path).expanduser().resolve()
    with h5py.File(path, 'r') as handle:
        matrix_node = handle['X']
        if not isinstance(matrix_node, h5py.Dataset):
            raise ValueError(f'{path} must store a dense matrix in X.')
        values = np.asarray(matrix_node[:], dtype=np.float32)
        groups = _decode(_read_h5ad_column(handle['obs'], 'Group'))
        gene_ids = _decode(_read_h5ad_column(handle['var'], 'gene_id'))

    if values.shape != (len(groups), len(gene_ids)):
        raise ValueError(
            f'Expression shape {values.shape} does not match '
            f'{len(groups)} groups and {len(gene_ids)} genes.'
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Expression values must be finite and non-negative.')
    if normalize_cpm:
        totals = values.sum(axis=1, keepdims=True, dtype=np.float64)
        if np.any(totals <= 0):
            raise ValueError('Every pseudobulk group must have positive total expression.')
        values = (values / totals * 1_000_000.0).astype(np.float32)
    return PseudobulkExpression(groups=groups, gene_ids=gene_ids, cpm=values)


_GENE_ID_RE = re.compile(r'(?:^|;\s*)gene_id\s+"([^"]+)"')


def read_gene_bodies(
    gtf_path: Path,
    *,
    gene_ids: Sequence[str],
    chromosome_sizes: Mapping[str, int],
) -> dict[str, GeneBody]:
    """Read GTF gene records for the requested stable Ensembl identifiers."""
    wanted = {gene_id.split('.', 1)[0] for gene_id in gene_ids}
    records: dict[str, GeneBody] = {}
    opener = gzip.open if str(gtf_path).endswith('.gz') else open
    with opener(gtf_path, 'rt') as handle:
        for line in handle:
            if not line or line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9 or fields[2] != 'gene':
                continue
            match = _GENE_ID_RE.search(fields[8])
            if match is None:
                continue
            gene_id = match.group(1).split('.', 1)[0]
            chromosome = fields[0]
            if gene_id not in wanted or chromosome not in chromosome_sizes:
                continue
            start = max(0, int(fields[3]) - 1)
            end = min(int(fields[4]), int(chromosome_sizes[chromosome]))
            strand = fields[6]
            if end > start and strand in {'+', '-'}:
                records[gene_id] = GeneBody(gene_id, chromosome, start, end, strand)
    return records


def _safe_track_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')


def _write_group_strand_track(
    path: Path,
    *,
    expression: np.ndarray,
    gene_ids: Sequence[str],
    gene_bodies: Mapping[str, GeneBody],
    strand: str,
    chromosome_sizes: Mapping[str, int],
) -> None:
    events: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for gene_id_raw, cpm in zip(gene_ids, expression, strict=True):
        if cpm <= 0:
            continue
        gene = gene_bodies.get(gene_id_raw.split('.', 1)[0])
        if gene is None or gene.strand != strand:
            continue
        density = float(cpm) / float(gene.end - gene.start)
        events[gene.chromosome].append((gene.start, density))
        events[gene.chromosome].append((gene.end, -density))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f'{path.name}.tmp-{os.getpid()}')
    with pyBigWig.open(str(temporary_path), 'w') as output:
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
                while idx < len(chrom_events) and chrom_events[idx][0] == position:
                    current += chrom_events[idx][1]
                    idx += 1
                if abs(current) < 1e-12:
                    current = 0.0
                previous = position
            if starts:
                output.addEntries(
                    [chromosome] * len(starts),
                    starts,
                    ends=ends,
                    values=values,
                )
    temporary_path.replace(path)


def write_stranded_gene_body_bigwigs(
    expression: PseudobulkExpression,
    *,
    gene_bodies: Mapping[str, GeneBody],
    chromosome_sizes: Mapping[str, int],
    output_dir: Path,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    """Write paired strand-specific tracks, preserving each gene's CPM integral."""
    output_dir = Path(output_dir).expanduser().resolve()
    targets: list[dict[str, str]] = []
    for group_idx, group in enumerate(expression.groups):
        stem = _safe_track_name(group)
        for strand, suffix in (('+', 'plus'), ('-', 'minus')):
            path = output_dir / f'{stem}.{suffix}.bw'
            if overwrite or not path.exists():
                _write_group_strand_track(
                    path,
                    expression=expression.cpm[group_idx],
                    gene_ids=expression.gene_ids,
                    gene_bodies=gene_bodies,
                    strand=strand,
                    chromosome_sizes=chromosome_sizes,
                )
            targets.append({'path': str(path), 'label': f'{group} ({strand})', 'strand': strand})
    return targets


__all__ = [
    'GeneBody',
    'PseudobulkExpression',
    'read_gene_bodies',
    'read_pseudobulk_expression',
    'write_stranded_gene_body_bigwigs',
]
