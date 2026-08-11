"""Shared dataset-specific helpers for Liu HDMA preprocessing."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import re
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd


@dataclasses.dataclass(frozen=True)
class ClusterSpec:
    cluster: str
    organ: str
    organ_code: str
    annotation: str
    chrombpnet_name: str
    cells: int
    released_track: bool


def _column_name(cell_reference: str) -> str:
    return re.match(r"[A-Z]+", cell_reference).group(0)


def read_inline_xlsx(path: Path) -> list[dict[str, str]]:
    """Read the first worksheet of an inline-string XLSX without an Excel dependency."""
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    raw_rows: list[dict[str, str]] = []
    for row in root.findall(".//m:sheetData/m:row", namespace):
        values: dict[str, str] = {}
        for cell in row.findall("m:c", namespace):
            inline = cell.find("m:is/m:t", namespace)
            scalar = cell.find("m:v", namespace)
            values[_column_name(cell.attrib["r"])] = (
                inline.text if inline is not None else scalar.text if scalar is not None else ""
            )
        raw_rows.append(values)
    if not raw_rows:
        raise ValueError(f"No worksheet rows found in {path}.")
    headers = raw_rows[0]
    return [
        {headers[column]: value for column, value in row.items() if column in headers}
        for row in raw_rows[1:]
    ]


def read_cluster_specs(supplementary_table: Path, bigwig_root: Path) -> tuple[ClusterSpec, ...]:
    """Read clusters in the published dendrogram order and mark released tracks."""
    released = {
        f"{organ.name}_{cluster.name}"
        for organ in bigwig_root.iterdir()
        if organ.is_dir()
        for cluster in organ.iterdir()
        if cluster.is_dir()
    }
    specs = tuple(
        ClusterSpec(
            cluster=row["Cluster"],
            organ=row["organ"],
            organ_code=row["organ_code"],
            annotation=row["L1_annot"],
            chrombpnet_name=row["Cluster_ChromBPNet"],
            cells=int(float(row["ncell"])),
            released_track=row["Cluster_ChromBPNet"] in released,
        )
        for row in read_inline_xlsx(supplementary_table)
    )
    if len(specs) != len({spec.cluster for spec in specs}):
        raise ValueError("Supplementary table contains duplicate cluster identifiers.")
    return specs


def read_cell_metadata(path: Path) -> pd.DataFrame:
    """Read cell assignments and split the compound cell identifier."""
    metadata = pd.read_csv(path)
    required = {"cb", "Cluster", "organ_code"}
    if not required <= set(metadata):
        raise ValueError(f"Cell metadata is missing columns: {sorted(required - set(metadata))}")
    identifiers = metadata["cb"].str.split("#", n=1, expand=True)
    if identifiers.shape[1] != 2 or identifiers.isna().any().any():
        raise ValueError("Every Liu cell identifier must have the form sample#barcode.")
    metadata = metadata.assign(sample=identifiers[0], barcode=identifiers[1])
    if metadata["cb"].duplicated().any():
        raise ValueError("Cell metadata contains duplicate compound identifiers.")
    return metadata


def selected_clusters(manifest: dict[str, object]) -> tuple[str, ...]:
    clusters = tuple(str(item["cluster"]) for item in manifest["selected_clusters"])
    if not clusters or len(set(clusters)) != len(clusters):
        raise ValueError("Cluster manifest must contain unique selected clusters.")
    return clusters
