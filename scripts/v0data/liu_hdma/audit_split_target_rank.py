#!/usr/bin/env python3
"""Liu entry point for the reusable held-out RNA target-rank audit."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v0data.audit_split_target_rank import (
    audit_chromosomes,
    main as _main,
    render_markdown as _render_markdown,
)


def render_markdown(result: dict[str, object]) -> str:
    return _render_markdown(result, "Liu")


def main() -> None:
    _main(default_dataset_label="Liu")


if __name__ == "__main__":
    main()
