#!/usr/bin/env python
"""Backfill torch W&B summary aliases and delete no-epoch runs for a group."""

from __future__ import annotations

import argparse
from typing import Any

import wandb


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="daknowles-columbia-university")
    parser.add_argument("--project", default="alphagenome-finetune")
    parser.add_argument("--group", required=True)
    parser.add_argument("--delete-no-epoch", action="store_true")
    parser.add_argument("--log-history-alias", action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", filters={"group": args.group})

    deleted = []
    updated = []
    kept = []
    history_aliases: list[tuple[str, str, float, float]] = []
    for run in runs:
        epoch = _as_float(run.summary.get("epoch"))
        if epoch is None or epoch < 1:
            if args.delete_no_epoch:
                print(f"delete no-epoch run: {run.name} {run.id} state={run.state}")
                run.delete()
                deleted.append(run.id)
            else:
                print(f"would delete no-epoch run: {run.name} {run.id} state={run.state}")
            continue

        detailed = _as_float(run.summary.get("epoch/atac_128bp_differential_pearson_r"))
        current_alias = _as_float(run.summary.get("epoch/valid_differential_pearson_r"))
        if detailed is not None and current_alias is None:
            run.summary["epoch/valid_differential_pearson_r"] = detailed
            run.summary["valid/differential_pearson_r"] = detailed
            run.summary["best/valid/differential_pearson_r"] = detailed
            run.summary.update()
            print(f"updated torch alias: {run.name} {run.id} epoch={epoch:g} value={detailed:.6g}")
            updated.append(run.id)
            history_aliases.append((run.id, run.name, epoch, detailed))
        else:
            kept.append(run.id)

    if args.log_history_alias:
        for run_id, run_name, epoch, value in history_aliases:
            print(f"log history alias: {run_name} {run_id} epoch={epoch:g} value={value:.6g}")
            run = wandb.init(
                entity=args.entity,
                project=args.project,
                id=run_id,
                resume="allow",
                name=run_name,
                reinit="finish_previous",
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "epoch/valid_differential_pearson_r": value,
                    "valid/differential_pearson_r": value,
                    "best/valid/differential_pearson_r": value,
                }
            )
            run.summary["epoch/valid_differential_pearson_r"] = value
            run.summary["valid/differential_pearson_r"] = value
            run.summary["best/valid/differential_pearson_r"] = value
            run.finish()

    print(
        "summary: "
        f"deleted={len(deleted)} updated={len(updated)} kept_or_already_ok={len(kept)}"
    )


if __name__ == "__main__":
    main()
