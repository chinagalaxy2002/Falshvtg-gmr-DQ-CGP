#!/usr/bin/env python3
"""Summarize GMR metrics evaluated at existence thresholds 0.4/0.5/0.6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THRESHOLDS = ("0.4", "0.5", "0.6")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    loaded = {
        threshold: json.loads(
            (metrics_dir / f"metrics_tau_{threshold}.json").read_text()
        )["brief"]
        for threshold in THRESHOLDS
    }
    first = loaded[THRESHOLDS[0]]
    per_threshold = {
        threshold: {
            "Rej-F1": loaded[threshold][f"Rej-F1@{threshold}"],
            "Acc": loaded[threshold][f"Acc@{threshold}"],
            "G-mIoU@1": loaded[threshold]["G-mIoU@1"],
            "G-mIoU@3": loaded[threshold]["G-mIoU@3"],
            "G-mIoU@5": loaded[threshold]["G-mIoU@5"],
        }
        for threshold in THRESHOLDS
    }
    mean_keys = ("Rej-F1", "Acc", "G-mIoU@1", "G-mIoU@3", "G-mIoU@5")
    threshold_mean = {
        key: round(
            sum(per_threshold[threshold][key] for threshold in THRESHOLDS)
            / len(THRESHOLDS),
            2,
        )
        for key in mean_keys
    }
    independent_keys = (
        "AUROC",
        "mAP",
        "mR@1",
        "mR@3",
        "mR@5",
        "mR+@1",
        "mR+@3",
        "mR+@5",
        "mIoU@1",
        "mIoU@3",
        "mIoU@5",
        "mIoU+@1",
        "mIoU+@3",
        "mIoU+@5",
    )
    summary = {
        "protocol": {
            "thresholds": [float(value) for value in THRESHOLDS],
            "threshold_average": "arithmetic mean over 0.4, 0.5, and 0.6",
        },
        "threshold_independent": {key: first[key] for key in independent_keys},
        "per_threshold": per_threshold,
        "threshold_mean": threshold_mean,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
