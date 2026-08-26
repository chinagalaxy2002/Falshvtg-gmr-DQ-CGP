#!/usr/bin/env python3
"""Apply a monotonic logit-bias calibration to GMR existence scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def calibrate_probability(probability: float, logit_bias: float, eps: float) -> float:
    probability = min(max(float(probability), eps), 1.0 - eps)
    logit = math.log(probability / (1.0 - probability))
    return 1.0 / (1.0 + math.exp(-(logit + logit_bias)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw prediction JSONL")
    parser.add_argument("--output", required=True, help="Calibrated prediction JSONL")
    parser.add_argument(
        "--logit_bias",
        type=float,
        default=-0.34110591,
        help="Validation-fitted bias added to the existence logit",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if "pred_exist_score" not in record:
                raise KeyError(
                    f"record qid={record.get('qid')} has no pred_exist_score"
                )
            record["pred_exist_score"] = calibrate_probability(
                record["pred_exist_score"], args.logit_bias, args.eps
            )
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(
        f"calibrated {count} predictions with logit_bias={args.logit_bias:.8f}: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
