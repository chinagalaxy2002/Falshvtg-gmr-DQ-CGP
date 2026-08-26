# Training logs

This directory contains the native training (`train.log.txt`) and validation
(`eval.log.txt`) logs for the three Soccer-GMR runs used in this release.  It
intentionally excludes checkpoints, prediction files, TensorBoard events, and
terminal-progress logs.

| Directory | Run |
| --- | --- |
| `baseline/` | Matched unmodified FlashVTG-GMR baseline, seed 2024 |
| `dq_cgp/` | DQ-CGP warm-start run, seed 2024 |
| `dq_cgp_v2/` | DQ-CGPv2 scratch run, seed 2024 |

The run configuration and model definitions are in the repository; see the
training instructions in the root README for the reproducible command.
