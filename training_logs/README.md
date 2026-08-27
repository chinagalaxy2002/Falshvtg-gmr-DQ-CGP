# Training logs

This directory contains native training, validation, and test logs for the
Soccer-GMR runs used in this repository. It intentionally excludes checkpoints,
TensorBoard events, and terminal-progress archives.

| Directory | Run |
| --- | --- |
| `baseline/` | Matched unmodified FlashVTG-GMR baseline, seed 2024 |
| `dq_cgp/` | DQ-CGP warm-start run, seed 2024 |
| `dq_cgp_v2/` | DQ-CGPv2 scratch run, seed 2024 |
| `flashvtg_dq_cgp_v3_topk4_seed2024/` | Sparse top-k4 DQ-CGP v3 scratch run and reproduced test, seed 2024 |

The run configuration and model definitions are in the repository; see the
training instructions in the root README for the reproducible command.
