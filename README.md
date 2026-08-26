# FlashVTG-GMR with DQ-CGP

本仓库提供 FlashVTG-DQ-CGP 的完整训练、推理、三阈值 GMR 评测代码，以及已经训练好的
checkpoint。实现参考 DQ-CGP 在 Moment-DETR 中的 candidate-wise 设计，但针对 FlashVTG
没有 DETR decoder queries 的结构差异，将每个原生多尺度 dense point 视为一个候选实例。

本发布版使用 Soccer-GMR Standard split，从头联合训练 FlashVTG、existence head 和
DQ-CGP。原始 FlashVTG-GMR baseline 保留在 `models/flash_vtg_gmr`，没有修改。

## 1. 方法概览

```text
video/text features
  -> FlashVTG text-to-video fusion
  -> native multi-scale dense candidates
  -> candidate-local temporal binding
  -> per-candidate RCG basis routing
  -> BPS/FRF residual refinement
  -> original FlashVTG class/coordinate heads
```

DQ-CGP 概念到 FlashVTG 的映射如下：

| DQ-CGP 概念 | FlashVTG-DQ-CGP 实现 |
|---|---|
| object/query instance | 原生多尺度 dense point candidate |
| local visual feature | candidate-centered temporal context |
| static semantic feature | masked-pooled CLIP query semantic |
| per-instance routing | 每个 dense candidate 独立的 basis weights |
| adapted feature | locality-gated FRF residual |

本版本包含 candidate locality、candidate-valid gate 和 prompt attention，并使用较小的辅助
loss 权重。训练时所有参数均可学习，不冻结 FlashVTG baseline。

## 2. 发布 checkpoint

Checkpoint 作为 GitHub Release asset 发布：

```text
checkpoints/flashvtg_dq_cgp_gmr_best_epoch32.ckpt
```

- checkpoint 内部 epoch：`32`（第 33 个 epoch）
- seed：`2024`
- 最大训练轮数：`400`
- early-stop patience：`80`
- 实际 early stop：epoch `113`
- checkpoint-selection：validation no-NMS `MR-full-mAP`
- SHA256：`0a5674508577dfedc4a3db0ae1c59a8ed9adf061a3fe54876b77ef8e8d51369c`

下载：

```bash
mkdir -p checkpoints
wget -O checkpoints/flashvtg_dq_cgp_gmr_best_epoch32.ckpt \
  https://github.com/chinagalaxy2002/Falshvtg-gmr-DQ-CGP/releases/download/v1.0.0/flashvtg_dq_cgp_gmr_best_epoch32.ckpt

cd checkpoints
sha256sum -c SHA256SUMS
cd ..
```

也可以使用 GitHub CLI：

```bash
gh release download v1.0.0 \
  --pattern flashvtg_dq_cgp_gmr_best_epoch32.ckpt \
  --dir checkpoints
```

## 3. 复现结果

### 3.1 Validation checkpoint-selection

| Metric | no NMS | NMS=0.7 |
|---|---:|---:|
| MR-full-mAP | 25.91 | **27.82** |
| MR-full-R1@0.5 | 42.35 | 42.35 |
| MR-full-R1@0.7 | 25.88 | 25.88 |
| MR-full-mAP@0.5 | 46.02 | 49.48 |
| MR-full-mAP@0.75 | 27.78 | 30.29 |
| MR-full-mIoU | 35.71 | 36.02 |

### 3.2 Standard test

测试保留 top-10 NMS predictions，分别使用 existence thresholds `0.4`、`0.5`、`0.6`。
表中的 `Mean` 是三个阈值的算术平均；AUROC、mAP、mR 和 mIoU 与 gate threshold 无关。

`Calibrated` 只对 existence probability 使用 validation 拟合的单调变换：

```text
calibrated_score = sigmoid(logit(raw_score) - 0.34110591)
```

它不改变候选窗口、候选排序或 AUROC。

| Metric | Matched FlashVTG scratch | DQ-CGP raw | DQ-CGP calibrated |
|---|---:|---:|---:|
| AUROC | 73.74 | **75.89** | **75.89** |
| Mean Rej-F1 | **66.47** | 56.90 | 66.12 |
| mAP | 25.42 | **25.63** | **25.63** |
| mR@5 | 34.46 | **35.45** | **35.45** |
| mR+@3 | 8.16 | **11.80** | **11.80** |
| mR+@5 | 17.44 | **20.44** | **20.44** |
| mIoU@1 | **34.04** | 33.49 | 33.49 |
| mIoU@3 | **30.68** | 30.40 | 30.40 |
| mIoU@5 | **30.66** | 30.26 | 30.26 |
| Mean G-mIoU@1 | **46.08** | 37.09 | 44.10 |
| Mean G-mIoU@3 | **40.37** | 30.89 | 38.28 |
| Mean G-mIoU@5 | **38.85** | 28.73 | 36.41 |

校准后的逐阈值结果：

| Threshold | Rej-F1 | Accuracy | G-mIoU@1 | G-mIoU@3 | G-mIoU@5 |
|---:|---:|---:|---:|---:|---:|
| 0.4 | 57.11 | 64.77 | 35.71 | 29.37 | 27.12 |
| 0.5 | 68.12 | 67.47 | 45.41 | 39.69 | 37.81 |
| 0.6 | 73.13 | 69.21 | 51.17 | 45.77 | 44.31 |
| **Mean** | **66.12** | **67.15** | **44.10** | **38.28** | **36.41** |

主要观察：DQ-CGP 将 test `mR+@5` 从 `17.44` 提高到 `20.44`，同时 mAP、mR@5
和 AUROC 小幅提高。单一 validation-fitted bias 将 Mean Rej-F1 提高 `9.22`，并将 Mean
G-mIoU@1/@3/@5 分别提高 `7.01/7.39/7.68`。当前版本仍未超过 matched baseline 的
Mean G-mIoU，因此结果支持“多 occurrence 检索改善”，但不宣称端到端 GMR 已全面领先。

详细结果：

- [calibrated test summary](results/test/calibrated/summary_three_thresholds.json)
- [raw test summary](results/test/raw/summary_three_thresholds.json)
- [matched baseline summary](results/test/matched_baseline/summary_three_thresholds.json)
- [best validation metrics](results/validation/best_nms_metrics.json)

## 4. 环境

本结果实际运行环境：

```text
Python 3.13.5
PyTorch 2.10.0+cu128
CUDA 12.8
NumPy 2.4.2
SciPy 1.17.1
nncore 0.4.7
GPU: NVIDIA GeForce RTX 3090
```

安装：

```bash
conda create -n flashvtg-dq-cgp python=3.10 -y
conda activate flashvtg-dq-cgp

# 先按本机 CUDA 版本安装 PyTorch，再安装其余依赖。
pip install -r requirements-flash-vtg.txt
```

不同 PyTorch/CUDA 版本可能造成小幅数值波动。

## 5. Soccer-GMR 特征

Soccer-GMR 视频和预计算特征受 gated access、NDA 和禁止再分发条款约束，本仓库不包含这些
资产。请从 Soccer-GMR 官方 release 获取特征，并组织为：

```text
Soccergmr/
├── clip/
│   └── <video_id>.npz
├── slowfast/
│   └── <video_id>.npz
└── clip_text/
    └── qid<query_id>.npz
```

Standard split labels 位于：

```text
data/label/Standard/train.jsonl
data/label/Standard/val.jsonl
data/label/Standard/test.jsonl
```

## 6. 从头训练

以下命令复现发布 checkpoint 的配置：seed 2024、batch size 8、最多 400 epochs、patience
80、全网络联合训练，不加载 baseline checkpoint。

```bash
DQ_PYTHON=python \
DQ_GPU=0 \
DQ_EPOCHS=400 \
DQ_OUTPUT=outputs/flashvtg_dq_cgp_seed2024 \
bash scripts/train_flashvtg_dq_cgp.sh
```

自定义特征目录：

```bash
DQ_TEXT_FEATURES=/path/to/clip_text \
DQ_CLIP_FEATURES=/path/to/clip \
DQ_SLOWFAST_FEATURES=/path/to/slowfast \
bash scripts/train_flashvtg_dq_cgp.sh
```

训练输出目录名称包含时间戳，其中包括：

```text
model_best.ckpt
model_latest.ckpt
opt.json
train.log.txt
eval.log.txt
best_hl_val_preds.jsonl
best_hl_val_preds_nms_thd_0.7.jsonl
```

## 7. 使用发布 checkpoint 测试

下载 checkpoint 后运行 Standard test。脚本会执行一次模型推理、existence calibration、
三个阈值的完整 GMR 评测和均值汇总：

```bash
DQ_PYTHON=python \
DQ_GPU=0 \
DQ_SPLIT=test \
DQ_CHECKPOINT=checkpoints/flashvtg_dq_cgp_gmr_best_epoch32.ckpt \
DQ_OUTPUT=outputs/flashvtg_dq_cgp_test \
bash scripts/evaluate_flashvtg_dq_cgp.sh
```

输出：

```text
outputs/flashvtg_dq_cgp_test/
├── raw/
│   └── hl_test_submission_nms_thd_0.7.jsonl
├── calibrated/
│   ├── hl_test_submission_nms_thd_0.7.jsonl
│   ├── metrics_tau_0.4.json
│   ├── metrics_tau_0.5.json
│   └── metrics_tau_0.6.json
└── summary_three_thresholds.json
```

运行 raw-score 反事实：

```bash
DQ_EXIST_LOGIT_BIAS=0 \
DQ_OUTPUT=outputs/flashvtg_dq_cgp_test_raw \
bash scripts/evaluate_flashvtg_dq_cgp.sh
```

在 full validation 上评估：

```bash
DQ_SPLIT=val \
DQ_OUTPUT=outputs/flashvtg_dq_cgp_full_val \
bash scripts/evaluate_flashvtg_dq_cgp.sh
```

## 8. 测试代码

运行 DQ-CGP 单元测试：

```bash
python -m unittest discover \
  -s models/flashvtg_dq-cgp-gmr-v2/tests \
  -v
```

检查脚本和 checkpoint：

```bash
bash -n scripts/train_flashvtg_dq_cgp.sh
bash -n scripts/evaluate_flashvtg_dq_cgp.sh
python -m compileall models training eval scripts

cd checkpoints
sha256sum -c SHA256SUMS
```

## 9. 代码结构

```text
models/flashvtg_dq-cgp-gmr-v2/dq_cgp.py  # temporal binding, RCG, BPS, FRF
models/flashvtg_dq-cgp-gmr-v2/model.py   # FlashVTG integration and losses
models/flashvtg_dq-cgp-gmr-v2/run.py     # independent launcher
models/flash_vtg_gmr/                    # unmodified matched baseline model
training/flash_vtg_gmr/                  # training and inference loops
eval/                                    # full Soccer-GMR evaluation
scripts/train_flashvtg_dq_cgp.sh         # exact scratch-training command
scripts/evaluate_flashvtg_dq_cgp.sh      # test + calibration + 3-threshold eval
```

## 10. 可复现性说明

- validation 只用于 checkpoint selection 和 existence bias calibration；test 不参与拟合。
- calibrated 与 raw 结果来自同一个 checkpoint 和同一组 temporal windows。
- calibration 是单调变换，因此 AUROC、mAP、mR、mR+、mIoU 和 mIoU+ 保持不变。
- matched baseline 和 DQ-CGP 使用相同 Standard split、特征、seed、optimizer、训练轮数上限和
  checkpoint-selection metric；唯一主要模型差异是 DQ-CGP 分支及其辅助 losses。
- 所有 test prediction 覆盖 Standard test 的 1,036 个 query-video pairs。
- 默认 FlashVTG inference 使用 cuDNN benchmark；不同运行的窗口分数可能在浮点末位有差异，
  本机重复运行得到相同的两位小数汇总指标。

## Citation

Soccer-GMR benchmark：

```bibtex
@article{ding2026retrieving,
  title={Retrieving Any Relevant Moments: Benchmark and Models for Generalized Moment Retrieval},
  author={Ding, Yiming and Cao, Siyu and Jiao, Luyuan and Li, Yixuan and Wang, Zitong and Liu, Zhiyong and Zhang, Lu},
  journal={arXiv preprint arXiv:2605.02623},
  year={2026}
}
```

## License

仓库代码遵循 [MIT License](LICENSE)。FlashVTG 派生组件的说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。Soccer-GMR 数据、视频和特征遵循其各自的
访问协议、NDA 和版权条款。
