# FlashVTG-GMR with sparse DQ-CGP v3

本仓库提供 `flashvtg_dq_cgp_v3_topk4_seed2024` 的完整模型、训练、推理和 Soccer-GMR
三阈值评测代码，以及本次训练日志、测试预测和指标文件。实现参考 Moment-DETR 版本 DQ-CGP
的 candidate-wise 设计：Moment-DETR 中每个 DETR query 是一个候选实例；FlashVTG 没有
DETR decoder query，因此本实现将每个原生多尺度 dense point 作为候选实例。

本次增量发布不包含 checkpoint、TensorBoard event、特征或数据资产。测试命令需要使用者提供
自己训练得到的 `model_best.ckpt`；仓库内已提交与该 checkpoint 配套的 `opt.json`、训练/验证
日志以及完整 Standard test 结果。

## 1. 方法

```text
video/text features
  -> FlashVTG text-to-video fusion
  -> native multi-scale dense candidates
  -> candidate-wise temporal binding
  -> query-conditioned global/local prototypes
  -> sparse differentiable top-4 basis routing
  -> shared prompt bank + prompt-token attention
  -> point-wise FRF residual
  -> original FlashVTG class/coordinate heads
```

| Moment-DETR DQ-CGP 概念 | FlashVTG sparse DQ-CGP v3 |
|---|---|
| DETR query / candidate | 多尺度 dense point candidate |
| candidate local feature | candidate-centered temporal context |
| static semantic feature | masked-pooled CLIP query semantic |
| per-query basis routing | level route 与 point route 的 candidate-wise 稀疏组合 |
| decoder-state refinement | 注入原 FlashVTG head 前的 FRF residual |

v3 使用 16 个 basis、6 个 prompt token、top-k 4 路由、`0.9` level route 与 `0.1`
point route 的概率空间混合，以及固定 `beta=0.05` 的 residual。它额外使用 binding loss、
route load/entropy loss 和弱 relation loss，以缓解 basis collapse，同时保留原 FlashVTG 的
head、dense point 生成方式和 NMS 流程。具体设计见
[ANTI_COLLAPSE_DESIGN.md](models/flashvtg_dq_cgp_v3_gmr/ANTI_COLLAPSE_DESIGN.md)。

## 2. 实验配置

| 配置项 | 值 |
|---|---:|
| Dataset / split | Soccer-GMR Standard |
| Seed | 2024 |
| Batch size | 8 |
| Learning rate | 3e-5 |
| Weight decay | 1e-4 |
| Maximum epochs | 400 |
| Early-stop patience | 80 |
| Checkpoint selection | validation no-NMS `MR-full-mAP` |
| Best epoch | 60（checkpoint 内部 `epoch=59`） |
| Actual stop | 完成第 141 轮后（日志内部 `epoch=140`） |
| Learnable parameters | 11.646 M（100%） |
| NMS threshold / kept windows | 0.7 / top 10 |

保存的实际参数文件为
[opt.json](results/flashvtg_dq_cgp_v3_topk4_seed2024/validation/opt.json)。

## 3. 与论文及论文复现结果对比

对照对象统一为 Soccer-GMR Standard split 上的 **FlashVTG-GMR**，而不是不含 existence
adapter 的原始 FlashVTG：

- `论文报告`：Ding et al. 的 [论文 Table 2](https://arxiv.org/pdf/2605.02623)及
  [官方仓库 Main Results](https://github.com/dymm9977/generalized-moment-retrieval#main-results)；
  主表采用 `τ=0.4`，附录 Table 8 采用 `τ∈{0.4,0.6,0.8}`。
- `论文复现`：我们使用相同 Standard split 和 FlashVTG-GMR 代码、seed 2024 从头训练的
  matched baseline；测试结果位于 `results/test/matched_baseline/`。
- `DQ-CGP v3 raw`：`flashvtg_dq_cgp_v3_topk4_seed2024` 的原始 existence score。
- `DQ-CGP v3 calibrated`：同一个 v3 checkpoint、同一组窗口，只对 existence score 使用
  validation 拟合的单调校准。

论文未报告的项目统一写作 `—`，不通过其他表格或近似值补填。

### 3.1 Standard validation

论文只报告 test set，没有给出 validation 指标。因此下表同时保留论文列并明确标记为 `—`，
比较我们对论文 FlashVTG-GMR 的复现与 DQ-CGP v3。v3 最佳 checkpoint 来自第 60 轮，按
no-NMS `MR-full-mAP=26.03` 选择。

| Metric | 论文报告 | 论文复现 no NMS | v3 no NMS | 论文复现 NMS=0.7 | v3 NMS=0.7 |
|---|---:|---:|---:|---:|---:|
| MR-full-R1@0.3 | — | **52.55** | 47.84 | **50.98** | 47.45 |
| MR-full-R1@0.5 | — | **43.92** | 41.18 | **41.96** | 41.18 |
| MR-full-R1@0.7 | — | 25.49 | **26.27** | 23.92 | **26.27** |
| MR-full-mAP | — | **26.98** | 26.03 | **28.33** | 27.91 |
| MR-full-mAP@0.5 | — | **49.05** | 45.91 | **50.96** | 49.34 |
| MR-full-mAP@0.75 | — | **28.88** | 28.74 | **31.06** | 30.88 |
| MR-full-mIoU | — | **37.02** | 35.12 | **35.48** | 34.96 |
| GMR-TPR | — | 51.76 | **63.53** | 51.76 | **63.53** |
| GMR-TNR | — | **82.38** | 69.05 | **82.38** | 69.05 |
| GMR-BalancedAcc | — | **67.07** | 66.29 | **67.07** | 66.29 |

### 3.2 Standard test：严格对齐论文主表（τ=0.4）

以下八项与论文 Table 2 的列完全一致。`v3 raw` 是无需校准的直接模型结果，适合作为主要
方法对比；`v3 calibrated` 展示 validation-fitted operating point 的效果。

| Model | AUROC | Rej-F1 | mAP | mR@1 | mR@5 | mR+@5 | G-mIoU@1 | G-mIoU@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 论文 FlashVTG-GMR | 74.00 | 61.72 | 24.62 | 15.08 | 33.36 | 19.10 | 39.58 | 33.53 |
| 我们复现 FlashVTG-GMR | 73.74 | 52.83 | 25.42 | 15.94 | 34.46 | 17.44 | 33.20 | 26.70 |
| DQ-CGP v3 raw | **76.02** | 61.13 | **26.13** | **16.21** | **36.03** | **21.04** | 38.59 | 32.24 |
| DQ-CGP v3 calibrated | **76.02** | **66.01** | **26.13** | **16.21** | **36.03** | **21.04** | **43.13** | **37.12** |

相对提升如下；单位均为百分点：

| Comparison | AUROC | Rej-F1 | mAP | mR@1 | mR@5 | mR+@5 | G-mIoU@1 | G-mIoU@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 raw − 论文 | +2.02 | -0.59 | +1.51 | +1.13 | +2.67 | +1.94 | -0.99 | -1.29 |
| v3 calibrated − 论文 | +2.02 | +4.29 | +1.51 | +1.13 | +2.67 | +1.94 | +3.55 | +3.59 |
| v3 raw − 我们复现 | +2.28 | +8.30 | +0.71 | +0.27 | +1.57 | +3.60 | +5.39 | +5.54 |
| v3 calibrated − 我们复现 | +2.28 | +13.18 | +0.71 | +0.27 | +1.57 | +3.60 | +9.93 | +10.42 |

### 3.3 Standard test：论文未报告的扩展指标（τ=0.4）

这些指标由官方评测代码输出，但论文 Table 2 没有对应列。为保持完整性，仍展示论文复现与
v3 的所有结果，并将论文值明确标记为 `—`。

| Metric | 论文报告 | 我们复现 FlashVTG-GMR | v3 raw | v3 calibrated |
|---|---:|---:|---:|---:|
| Accuracy | — | 62.26 | 66.12 | **66.89** |
| mR@3 | — | 27.33 | **28.42** | **28.42** |
| mR+@1 | — | 0.00 | 0.00 | 0.00 |
| mR+@3 | — | 8.16 | **9.34** | **9.34** |
| mIoU@1 | — | 34.04 | **34.88** | **34.88** |
| mIoU@3 | — | 30.68 | **32.07** | **32.07** |
| mIoU@5 | — | 30.66 | **31.94** | **31.94** |
| mIoU+@1 | — | 0.00 | 0.00 | 0.00 |
| mIoU+@3 | — | 12.17 | **14.40** | **14.40** |
| mIoU+@5 | — | 12.24 | **14.06** | **14.06** |
| G-mIoU@5 | — | 24.64 | 30.11 | **35.24** |

### 3.4 论文 Table 8 阈值敏感性复现

此表严格使用论文附录协议 `τ∈{0.4,0.6,0.8}`；`AP` 是这三个阈值的算术平均，不与下一节
`τ∈{0.4,0.5,0.6}` 的 Mean 混用。

| Model | Rej-F1@0.4 | @0.6 | @0.8 | AP | G-mIoU@1 (τ=0.4) | @0.6 | @0.8 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 论文 FlashVTG-GMR | 61.72 | 73.06 | 74.63 | 70.94 | 39.58 | 51.38 | 54.13 | 49.43 |
| 我们复现 FlashVTG-GMR | 52.83 | **74.11** | **75.04** | 67.33 | 33.20 | **53.75** | **55.16** | 47.37 |
| DQ-CGP v3 raw | 61.13 | 71.60 | 74.43 | 69.05 | 38.59 | 49.35 | 53.96 | 47.30 |
| DQ-CGP v3 calibrated | **66.01** | 72.49 | **75.04** | **71.18** | **43.13** | 51.25 | 54.80 | **49.73** |

### 3.5 本仓库完整三阈值结果（τ=0.4/0.5/0.6）

本仓库发布的测试产物使用 `0.4/0.5/0.6`，表中 `Mean` 是三者的算术平均。论文没有报告
`τ=0.5`，也没有报告这一阈值集合的均值；对应位置以 `—` 标记。论文在 `τ=0.4/0.6`
报告了哪些指标，就在同一张表中列出哪些指标。

| Model / score | Threshold | Rej-F1 | Accuracy | G-mIoU@1 | G-mIoU@3 | G-mIoU@5 |
|---|---:|---:|---:|---:|---:|---:|
| 论文 FlashVTG-GMR | 0.4 | 61.72 | — | 39.58 | 33.53 | — |
| 论文 FlashVTG-GMR | 0.5 | — | — | — | — | — |
| 论文 FlashVTG-GMR | 0.6 | 73.06 | — | 51.38 | — | — |
| 论文 FlashVTG-GMR | **Mean** | — | — | — | — | — |
| 我们复现 FlashVTG-GMR | 0.4 | 52.83 | 62.26 | 33.20 | 26.70 | 24.64 |
| 我们复现 FlashVTG-GMR | 0.5 | 72.46 | 67.86 | 51.30 | 45.89 | 44.55 |
| 我们复现 FlashVTG-GMR | 0.6 | 74.11 | 68.24 | 53.75 | 48.52 | 47.35 |
| 我们复现 FlashVTG-GMR | **Mean** | **66.47** | **66.12** | **46.08** | **40.37** | **38.85** |
| DQ-CGP v3 raw | 0.4 | 61.13 | 66.12 | 38.59 | 32.24 | 30.11 |
| DQ-CGP v3 raw | 0.5 | 67.25 | 67.47 | 44.27 | 38.30 | 36.44 |
| DQ-CGP v3 raw | 0.6 | 71.60 | 68.53 | 49.35 | 43.86 | 42.27 |
| DQ-CGP v3 raw | **Mean** | **66.66** | **67.37** | **44.07** | **38.13** | **36.27** |
| DQ-CGP v3 calibrated | 0.4 | 66.01 | 66.89 | 43.13 | 37.12 | 35.24 |
| DQ-CGP v3 calibrated | 0.5 | 71.30 | 68.53 | 48.96 | 43.35 | 41.75 |
| DQ-CGP v3 calibrated | 0.6 | 72.49 | 68.05 | 51.25 | 45.85 | 44.43 |
| DQ-CGP v3 calibrated | **Mean** | **69.93** | **67.82** | **47.78** | **42.11** | **40.47** |

校准公式为：

```text
calibrated_score = sigmoid(logit(raw_score) - 0.34110591)
```

它不改变窗口、窗口排序、AUROC、mAP、mR、mR+、mIoU 或 mIoU+。因此，v3 相对论文和
论文复现的 localization 提升来自模型本身；校准只影响 Rej-F1、Accuracy 和 G-mIoU。

详细产物：

- [论文复现 test summary](results/test/matched_baseline/summary_three_thresholds.json)
- [论文复现 validation metrics, NMS](results/paper_reproduction/validation/best_nms_metrics.json)
- [论文复现 validation metrics, no NMS](results/paper_reproduction/validation/best_no_nms_metrics.json)
- [论文 Table 8 协议对照汇总](results/paper_reproduction/test/table8_threshold_comparison.json)
- [v3 calibrated test summary](results/flashvtg_dq_cgp_v3_topk4_seed2024/test/calibrated/summary_three_thresholds.json)
- [v3 raw test summary](results/flashvtg_dq_cgp_v3_topk4_seed2024/test/raw/summary_three_thresholds.json)
- [v3 validation metrics, NMS](results/flashvtg_dq_cgp_v3_topk4_seed2024/validation/best_nms_metrics.json)
- [v3 validation metrics, no NMS](results/flashvtg_dq_cgp_v3_topk4_seed2024/validation/best_no_nms_metrics.json)

## 4. 环境

本结果实际运行环境：

```text
Python 3.13.5
PyTorch 2.10.0+cu128
CUDA 12.8
NumPy 2.4.2
SciPy 1.17.1
nncore 0.4.7
GPU: NVIDIA GeForce RTX 3090 (24 GiB)
```

建议先按本机 CUDA 版本安装 PyTorch，再安装其余依赖：

```bash
conda create -n flashvtg-dq-cgp python=3.10 -y
conda activate flashvtg-dq-cgp
pip install -r requirements-flash-vtg.txt
```

不同 PyTorch/CUDA 版本可能造成小幅数值波动。

## 5. Soccer-GMR 数据与特征

Soccer-GMR 视频和预计算特征受 gated access、NDA 和禁止再分发条款约束，本仓库不包含这些
资产。请从 Soccer-GMR 官方 release 获取，并组织为：

```text
Soccergmr/
├── clip/
│   └── <video_id>.npz
├── slowfast/
│   └── <video_id>.npz
└── clip_text/
    └── qid<query_id>.npz
```

Standard split labels 已包含在：

```text
data/label/Standard/train.jsonl
data/label/Standard/val.jsonl
data/label/Standard/test.jsonl
```

## 6. 从头训练

以下命令复现 `flashvtg_dq_cgp_v3_topk4_seed2024`：seed 2024、batch size 8、最多
400 epochs、patience 80，FlashVTG、existence head 和 DQ-CGP v3 从头联合训练。

```bash
DQ_V3_PYTHON=python \
DQ_V3_GPU=0 \
DQ_V3_EPOCHS=400 \
DQ_V3_OUTPUT=outputs/flashvtg_dq_cgp_v3_topk4_seed2024 \
bash scripts/train_flashvtg_dq_cgp_v3.sh
```

若特征不在默认目录：

```bash
DQ_V3_TEXT_FEATURES=/path/to/clip_text \
DQ_V3_CLIP_FEATURES=/path/to/clip \
DQ_V3_SLOWFAST_FEATURES=/path/to/slowfast \
bash scripts/train_flashvtg_dq_cgp_v3.sh
```

训练会创建带时间戳的目录：

```text
outputs/flashvtg_dq_cgp_v3_topk4_seed2024/
└── hl-video_tef-soccer_gmr-<timestamp>/
    ├── model_best.ckpt
    ├── model_latest.ckpt
    ├── opt.json
    ├── train.log.txt
    ├── eval.log.txt
    ├── best_hl_val_preds.jsonl
    └── best_hl_val_preds_nms_thd_0.7.jsonl
```

## 7. 测试

本仓库不上传 checkpoint。训练完成后，将 `DQ_V3_CHECKPOINT` 指向本地最佳 checkpoint；
`DQ_V3_OPT` 可指向同一训练目录下的 `opt.json`，或使用仓库提交的实际参数文件：

```bash
DQ_V3_PYTHON=python \
DQ_V3_GPU=0 \
DQ_V3_SPLIT=test \
DQ_V3_CHECKPOINT=/path/to/model_best.ckpt \
DQ_V3_OPT=results/flashvtg_dq_cgp_v3_topk4_seed2024/validation/opt.json \
DQ_V3_OUTPUT=outputs/flashvtg_dq_cgp_v3_topk4_test \
bash scripts/evaluate_flashvtg_dq_cgp_v3.sh
```

该脚本依次执行一次 v3 推理、NMS、existence calibration、raw/calibrated 各三个阈值的
完整 GMR 评测以及均值汇总。输出为：

```text
outputs/flashvtg_dq_cgp_v3_topk4_test/
├── raw/
│   ├── hl_test_submission_nms_thd_0.7.jsonl
│   ├── metrics_tau_0.4.json
│   ├── metrics_tau_0.5.json
│   ├── metrics_tau_0.6.json
│   └── summary_three_thresholds.json
└── calibrated/
    ├── hl_test_submission_nms_thd_0.7.jsonl
    ├── metrics_tau_0.4.json
    ├── metrics_tau_0.5.json
    ├── metrics_tau_0.6.json
    └── summary_three_thresholds.json
```

在 validation 上执行相同流程：

```bash
DQ_V3_SPLIT=val \
DQ_V3_CHECKPOINT=/path/to/model_best.ckpt \
DQ_V3_OUTPUT=outputs/flashvtg_dq_cgp_v3_topk4_val \
bash scripts/evaluate_flashvtg_dq_cgp_v3.sh
```

关闭 existence calibration 的反事实：

```bash
DQ_V3_EXIST_LOGIT_BIAS=0 \
DQ_V3_CHECKPOINT=/path/to/model_best.ckpt \
DQ_V3_OUTPUT=outputs/flashvtg_dq_cgp_v3_topk4_no_calibration \
bash scripts/evaluate_flashvtg_dq_cgp_v3.sh
```

## 8. 日志与结果文件

此次发布保留：

- [训练日志](training_logs/flashvtg_dq_cgp_v3_topk4_seed2024/train.log.txt)
- [逐轮 validation 日志](training_logs/flashvtg_dq_cgp_v3_topk4_seed2024/eval.log.txt)
- [测试日志](training_logs/flashvtg_dq_cgp_v3_topk4_seed2024/test.log.txt)
- `results/flashvtg_dq_cgp_v3_topk4_seed2024/test/raw/` 下的 raw prediction 与三阈值指标
- `results/flashvtg_dq_cgp_v3_topk4_seed2024/test/calibrated/` 下的 calibrated prediction 与三阈值指标

不提交 `model_best.ckpt`、`model_latest.ckpt`、TensorBoard event 或训练期 `code.zip`。

## 9. 测试代码

运行 v3 单元测试和静态检查：

```bash
python -m unittest discover \
  -s models/flashvtg_dq_cgp_v3_gmr/tests \
  -v

bash -n scripts/train_flashvtg_dq_cgp_v3.sh
bash -n scripts/evaluate_flashvtg_dq_cgp_v3.sh
python -m compileall models/flashvtg_dq_cgp_v3_gmr training eval scripts
```

核心测试覆盖 sparse top-k support、candidate mask、level/point route 混合、`beta=0` 恒等性、
route/relation loss 梯度和 metric-aware scheduler。

## 10. 代码结构

```text
models/flashvtg_dq_cgp_v3_gmr/dq_cgp.py       # temporal binding, sparse RCG, prompt attention, FRF
models/flashvtg_dq_cgp_v3_gmr/model.py        # FlashVTG integration and auxiliary losses
models/flashvtg_dq_cgp_v3_gmr/model_config.py # exact v3 top-k4 configuration
models/flashvtg_dq_cgp_v3_gmr/run.py          # independent train/infer launcher
training/flash_vtg_gmr/                       # shared data, training and inference loops
eval/                                         # full Soccer-GMR evaluation
scripts/train_flashvtg_dq_cgp_v3.sh           # exact training command
scripts/evaluate_flashvtg_dq_cgp_v3.sh        # inference + calibration + three-threshold evaluation
```

## 11. 可复现性说明

- validation 只用于 checkpoint selection 和 existence bias calibration；test 不参与拟合。
- raw 与 calibrated 结果来自同一个 checkpoint 和同一组 temporal windows。
- calibration 是单调变换，因此 AUROC、mAP、mR、mR+、mIoU 和 mIoU+ 保持不变。
- 所有 Standard test prediction 覆盖 1,036 个 query-video pairs。
- 默认 FlashVTG inference 使用 cuDNN benchmark；复跑时少量浮点末位值或零分并列窗口顺序可能
  不同，本次复跑的全部两位小数汇总指标与提交结果一致。

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

仓库代码遵循 [MIT License](LICENSE)。FlashVTG 派生组件说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。Soccer-GMR 数据、视频和特征遵循其各自的
访问协议、NDA 和版权条款。
