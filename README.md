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

## 3. 复现结果

### 3.1 Standard validation

最佳 checkpoint 来自第 60 轮，按 no-NMS `MR-full-mAP=26.03` 选择。

| Metric | no NMS | NMS=0.7 |
|---|---:|---:|
| MR-full-R1@0.3 | **47.84** | 47.45 |
| MR-full-R1@0.5 | 41.18 | 41.18 |
| MR-full-R1@0.7 | 26.27 | 26.27 |
| MR-full-mAP | 26.03 | **27.91** |
| MR-full-mAP@0.5 | 45.91 | **49.34** |
| MR-full-mAP@0.75 | 28.74 | **30.88** |
| MR-full-mIoU | **35.12** | 34.96 |
| GMR-TPR | 63.53 | 63.53 |
| GMR-TNR | 69.05 | 69.05 |
| GMR-BalancedAcc | 66.29 | 66.29 |

### 3.2 Standard test

测试保留 top-10 NMS predictions，并分别使用 existence thresholds `0.4`、`0.5`、`0.6`。
表中的 `Mean` 是三个阈值的算术平均。`Calibrated` 只对 existence probability 应用在
validation 上确定的单调变换，不改变窗口、窗口排序、AUROC、mAP、mR 或 mIoU：

```text
calibrated_score = sigmoid(logit(raw_score) - 0.34110591)
```

| Metric | Matched FlashVTG | Previous DQ-CGP v2 calibrated | v3 raw | v3 calibrated |
|---|---:|---:|---:|---:|
| AUROC | 73.74 | 75.89 | **76.02** | **76.02** |
| Mean Rej-F1 | 66.47 | 66.12 | 66.66 | **69.93** |
| mAP | 25.42 | 25.63 | **26.13** | **26.13** |
| mR@1 | 15.94 | 15.34 | **16.21** | **16.21** |
| mR@3 | 27.33 | **28.90** | 28.42 | 28.42 |
| mR@5 | 34.46 | 35.45 | **36.03** | **36.03** |
| mR+@3 | 8.16 | **11.80** | 9.34 | 9.34 |
| mR+@5 | 17.44 | 20.44 | **21.04** | **21.04** |
| mIoU@1 | 34.04 | 33.49 | **34.88** | **34.88** |
| mIoU@3 | 30.68 | 30.40 | **32.07** | **32.07** |
| mIoU@5 | 30.66 | 30.26 | **31.94** | **31.94** |
| Mean G-mIoU@1 | 46.08 | 44.10 | 44.07 | **47.78** |
| Mean G-mIoU@3 | 40.37 | 38.28 | 38.13 | **42.11** |
| Mean G-mIoU@5 | 38.85 | 36.41 | 36.27 | **40.47** |

v3 的逐阈值 raw / calibrated 结果如下：

| Threshold | Raw Rej-F1 | Cal. Rej-F1 | Raw G-mIoU@1 | Cal. G-mIoU@1 | Raw G-mIoU@3 | Cal. G-mIoU@3 | Raw G-mIoU@5 | Cal. G-mIoU@5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 61.13 | 66.01 | 38.59 | 43.13 | 32.24 | 37.12 | 30.11 | 35.24 |
| 0.5 | 67.25 | 71.30 | 44.27 | 48.96 | 38.30 | 43.35 | 36.44 | 41.75 |
| 0.6 | 71.60 | 72.49 | 49.35 | 51.25 | 43.86 | 45.85 | 42.27 | 44.43 |
| **Mean** | **66.66** | **69.93** | **44.07** | **47.78** | **38.13** | **42.11** | **36.27** | **40.47** |

相对 matched FlashVTG，v3 calibrated 的 AUROC、mAP、mR@5、mR+@5 和 Mean
G-mIoU@1/@3/@5 分别提高 `2.28`、`0.71`、`1.57`、`3.60` 和
`1.70/1.74/1.62` 个百分点。这里报告同一次测试的原始值与单调校准值，不把校准结果误作
新的检索窗口结果。

详细产物：

- [v3 calibrated test summary](results/flashvtg_dq_cgp_v3_topk4_seed2024/test/calibrated/summary_three_thresholds.json)
- [v3 raw test summary](results/flashvtg_dq_cgp_v3_topk4_seed2024/test/raw/summary_three_thresholds.json)
- [v3 best validation metrics, NMS](results/flashvtg_dq_cgp_v3_topk4_seed2024/validation/best_nms_metrics.json)
- [v3 best validation metrics, no NMS](results/flashvtg_dq_cgp_v3_topk4_seed2024/validation/best_no_nms_metrics.json)

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
