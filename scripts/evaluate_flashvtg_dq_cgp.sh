#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DQ_PYTHON="${DQ_PYTHON:-python}"
DQ_GPU="${DQ_GPU:-0}"
DQ_SPLIT="${DQ_SPLIT:-test}"
DQ_CHECKPOINT="${DQ_CHECKPOINT:-checkpoints/flashvtg_dq_cgp_gmr_best_epoch32.ckpt}"
DQ_OPT="${DQ_OPT:-checkpoints/flashvtg_dq_cgp_gmr_best_opt.json}"
DQ_OUTPUT="${DQ_OUTPUT:-outputs/flashvtg_dq_cgp_${DQ_SPLIT}}"
DQ_TEXT_FEATURES="${DQ_TEXT_FEATURES:-Soccergmr/clip_text}"
DQ_CLIP_FEATURES="${DQ_CLIP_FEATURES:-Soccergmr/clip}"
DQ_SLOWFAST_FEATURES="${DQ_SLOWFAST_FEATURES:-Soccergmr/slowfast}"
DQ_EXIST_LOGIT_BIAS="${DQ_EXIST_LOGIT_BIAS:--0.34110591}"

case "${DQ_SPLIT}" in
  val|test)
    DQ_LABELS="data/label/Standard/${DQ_SPLIT}.jsonl"
    ;;
  *)
    printf 'DQ_SPLIT must be val or test, got: %s\n' "${DQ_SPLIT}" >&2
    exit 2
    ;;
esac

RAW_DIR="${DQ_OUTPUT}/raw"
CALIBRATED_DIR="${DQ_OUTPUT}/calibrated"
mkdir -p "${RAW_DIR}" "${CALIBRATED_DIR}"

CUDA_VISIBLE_DEVICES="${DQ_GPU}" "${DQ_PYTHON}" \
  -m models.flashvtg_dq-cgp-gmr-v2.run infer \
  models/flashvtg_dq-cgp-gmr-v2/model_config.py \
  --resume "${DQ_CHECKPOINT}" \
  --opt_path "${DQ_OPT}" \
  --eval_split_name "${DQ_SPLIT}" \
  --eval_path "${DQ_LABELS}" \
  --eval_results_dir "${RAW_DIR}" \
  --v_feat_dirs "${DQ_SLOWFAST_FEATURES}" "${DQ_CLIP_FEATURES}" \
  --t_feat_dir "${DQ_TEXT_FEATURES}" \
  --v_feat_dim 2816 \
  --t_feat_dim 512 \
  --device 0 \
  --nms_thd 0.7

RAW_PREDICTIONS="${RAW_DIR}/hl_${DQ_SPLIT}_submission_nms_thd_0.7.jsonl"
CALIBRATED_PREDICTIONS="${CALIBRATED_DIR}/hl_${DQ_SPLIT}_submission_nms_thd_0.7.jsonl"

"${DQ_PYTHON}" scripts/calibrate_existence_scores.py \
  --input "${RAW_PREDICTIONS}" \
  --output "${CALIBRATED_PREDICTIONS}" \
  --logit_bias "${DQ_EXIST_LOGIT_BIAS}"

for threshold in 0.4 0.5 0.6; do
  "${DQ_PYTHON}" -m eval.eval_main \
    --submission_path "${CALIBRATED_PREDICTIONS}" \
    --gt_path "${DQ_LABELS}" \
    --save_path "${CALIBRATED_DIR}/metrics_tau_${threshold}.json" \
    --cls_thresholds 0.4 0.5 0.6 \
    --gmiou_cls_threshold "${threshold}" \
    --map_num_workers 1 \
    --not_verbose
done

"${DQ_PYTHON}" scripts/summarize_three_thresholds.py \
  --metrics_dir "${CALIBRATED_DIR}" \
  --output "${DQ_OUTPUT}/summary_three_thresholds.json"
