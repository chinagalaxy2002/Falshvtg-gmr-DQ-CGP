#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DQ_V3_PYTHON="${DQ_V3_PYTHON:-python}"
DQ_V3_GPU="${DQ_V3_GPU:-0}"
DQ_V3_SPLIT="${DQ_V3_SPLIT:-test}"
DQ_V3_CHECKPOINT="${DQ_V3_CHECKPOINT:?Set DQ_V3_CHECKPOINT to a trained model_best.ckpt}"
DQ_V3_OPT="${DQ_V3_OPT:-results/flashvtg_dq_cgp_v3_topk4_seed2024/validation/opt.json}"
DQ_V3_OUTPUT="${DQ_V3_OUTPUT:-outputs/flashvtg_dq_cgp_v3_topk4_${DQ_V3_SPLIT}}"
DQ_V3_TEXT_FEATURES="${DQ_V3_TEXT_FEATURES:-Soccergmr/clip_text}"
DQ_V3_CLIP_FEATURES="${DQ_V3_CLIP_FEATURES:-Soccergmr/clip}"
DQ_V3_SLOWFAST_FEATURES="${DQ_V3_SLOWFAST_FEATURES:-Soccergmr/slowfast}"
DQ_V3_EXIST_LOGIT_BIAS="${DQ_V3_EXIST_LOGIT_BIAS:--0.34110591}"

if [[ ! -f "${DQ_V3_CHECKPOINT}" ]]; then
  printf 'checkpoint not found: %s\n' "${DQ_V3_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${DQ_V3_OPT}" ]]; then
  printf 'training options not found: %s\n' "${DQ_V3_OPT}" >&2
  exit 2
fi

case "${DQ_V3_SPLIT}" in
  val|test)
    DQ_V3_LABELS="data/label/Standard/${DQ_V3_SPLIT}.jsonl"
    ;;
  *)
    printf 'DQ_V3_SPLIT must be val or test, got: %s\n' "${DQ_V3_SPLIT}" >&2
    exit 2
    ;;
esac

RAW_DIR="${DQ_V3_OUTPUT}/raw"
CALIBRATED_DIR="${DQ_V3_OUTPUT}/calibrated"
mkdir -p "${RAW_DIR}" "${CALIBRATED_DIR}"

CUDA_VISIBLE_DEVICES="${DQ_V3_GPU}" "${DQ_V3_PYTHON}" \
  -m models.flashvtg_dq_cgp_v3_gmr.run infer \
  models/flashvtg_dq_cgp_v3_gmr/model_config.py \
  --resume "${DQ_V3_CHECKPOINT}" \
  --opt_path "${DQ_V3_OPT}" \
  --eval_split_name "${DQ_V3_SPLIT}" \
  --eval_path "${DQ_V3_LABELS}" \
  --eval_results_dir "${RAW_DIR}" \
  --v_feat_dirs "${DQ_V3_SLOWFAST_FEATURES}" "${DQ_V3_CLIP_FEATURES}" \
  --t_feat_dir "${DQ_V3_TEXT_FEATURES}" \
  --v_feat_dim 2816 \
  --t_feat_dim 512 \
  --device 0 \
  --nms_thd 0.7

RAW_PREDICTIONS="${RAW_DIR}/hl_${DQ_V3_SPLIT}_submission_nms_thd_0.7.jsonl"
CALIBRATED_PREDICTIONS="${CALIBRATED_DIR}/hl_${DQ_V3_SPLIT}_submission_nms_thd_0.7.jsonl"

"${DQ_V3_PYTHON}" scripts/calibrate_existence_scores.py \
  --input "${RAW_PREDICTIONS}" \
  --output "${CALIBRATED_PREDICTIONS}" \
  --logit_bias "${DQ_V3_EXIST_LOGIT_BIAS}"

for variant in raw calibrated; do
  if [[ "${variant}" == "raw" ]]; then
    predictions="${RAW_PREDICTIONS}"
    metrics_dir="${RAW_DIR}"
  else
    predictions="${CALIBRATED_PREDICTIONS}"
    metrics_dir="${CALIBRATED_DIR}"
  fi

  for threshold in 0.4 0.5 0.6; do
    "${DQ_V3_PYTHON}" -m eval.eval_main \
      --submission_path "${predictions}" \
      --gt_path "${DQ_V3_LABELS}" \
      --save_path "${metrics_dir}/metrics_tau_${threshold}.json" \
      --cls_thresholds 0.4 0.5 0.6 \
      --gmiou_cls_threshold "${threshold}" \
      --map_num_workers 1 \
      --not_verbose
  done

  "${DQ_V3_PYTHON}" scripts/summarize_three_thresholds.py \
    --metrics_dir "${metrics_dir}" \
    --output "${metrics_dir}/summary_three_thresholds.json"
done
