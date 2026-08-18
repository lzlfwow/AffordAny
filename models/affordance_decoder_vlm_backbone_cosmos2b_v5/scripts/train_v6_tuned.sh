# Multi-GPU retrain of AffordAny (cosmos2b_v5) with tuned hyperparameters.
#
# Goal: close the remaining gaps against baselines without changing model
# architecture. Prior best run (cosmos2b_v5_decoder) peaked at epoch 140/300
# with train_iou=0.79 vs val_iou=0.47 — heavy overfitting + over-prediction
# of positives (pred_pos 0.29 vs truth_pos 0.23) that inflates MAE.
#
# Key changes vs. previous best run:
#   epochs           300  -> 180   (peak was at 140; long tail wasted)
#   weight_decay    5e-4  -> 1.5e-3 (stronger L2)
#   dropout          0.1  -> 0.15  (mild regularization bump)
#   instruction_dropout 0.2 -> 0.3  (paraphrase robustness)
#   pos_weight       8.0  -> 5.0   (curb positive over-prediction -> MAE)
#   focal_alpha     0.75  -> 0.6   (same rationale)
#   warmup_epochs      5  -> 8     (smoother start)
# Kept: lr=2e-4, min_lr=1e-5, batch_size=16 per GPU (8xGPU -> 128 effective),
#       lambda_bce=0.2, lambda_focal=1.0, lambda_dice=0.5, focal_gamma=2.0,
#       hidden_size=256, prototypes=16, grid_size=12, decoder 4L/8H.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 CACHE_ROOT OUTPUT_DIR" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR" && cd ../../.. && pwd)"
V5="$REPO/models/affordance_decoder_vlm_backbone_cosmos2b_v5"
CACHE="$(realpath "$1")"
mkdir -p "$2"
OUT="$(realpath "$2")"

mkdir -p "$OUT"

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO:$V5/src:${PYTHONPATH:-}"

cd "$REPO"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  "$V5/scripts/train.py" \
  --cache-root "$CACHE" \
  --train-manifest "$CACHE/train_unseen_instruction_manifest.json" \
  --val-manifest   "$CACHE/val_unseen_object_seen_instruction_manifest.json" \
  --test-manifest  "$CACHE/test_unseen_object_seen_instruction_manifest.json" \
  --formal-val-unseen-object-manifest    "$CACHE/val_unseen_object_seen_instruction_manifest.json" \
  --formal-val-unseen-category-manifest  "$CACHE/val_unseen_category_seen_instruction_manifest.json" \
  --formal-test-unseen-object-manifest   "$CACHE/test_unseen_object_seen_instruction_manifest.json" \
  --formal-test-unseen-category-manifest "$CACHE/test_unseen_category_seen_instruction_manifest.json" \
  --formal-train-seen-manifest           "$CACHE/train_seen_instruction_manifest.json" \
  --formal-train-unseen-manifest         "$CACHE/train_unseen_instruction_manifest.json" \
  --output-dir "$OUT" \
  --seed 42 \
  --epochs 180 \
  --batch-size 16 \
  --num-workers 2 \
  --learning-rate 2e-4 \
  --weight-decay 1.5e-3 \
  --warmup-epochs 8 \
  --min-lr 1e-5 \
  --max-grad-norm 1.0 \
  --vlm-hidden-size 2048 \
  --hidden-size 256 \
  --point-depth 3 \
  --decoder-depth 4 \
  --decoder-heads 8 \
  --dropout 0.15 \
  --instruction-dropout-rate 0.3 \
  --num-prototypes 16 \
  --bottleneck-layers 2 \
  --num-views 1 \
  --grid-size 12 \
  --lambda-bce 0.2 \
  --lambda-focal 1.0 \
  --lambda-dice 0.5 \
  --pos-weight 5.0 \
  --focal-alpha 0.6 \
  --focal-gamma 2.0 \
  --selection-metric val_iou \
  --selection-mode max \
  --log-every-batches 30 \
  --eval-every 5 \
  2>&1 | tee -a "$OUT/train.log"
