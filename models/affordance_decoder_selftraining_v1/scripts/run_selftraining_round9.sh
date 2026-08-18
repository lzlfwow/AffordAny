#
# Self-training Round 9: EMA Dynamic Pseudo-Label Refresh
#
# Based on R8 config. Key addition:
#   - --refresh-every 20: every 20 epochs, re-infer all pseudo-label samples
#     with the EMA teacher model and replace stale heatmaps.
#     Breaks confirmation bias by letting pseudo labels improve with the model.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && cd ../../.. && pwd)"
cd "$REPO_ROOT"

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 REAL_CACHE PSEUDO_CACHE INIT_CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

# Paths are supplied by the caller to keep the release machine-independent.
V5_CACHE="$(realpath "$1")"
PSEUDO_CACHE="$(realpath "$2")"
V5_CKPT="$(realpath "$3")"
mkdir -p "$4"
OUTPUT_DIR="$(realpath "$4")"

# ──────── Config ────────
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
EPOCHS=200

# No curriculum
CURRICULUM_START=0
CURRICULUM_RAMPUP=0

# Ultra-low pseudo weight (same as R7/R8)
PSEUDO_WEIGHT=0.1
PSEUDO_SAMPLE_RATIO=0.37
PSEUDO_HIGH_THRESH=0.65
PSEUDO_LOW_THRESH=0.15

# Loss weights (same as R7/R8)
PSEUDO_BCE_SCALE=0.5
PSEUDO_FOCAL_SCALE=0.5
LAMBDA_DICE=0.7

# Fine-tuning LR
LR=5e-5
WARMUP=10

# Label smoothing + EMA (same as R8)
LABEL_SMOOTHING=0.05
EMA_DECAY=0.999

# NEW: Dynamic pseudo-label refresh
REFRESH_EVERY=20
REFRESH_BATCH_SIZE=64

echo "=========================================="
echo " Self-Training Round 9 (R8 + EMA Pseudo-Label Refresh)"
echo "=========================================="
echo "  Real data:        $V5_CACHE"
echo "  Pseudo data:      $PSEUDO_CACHE"
echo "  Init:             $V5_CKPT"
echo "  Output:           $OUTPUT_DIR"
echo "  GPUs:             $GPUS ($NUM_GPUS)"
echo "  Epochs:           $EPOCHS"
echo "  Pseudo weight:    $PSEUDO_WEIGHT"
echo "  Pseudo BCE scale: $PSEUDO_BCE_SCALE"
echo "  Pseudo focal:     $PSEUDO_FOCAL_SCALE"
echo "  Lambda dice:      $LAMBDA_DICE"
echo "  Label smoothing:  $LABEL_SMOOTHING"
echo "  EMA decay:        $EMA_DECAY"
echo "  Refresh every:    $REFRESH_EVERY epochs (NEW)"
echo "  Refresh batch:    $REFRESH_BATCH_SIZE (NEW)"
echo "  LR:               $LR warmup=$WARMUP"
echo ""

CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node="$NUM_GPUS" \
    --master_port="${MASTER_PORT:-29505}" "$SCRIPT_DIR/train.py" \
    --cache-root "$V5_CACHE" \
    --pseudo-manifest \
        "$PSEUDO_CACHE/train_seen_instruction_manifest.json" \
        "$PSEUDO_CACHE/train_unseen_instruction_manifest.json" \
    --init-checkpoint "$V5_CKPT" \
    --pseudo-weight $PSEUDO_WEIGHT \
    --pseudo-sample-ratio $PSEUDO_SAMPLE_RATIO \
    --pseudo-high-thresh $PSEUDO_HIGH_THRESH \
    --pseudo-low-thresh $PSEUDO_LOW_THRESH \
    --pseudo-bce-scale $PSEUDO_BCE_SCALE \
    --pseudo-focal-scale $PSEUDO_FOCAL_SCALE \
    --curriculum-start $CURRICULUM_START \
    --curriculum-rampup $CURRICULUM_RAMPUP \
    --epochs $EPOCHS \
    --batch-size 16 \
    --learning-rate $LR \
    --warmup-epochs $WARMUP \
    --lambda-dice $LAMBDA_DICE \
    --label-smoothing $LABEL_SMOOTHING \
    --ema-decay $EMA_DECAY \
    --refresh-every $REFRESH_EVERY \
    --refresh-batch-size $REFRESH_BATCH_SIZE \
    --eval-every 10 \
    --disable-amp \
    --output-dir "$OUTPUT_DIR"
