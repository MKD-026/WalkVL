#!/bin/bash
#$ -S /bin/bash
#$ -P cs585
#$ -N walkvl_internvl_rgb_seg_depth
#$ -j y
#$ -o /projectnb/cs585/students/mkd/740/WalkVL/experiments/internvl/logs/$JOB_NAME.$JOB_ID.log
#$ -l h_rt=48:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -pe omp 8

# LoRA fine-tuning for InternVL2 on RGB + segmentation + depth.
#
# This uses the multimodal JSONL files produced by:
#   python /projectnb/cs585/students/mkd/740/WalkVL/internvl/build_dataset.py
#
# Each training image is a single vertical composite:
#   top    : RGB
#   middle : instance/semantic segmentation
#   bottom : depth
#
# Submit:
#   qsub -v MODEL=1b /projectnb/cs585/students/mkd/740/WalkVL/internvl/train_rgb_seg_depth_lora.sh
#   qsub -v MODEL=4b /projectnb/cs585/students/mkd/740/WalkVL/internvl/train_rgb_seg_depth_lora.sh

set -euo pipefail

BASE=/projectnb/cs585/students/mkd/740/WalkVL
CODE=$BASE/internvl
EXP=$BASE/experiments/internvl
DATA=$EXP/data
LOGS=$EXP/logs
mkdir -p "$LOGS" "$EXP/checkpoints"

if [ "${MODEL:-}" = "1b" ]; then
    MODEL_ID="OpenGVLab/InternVL2-1B"
    GRAD_ACCUM=8
    EPOCHS=40
elif [ "${MODEL:-}" = "4b" ]; then
    MODEL_ID="OpenGVLab/InternVL2-4B"
    GRAD_ACCUM=16
    EPOCHS=20
else
    echo "ERROR: set MODEL=1b or MODEL=4b via qsub -v MODEL=1b"
    exit 1
fi

OUTPUT=$EXP/checkpoints/internvl2-${MODEL}-rgb-seg-depth-lora

echo "============================================"
echo "WalkVL InternVL LoRA fine-tuning"
echo "Mode     : RGB + segmentation + depth"
echo "Model    : $MODEL_ID"
echo "Train    : $DATA/train_multimodal.jsonl"
echo "Val      : $DATA/val_multimodal.jsonl"
echo "Output   : $OUTPUT"
echo "Log dir  : $LOGS"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Epochs   : $EPOCHS | Save every: 5"
echo "============================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projectnb/cs585/students/mkd/740/envs/internvl

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_HOME="${HF_HOME:-$EXP/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$EXP/cache/pip}"

python "$CODE/train.py" \
    --model "$MODEL_ID" \
    --mode multimodal \
    --train "$DATA/train_multimodal.jsonl" \
    --val "$DATA/val_multimodal.jsonl" \
    --output "$OUTPUT" \
    --epochs "$EPOCHS" \
    --save-every 5 \
    --batch-size 1 \
    --grad-accum "$GRAD_ACCUM" \
    --lr 2e-4 \
    --lora-r 16 \
    --max-length 4096 \
    --wandb-project walkvl-internvl
