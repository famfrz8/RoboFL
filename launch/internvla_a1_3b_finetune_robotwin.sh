#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-internvla_a1}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login
fi

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

ACCELERATE_MODE_ARGS=()
if (( NUM_PROCESSES > 1 )); then
    ACCELERATE_MODE_ARGS+=(--multi_gpu)
fi

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_HOME="/usr/local/cuda-12.8"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd ${PROJ_ROOT}

# 1. policy config
POLICY="qwena1"
PRETRAINED_PATH="${PRETRAINED_PATH:-InternRobotics/InternVLA-A1-3B}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Match the nominal 16M samples used by the 8-client LoRA + server LoRAMoE run.
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_STEPS="${TRAIN_STEPS:-62500}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-3125}"
DECAY_STEPS="${DECAY_STEPS:-62500}"
DECAY_LR="${DECAY_LR:-1.0e-5}"

# 2. dataset config
DATASET_REPO_ID="$(
  find -L "data/robotwin" -mindepth 2 -maxdepth 2 -type d -name "aloha-agilex*" 2>/dev/null \
  | while read -r d; do
        if [[ -d "$d/meta" && -d "$d/videos" ]]; then
            echo "${d#data/}"
        fi
    done \
  | sort -u
)"
ACTION_TYPE=delta      # abs | delta
USE_EXTERNAL_STATS=true # true | false

# 3. output config
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
PRETRAINED_DETAIL="a1_agibotworld_700k"
JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-robotwin-${ACTION_TYPE}-${PRETRAINED_DETAIL}-lora"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"

ARGS=(
    "${ACCELERATE_MODE_ARGS[@]}"
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}" 
    src/lerobot/scripts/lerobot_train.py

    --output_dir="${OUTPUT_DIR}"
    --num_workers=12
    --job_name="${JOB_NAME}"

    # ---- Policy ----
    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.pretrained_path=${PRETRAINED_PATH}
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=${LEARNING_RATE}
    --policy.scheduler_warmup_steps=${WARMUP_STEPS}
    --policy.scheduler_decay_steps=${DECAY_STEPS}
    --policy.scheduler_decay_lr=${DECAY_LR}
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l

    # ---- Pure LoRA (no LoRAMoE, no federated learning) ----
    --policy.use_lora=true
    --policy.use_lora_moe=false
    --policy.use_lora_moe_forced_last=false
    --policy.lora_rank=${LORA_RANK}
    --policy.lora_alpha=${LORA_ALPHA}
    --policy.lora_dropout=${LORA_DROPOUT}

    # ---- Dataset ----
    --dataset.type=${POLICY}
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=${USE_EXTERNAL_STATS}
    --dataset.external_stats_path=${HF_HOME}/lerobot/stats/aloha/${ACTION_TYPE}/stats.json

    # ---- If using gpu num >1, you can use dist_loading
    # --dataset.dist_loading=true

    # ---- Training ----
    --seed=42
    --batch_size=${BATCH_SIZE}
    --steps=${TRAIN_STEPS}
    # --eval_freq=60000
    --save_freq=10000
    --log_freq=200

    # ---- Logging ----
    --wandb.enable=true
    --wandb.project=lerobot_lab_${POLICY}_robotwin_lora
    --wandb.mode=offline
)

accelerate launch "${ARGS[@]}"
