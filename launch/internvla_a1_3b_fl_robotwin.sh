#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################## RoboTwin Federated Learning Launch Script ##################
#                                                                             #
# Default: 3 GPUs = 3 clients, FedAvg, partition by task category                          #
###############################################################################

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

WANDB_TOKEN="your_wandb_token"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV=internvla

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# wandb login ${WANDB_TOKEN}  # Uncomment if using wandb

###############################################################################
########################## Distributed Config #################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29501}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# GPU config: use 3 GPUs by default
PROC_PER_NODE="${NUM_GPUS:-3}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

# If NUM_PROCESSES > 1, multi_gpu is required
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    MULTI_GPU_FLAG="--multi_gpu"
else
    MULTI_GPU_FLAG=""
fi

# NCCL settings
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

###############################################################################
####################### FEDERATED LEARNING PARAMETERS #########################

# Core FL params (passed via env vars)
export FL_NUM_CLIENTS="${FL_NUM_CLIENTS:-50}"              # Total number of clients
export NUM_GPUS="${NUM_GPUS:-3}"                         # number of GPUs actually used
export FL_LOCAL_STEPS="${FL_LOCAL_STEPS:-100}"           # local training steps per round
export FL_LOCAL_EPOCHS="${FL_LOCAL_EPOCHS:-1}"           # local epochs per round
export FL_NUM_ROUNDS="${FL_NUM_ROUNDS:-50}"              # total FL rounds
export FL_AGGREGATION="${FL_AGGREGATION:-fedavg}"       # aggregation: fedavg, fedprox, scaffold
export FL_FEDPROX_MU="${FL_FEDPROX_MU:-0.01}"           # FedProx mu

# Partition strategy
export FL_PARTITION="${FL_PARTITION:-by_episode}"           # by_task, by_episode, dirichlet
export FL_DIRICHLET_ALPHA="${FL_DIRICHLET_ALPHA:-0.5}"  # Dirichlet alpha
export FL_HELD_OUT_TASKS="${FL_HELD_OUT_TASKS:-5}"      # number of held-out validation tasks

# Other
export FL_SAVE_FREQ="${FL_SAVE_FREQ:-10}"
export FL_LOG_FREQ="${FL_LOG_FREQ:-200}"
export FL_SEED="${FL_SEED:-42}"

# ============ POLICY & MODEL ============
POLICY="qwena1"
PRETRAINED_PATH="InternRobotics/InternVLA-A1-3B"

# ============ DATASET (RoboTwin) ============
# Auto-detect datasets under data/robotwin/
DATASET_REPO_ID="$(
  find -L "data/robotwin" -mindepth 2 -maxdepth 2 -type d -name "aloha-*" 2>/dev/null \
  | while read -r d; do
        if [[ -d "$d/meta" && -d "$d/videos" ]]; then
            echo "${d#data/}"
        fi
    done \
  | sort -u
)"
ACTION_TYPE="delta"
USE_EXTERNAL_STATS="true"

# ============ OUTPUT ============
BASE_OUTPUT_DIR="outputs/${POLICY}"
JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-fl-robotwin-${FL_AGGREGATION}-${FL_NUM_CLIENTS}clients"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
FL_OUTPUT_DIR="${OUTPUT_DIR}/global_model"
export FL_OUTPUT_DIR="${FL_OUTPUT_DIR}"

echo "=============================================="
echo "ROBOTWIN FEDERATED LEARNING"
echo "=============================================="
echo "  Clients: ${FL_NUM_CLIENTS} (GPU processes: ${NUM_PROCESSES})"
echo "  Aggregation: ${FL_AGGREGATION}"
echo "  Partition: ${FL_PARTITION}"
echo "  Local Steps: ${FL_LOCAL_STEPS}, FL Rounds: ${FL_NUM_ROUNDS}"
echo "=============================================="

###############################################################################
######################### BUILD ARGS AND LAUNCH ##############################

# Detect dataset
if [[ -z "${DATASET_REPO_ID}" ]]; then
    echo "ERROR: No RoboTwin dataset found in data/robotwin/"
    echo "Please convert your data first:"
    exit 1
fi

# Build args
ARGS=(
    ${MULTI_GPU_FLAG}
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_fl_robotwin.py

    --output_dir="${OUTPUT_DIR}"
    --num_workers=8
    --job_name="${JOB_NAME}"

    # ---- Policy ----
    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.pretrained_path=${PRETRAINED_PATH}
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=5.0e-5
    --policy.scheduler_warmup_steps=2000
    --policy.scheduler_decay_steps=100000
    --policy.scheduler_decay_lr=5.0e-6
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l

    # ---- Dataset ----
    --dataset.type=${POLICY}
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=${USE_EXTERNAL_STATS}
    --dataset.external_stats_path=${HF_HOME}/lerobot/stats/aloha/${ACTION_TYPE}/stats.json

    # ---- Training ----
    --seed=42
    --batch_size=1                         # 3B model, batch=1
    --steps=$((FL_NUM_ROUNDS * FL_LOCAL_STEPS * FL_LOCAL_EPOCHS))
    --save_freq=10000
    --resume=false

    # ---- Logging ----
    --wandb.enable=true
    --wandb.project=lerobot_fl_robottwin
    --wandb.mode=offline
)

echo "Launching..."
accelerate launch "${ARGS[@]}"


###############################################################################
################################ USAGE ########################################
#
# 1. Default: 3 GPUs, FedAvg, partition by task
#    bash launch/internvla_a1_3b_fl_robotwin.sh
#
# 2. FedProx (for non-IID data)
#    FL_AGGREGATION=fedprox bash launch/internvla_a1_3b_fl_robotwin.sh
#
# 3. Dirichlet non-IID partition
#    FL_PARTITION=dirichlet FL_DIRICHLET_ALPHA=0.3 bash launch/internvla_a1_3b_fl_robotwin.sh
#
# 4. SCAFFOLD algorithm
#    FL_AGGREGATION=scaffold bash launch/internvla_a1_3b_fl_robotwin.sh
#
# 5. Quick test (5 rounds)
#    FL_NUM_ROUNDS=5 FL_LOCAL_STEPS=20 bash launch/internvla_a1_3b_fl_robotwin.sh
#
###############################################################################
