#!/usr/bin/env bash
set -euo pipefail

###############################################################################
############## RoboTwin Federated Learning + LoRA Launch Script ###############
#                                                                             #
# Default: 3 GPUs = 3 clients, FedAvg, partition by task category, LoRA fine-tuning                 #
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

# ============ LoRA params ============
export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Client-side LoRA-MoE has a larger activation peak than standard LoRA.
# Keep this independent from the nominal training batch size.
export CLIENT_BATCH_SIZE="${CLIENT_BATCH_SIZE:-8}"

# ============ Plain client/server LoRA-MoE ============
# One switch forces a standard single-router LoRA-MoE on every client and the
# global server model, with trainable parameters aggregated directly by FedAvg.
export ENABLE_VANILLA_LORA_MOE="${ENABLE_VANILLA_LORA_MOE:-false}"
export MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-8}"
export MOE_K_EXPERTS="${MOE_K_EXPERTS:-4}"
export LAMBDA_AUX="${LAMBDA_AUX:-0.001}"

if [[ "${ENABLE_VANILLA_LORA_MOE,,}" =~ ^(1|true|yes|on)$ ]]; then
    export ENABLE_VANILLA_LORA_MOE="true"
    export FL_AGGREGATION="fedavg"
    POLICY_USE_LORA="false"
    POLICY_USE_LORA_MOE="true"
    METHOD_NAME="vanilla-lora-moe"
    # Defaults preserve the old batch-32/500-step sample budget when using
    # batch 8 with 2,000 optimizer steps per round.
    POLICY_LR="${VANILLA_LR:-2.5e-5}"
    POLICY_WARMUP_STEPS="${VANILLA_WARMUP_STEPS:-10000}"
    POLICY_DECAY_STEPS="${VANILLA_DECAY_STEPS:-200000}"
    POLICY_DECAY_LR="${VANILLA_DECAY_LR:-2.5e-6}"
else
    POLICY_USE_LORA="true"
    POLICY_USE_LORA_MOE="false"
    METHOD_NAME="lora"
    POLICY_LR="5.0e-4"
    POLICY_WARMUP_STEPS="2000"
    POLICY_DECAY_STEPS="100000"
    POLICY_DECAY_LR="5.0e-6"
fi

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
JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-fl-robotwin-${METHOD_NAME}-${FL_AGGREGATION}-${FL_NUM_CLIENTS}clients-r${LORA_RANK}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
FL_OUTPUT_DIR="${OUTPUT_DIR}/global_model"
export FL_OUTPUT_DIR="${FL_OUTPUT_DIR}"

echo "=============================================="
echo "ROBOTWIN FEDERATED LEARNING + LoRA"
echo "=============================================="
echo "  Clients: ${FL_NUM_CLIENTS} (GPU processes: ${NUM_PROCESSES})"
echo "  Aggregation: ${FL_AGGREGATION}"
echo "  Partition: ${FL_PARTITION}"
echo "  Local Steps: ${FL_LOCAL_STEPS}, FL Rounds: ${FL_NUM_ROUNDS}"
echo "  Client Batch Size: ${CLIENT_BATCH_SIZE}"
echo "  LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "  Vanilla LoRA-MoE: ${ENABLE_VANILLA_LORA_MOE}"
if [[ "${ENABLE_VANILLA_LORA_MOE}" == "true" ]]; then
echo "  MoE: experts=${MOE_NUM_EXPERTS}, top_k=${MOE_K_EXPERTS}, lambda_aux=${LAMBDA_AUX}"
echo "  Client architecture: LoRA-MoE | Server architecture: LoRA-MoE | Aggregation: FedAvg"
echo "  Vanilla LR: peak=${POLICY_LR}, warmup=${POLICY_WARMUP_STEPS}, decay=${POLICY_DECAY_STEPS}, final=${POLICY_DECAY_LR}"
fi
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
    --policy.optimizer_lr=${POLICY_LR}
    --policy.scheduler_warmup_steps=${POLICY_WARMUP_STEPS}
    --policy.scheduler_decay_steps=${POLICY_DECAY_STEPS}
    --policy.scheduler_decay_lr=${POLICY_DECAY_LR}
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l

    # ---- LoRA Settings ----
    --policy.use_lora=${POLICY_USE_LORA}
    --policy.use_lora_moe=${POLICY_USE_LORA_MOE}
    --policy.use_lora_moe_forced_last=false
    --policy.lora_rank=${LORA_RANK}
    --policy.lora_alpha=${LORA_ALPHA}
    --policy.lora_dropout=${LORA_DROPOUT}
    --policy.loramoe_num_experts=${MOE_NUM_EXPERTS}
    --policy.loramoe_router_top_k=${MOE_K_EXPERTS}
    --policy.loramoe_enable_a_experts=true
    --policy.loramoe_enable_b_experts=true
    --policy.loramoe_share_a_across_experts=false
    --policy.loramoe_ab_routing=false
    --policy.lambda_aux=${LAMBDA_AUX}
    --policy.enable_affordance=false
    --policy.enable_affordance_v2=false
    --policy.enable_fard=false
    --policy.enable_pcea=false
    --policy.enable_tcr=false
    --policy.tcr_enable_loss=false
    --policy.use_visual_token_prune=false

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
    --wandb.project=lerobot_fl_robottwin_lora
    --wandb.mode=offline
)

echo "Launching..."
# DeepSpeed configuration
export DEEPSPEED_CONFIG_FILE="launch/ds_config.json"
accelerate launch --use_deepspeed "${ARGS[@]}"


###############################################################################
################################ USAGE ########################################
#
# 1. Default: 3 GPUs, FedAvg, LoRA rank=8
#    bash launch/internvla_a1_3b_fl_robotwin_lora.sh
#
# 2. Custom LoRA params
#    LORA_RANK=16 LORA_ALPHA=32 bash launch/internvla_a1_3b_fl_robotwin_lora.sh
#
# 3. FedProx + LoRA (for non-IID data)
#    FL_AGGREGATION=fedprox bash launch/internvla_a1_3b_fl_robotwin_lora.sh
#
# 4. Dirichlet non-IID partition + LoRA
#    FL_PARTITION=dirichlet FL_DIRICHLET_ALPHA=0.3 bash launch/internvla_a1_3b_fl_robotwin_lora.sh
#
# 5. SCAFFOLD + LoRA
#    FL_AGGREGATION=scaffold bash launch/internvla_a1_3b_fl_robotwin_lora.sh
#
# 6. Quick test (5 rounds)
#    FL_NUM_ROUNDS=5 FL_LOCAL_STEPS=20 bash launch/internvla_a1_3b_fl_robotwin_lora.sh
#
# 7. Plain client/server LoRA-MoE + FedAvg
#    ENABLE_VANILLA_LORA_MOE=true MOE_NUM_EXPERTS=8 MOE_K_EXPERTS=4 \
#      bash launch/internvla_a1_3b_fl_robotwin_lora.sh
#
###############################################################################
