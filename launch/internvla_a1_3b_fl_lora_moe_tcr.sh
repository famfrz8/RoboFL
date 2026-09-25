#!/usr/bin/env bash
set -euo pipefail

###############################################################################
############## LoRA-MoE + TCR Federated Learning Launch Script ###############
#                                                                             #
# Based on the working LoRA-MoE launch script, adding only the TCR-specific params.                   #
###############################################################################

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${WANDB_API_KEY:?Set WANDB_API_KEY in the environment before launching}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV=internvla_a1
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

wandb login

# wandb login ${WANDB_TOKEN}

###############################################################################
########################## Distributed Config #################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29502}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${NUM_GPUS:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    MULTI_GPU_FLAG="--multi_gpu"
else
    MULTI_GPU_FLAG=""
fi

export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

export CUDA_HOME="/usr/local/cuda-12.8"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=online
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

###############################################################################
####################### LoRA-MoE FL PARAMETERS ###############################

# Core FL parameters
export FL_NUM_CLIENTS="${FL_NUM_CLIENTS:-8}"
export NUM_GPUS="${NUM_GPUS:-8}"
export FL_LOCAL_STEPS="${FL_LOCAL_STEPS:-1000}"
export FL_LOCAL_EPOCHS="${FL_LOCAL_EPOCHS:-1}"
export FL_NUM_ROUNDS="${FL_NUM_ROUNDS:-100}"
export FL_AGGREGATION="${FL_AGGREGATION:-fedavg}"
export FL_FEDPROX_MU="${FL_FEDPROX_MU:-0.01}"

# MoE parameters
export MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-8}"
export MOE_K_EXPERTS="${MOE_K_EXPERTS:-4}"
export MOE_STEPS="${MOE_STEPS:-1000}"
export MOE_BATCH_SIZE="${MOE_BATCH_SIZE:-8}"

# Partition strategy
export FL_PARTITION="${FL_PARTITION:-by_category}"
export FL_DIRICHLET_ALPHA="${FL_DIRICHLET_ALPHA:-0.5}"
export FL_HELD_OUT_TASKS="${FL_HELD_OUT_TASKS:-5}"

# Other
export FL_SAVE_FREQ="${FL_SAVE_FREQ:-10}"
export FL_LOG_FREQ="${FL_LOG_FREQ:-200}"
export FL_SEED="${FL_SEED:-42}"

# LoRA parameters
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Aux loss (load-balancing) weight for LoRA-MoE
export LAMBDA_AUX="${LAMBDA_AUX:-0.0005}"

# Shared-A / Expert-B mode (recommended)
export MOE_SHARE_A="${MOE_SHARE_A:-true}"
export MOE_ENABLE_A_EXPERTS="${MOE_ENABLE_A_EXPERTS:-false}"
export MOE_ENABLE_B_EXPERTS="${MOE_ENABLE_B_EXPERTS:-true}"

# Legacy AB Routing: only meaningful when A experts are enabled
export MOE_AB_ROUTING="${MOE_AB_ROUTING:-false}"
export MOE_K_EXPERTS_A="${MOE_K_EXPERTS_A:-4}"
export MOE_K_EXPERTS_B="${MOE_K_EXPERTS_B:-4}"

# TCR parameters
export TCR_ENABLE="${TCR_ENABLE:-true}"
export TCR_LAMBDA_PROTO="${TCR_LAMBDA_PROTO:-0.5}"
export TCR_LAMBDA_CONTRAST="${TCR_LAMBDA_CONTRAST:-0.5}"
export TCR_MARGIN="${TCR_MARGIN:-0.1}"
export TCR_TAU_KEEP="${TCR_TAU_KEEP:-0.0}"
export TCR_USE_GEN_FEATURE="${TCR_USE_GEN_FEATURE:-true}"
export TCR_PROTOTYPE_MOMENTUM="${TCR_PROTOTYPE_MOMENTUM:-1.0}"
export TCR_LOSS_WEIGHT="${TCR_LOSS_WEIGHT:-0.001}"

###############################################################################
########################### MODEL & DATASET ###################################

POLICY="qwena1"
PRETRAINED_PATH="path/to/InternVLA-A1-3B"

DATASET_REPO_ID="$({
  find -L "data/robotwin" -mindepth 2 -maxdepth 2 -type d -name "aloha-*" 2>/dev/null \
  | while read -r d; do
        if [[ -d "$d/meta" && -d "$d/videos" ]]; then
            echo "${d#data/}"
        fi
    done \
  | sort -u \
  | tr '\n' ' ' \
  | xargs
})"
ACTION_TYPE="delta"
USE_EXTERNAL_STATS="true"

BASE_OUTPUT_DIR="outputs/${POLICY}"
JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-fl-lora-moe-tcr-${FL_NUM_CLIENTS}clients-r${LORA_RANK}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"

echo "=============================================="
echo "ROBOTWIN LoRA-MoE + TCR FEDERATED LEARNING"
echo "=============================================="
echo "  Clients: ${FL_NUM_CLIENTS} (GPU processes: ${NUM_PROCESSES})"
echo "  MoE Experts: ${MOE_NUM_EXPERTS}, Router Top-K: ${MOE_K_EXPERTS}"
echo "  Shared-A Mode: ${MOE_SHARE_A}"
echo "  A Experts Enabled: ${MOE_ENABLE_A_EXPERTS}"
echo "  B Experts Enabled: ${MOE_ENABLE_B_EXPERTS}"
echo "  AB Routing (legacy): ${MOE_AB_ROUTING}"
if [[ "${MOE_AB_ROUTING}" == "true" ]]; then
echo "  Router A Top-K: ${MOE_K_EXPERTS_A}, Router B Top-K: ${MOE_K_EXPERTS_B}"
fi
echo "  MoE Steps: ${MOE_STEPS}, MoE Batch Size: ${MOE_BATCH_SIZE}"
echo "  Local Steps: ${FL_LOCAL_STEPS}, FL Rounds: ${FL_NUM_ROUNDS}"
echo "  LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Loss Weights: lambda_aux=${LAMBDA_AUX}, tcr_loss_weight=${TCR_LOSS_WEIGHT}"
echo "  TCR: enable=${TCR_ENABLE}, lambda_proto=${TCR_LAMBDA_PROTO}, lambda_contrast=${TCR_LAMBDA_CONTRAST}, margin=${TCR_MARGIN}, tau_keep=${TCR_TAU_KEEP}, use_gen_feature=${TCR_USE_GEN_FEATURE}, prototype_momentum=${TCR_PROTOTYPE_MOMENTUM}"
echo "  Aggregation: ${FL_AGGREGATION}"
echo "  Partition: ${FL_PARTITION}"
echo "=============================================="

###############################################################################
######################### BUILD ARGS AND LAUNCH ##############################

if [[ -z "${DATASET_REPO_ID}" ]]; then
    echo "ERROR: No RoboTwin dataset found in data/robotwin/"
    exit 1
fi

ARGS=(
    ${MULTI_GPU_FLAG}
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_fl_robotwin_moe.py

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
    --policy.optimizer_lr=1.0e-4
    --policy.scheduler_warmup_steps=5000
    --policy.scheduler_decay_steps=100000
    --policy.scheduler_decay_lr=1.0e-5
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l

    # ---- LoRA-MoE Settings ----
    --policy.use_lora_moe=true
    --policy.use_lora=false
    --policy.loramoe_num_experts=${MOE_NUM_EXPERTS}
    --policy.loramoe_router_top_k=${MOE_K_EXPERTS}
    --policy.loramoe_share_a_across_experts=${MOE_SHARE_A}
    --policy.loramoe_enable_a_experts=${MOE_ENABLE_A_EXPERTS}
    --policy.loramoe_enable_b_experts=${MOE_ENABLE_B_EXPERTS}
    --policy.loramoe_ab_routing=${MOE_AB_ROUTING}
    --policy.loramoe_router_top_k_a=${MOE_K_EXPERTS_A}
    --policy.loramoe_router_top_k_b=${MOE_K_EXPERTS_B}
    --policy.lora_rank=${LORA_RANK}
    --policy.lora_alpha=${LORA_ALPHA}
    --policy.lora_dropout=${LORA_DROPOUT}
    --policy.lambda_aux=${LAMBDA_AUX}
    --policy.enable_tcr=${TCR_ENABLE}
    --policy.tcr_lambda_proto=${TCR_LAMBDA_PROTO}
    --policy.tcr_lambda_contrast=${TCR_LAMBDA_CONTRAST}
    --policy.tcr_margin=${TCR_MARGIN}
    --policy.tcr_tau_keep=${TCR_TAU_KEEP}
    --policy.tcr_use_gen_feature=${TCR_USE_GEN_FEATURE}
    --policy.tcr_prototype_momentum=${TCR_PROTOTYPE_MOMENTUM}
    --policy.tcr_loss_weight=${TCR_LOSS_WEIGHT}

    # ---- Dataset ----
    --dataset.type=${POLICY}
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=${USE_EXTERNAL_STATS}
    --dataset.external_stats_path=${HF_HOME}/lerobot/stats/aloha/${ACTION_TYPE}/stats.json

    # ---- Training ----
    --seed=42
    --batch_size=32
    --steps=$((FL_NUM_ROUNDS * FL_LOCAL_STEPS * FL_LOCAL_EPOCHS))
    --save_freq=10000
    --resume=false

    # ---- Logging ----
    --wandb.enable=true
    --wandb.project=lerobot_fl_robottwin_lora_moe_tcr
    --wandb.mode=online
)

echo "Launching..."

accelerate launch "${ARGS[@]}"


###############################################################################
################################ USAGE ########################################
#
# 1. Default: 8 GPUs, LoRA-MoE + TCR
#    bash launch/internvla_a1_3b_fl_lora_moe_tcr.sh
#
# 2. Tune TCR weights
#    TCR_LAMBDA_PROTO=0.2 TCR_LAMBDA_CONTRAST=1.0 bash launch/internvla_a1_3b_fl_lora_moe_tcr.sh
#
# 3. Enable legacy A/B dual experts + AB routing
#    MOE_SHARE_A=false MOE_ENABLE_A_EXPERTS=true MOE_AB_ROUTING=true \
#      bash launch/internvla_a1_3b_fl_lora_moe_tcr.sh
#
###############################################################################
