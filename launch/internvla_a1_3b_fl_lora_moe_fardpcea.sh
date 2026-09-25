#!/usr/bin/env bash
set -euo pipefail

###############################################################################
############ LoRA-MoE Federated Learning + FARD/PCEA Launch Script ############
#                                                                             #
# Default: 8 GPUs, 8 clients, FedForesight LoRA-MoE, task-category partition  #
#                                                                             #
# This is a self-contained launcher. It applies the FARD/PCEA-oriented        #
# affinity settings on top of the base router-weighted LoRA-MoE defaults.     #
###############################################################################

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

###############################################################################
#################### FARD / PCEA affinity overrides ##########################
# These values mirror the previous affordance_v3 wrapper and isolate this run
# from the earlier FARD/PCEA and visual-pruning ablations. Override any of them
# from the environment if a different configuration is needed.

export ENABLE_AFFORDANCE_V3="${ENABLE_AFFORDANCE_V3:-true}"
export AFFORDANCE_ROUTER_LEAD_IN_FRACTION="${AFFORDANCE_ROUTER_LEAD_IN_FRACTION:-0.0}"
export MOE_ROUTER_WEIGHTED_AGGREGATION="${MOE_ROUTER_WEIGHTED_AGGREGATION:-true}"
export ENABLE_FARD="${ENABLE_FARD:-false}"
export ENABLE_PCEA="${ENABLE_PCEA:-false}"
export USE_VISUAL_TOKEN_PRUNE="${USE_VISUAL_TOKEN_PRUNE:-false}"
export LAMBDA_AFFORDANCE="${LAMBDA_AFFORDANCE:-0.001}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV=internvla_a1
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export WANDB_MODE=online
: "${WANDB_API_KEY:?Set WANDB_API_KEY in the remote shell before launching}"
wandb login

# wandb login ${WANDB_TOKEN}

###############################################################################
########################## Distributed Config #################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29502}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

export NUM_GPUS="${NUM_GPUS:-8}"
PROC_PER_NODE="${NUM_GPUS}"
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
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

###############################################################################
####################### LoRA-MoE FL PARAMETERS ###############################

# Core FL parameters
export ENABLE_FEDFORESIGHT="${ENABLE_FEDFORESIGHT:-true}"
export FL_NUM_CLIENTS="${FL_NUM_CLIENTS:-8}"
export FL_LOCAL_STEPS="${FL_LOCAL_STEPS:-500}"
export FL_LOCAL_EPOCHS="${FL_LOCAL_EPOCHS:-1}"
export FL_NUM_ROUNDS="${FL_NUM_ROUNDS:-100}"
export FL_AGGREGATION="${FL_AGGREGATION:-fedavg}"
export FL_FEDPROX_MU="${FL_FEDPROX_MU:-0.01}"

# MoE parameters
export MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-8}"
export MOE_K_EXPERTS="${MOE_K_EXPERTS:-4}"
export MOE_STEPS="${MOE_STEPS:-500}"          # MoE local training steps per round
export MOE_BATCH_SIZE="${MOE_BATCH_SIZE:-8}"  # MoE training batch size
export MOE_ROUTER_WEIGHTED_AGGREGATION="${MOE_ROUTER_WEIGHTED_AGGREGATION:-false}"

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
# export LAMBDA_AUX="${LAMBDA_AUX:-0.001}"
export LAMBDA_AUX="${LAMBDA_AUX:-0.001}"
export LAMBDA_OT="${LAMBDA_OT:-0.01}"

# Router-guided adaptive visual token pruning
export USE_VISUAL_TOKEN_PRUNE="${USE_VISUAL_TOKEN_PRUNE:-false}"
export USE_VISUAL_TOKEN_PRUNE_PRINT_SUMMARY="${USE_VISUAL_TOKEN_PRUNE_PRINT_SUMMARY:-true}"
export USE_VISUAL_TOKEN_PRUNE_PRINT_STATS="${USE_VISUAL_TOKEN_PRUNE_PRINT_STATS:-false}"

# FedForesight path-consensus routing alignment
export ENABLE_FARD="${ENABLE_FARD:-true}"
export LAMBDA_FARD="${LAMBDA_FARD:-0.01}"
export FARD_WARMUP_ROUNDS="${FARD_WARMUP_ROUNDS:-10}"
export ENABLE_PCEA="${ENABLE_PCEA:-false}"

# Future-supervised spatial affordance conditioning
export ENABLE_AFFORDANCE="${ENABLE_AFFORDANCE:-true}"
export LAMBDA_AFFORDANCE="${LAMBDA_AFFORDANCE:-0.001}"
export AFFORDANCE_DIM="${AFFORDANCE_DIM:-128}"
export ENABLE_AFFORDANCE_V2="${ENABLE_AFFORDANCE_V2:-false}"
export ENABLE_AFFORDANCE_V3="${ENABLE_AFFORDANCE_V3:-false}"
export AFFORDANCE_ACTION_HORIZON="${AFFORDANCE_ACTION_HORIZON:-15}"
export AFFORDANCE_ACTION_DECAY_END="${AFFORDANCE_ACTION_DECAY_END:-30}"
export AFFORDANCE_GATE_INIT="${AFFORDANCE_GATE_INIT:-0.05}"
export AFFORDANCE_ROUTER_LEAD_IN_FRACTION="${AFFORDANCE_ROUTER_LEAD_IN_FRACTION:-0.05}"

# FedForesight requires complete A/B experts with one router per module.
export MOE_SHARE_A="${MOE_SHARE_A:-false}"
export MOE_ENABLE_A_EXPERTS="${MOE_ENABLE_A_EXPERTS:-true}"
export MOE_ENABLE_B_EXPERTS="${MOE_ENABLE_B_EXPERTS:-true}"

# Legacy AB Routing: only meaningful when A experts are enabled
export MOE_AB_ROUTING="${MOE_AB_ROUTING:-false}"
export MOE_K_EXPERTS_A="${MOE_K_EXPERTS_A:-4}"  # top-k for Router A (legacy)
export MOE_K_EXPERTS_B="${MOE_K_EXPERTS_B:-4}"  # top-k for Router B (legacy)

###############################################################################
########################### MODEL & DATASET ###################################

POLICY="qwena1"
PRETRAINED_PATH="${PRETRAINED_PATH:-InternRobotics/InternVLA-A1-3B}"

if [[ -z "${DATASET_REPO_ID:-}" ]]; then
    DATASET_REPO_ID="$(
      find -L "data/robotwin" -mindepth 2 -maxdepth 2 -type d -name "aloha-*" 2>/dev/null \
      | while read -r d; do
            if [[ -d "$d/meta" && -d "$d/videos" ]]; then
                echo "${d#data/}"
            fi
        done \
      | sort -u \
      | tr '\n' ' ' \
      | xargs
    )"
fi
ACTION_TYPE="delta"
USE_EXTERNAL_STATS="true"

BASE_OUTPUT_DIR="outputs/${POLICY}"
JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-fl-fedforesight-${FL_NUM_CLIENTS}clients-r${LORA_RANK}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"

echo "=============================================="
echo "ROBOTWIN LoRA-MoE FEDERATED LEARNING"
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
echo "  Router Weighted Aggregation: ${MOE_ROUTER_WEIGHTED_AGGREGATION}"
echo "  FedForesight Enabled: ${ENABLE_FEDFORESIGHT}"
echo "  FARD: ${ENABLE_FARD}, lambda=${LAMBDA_FARD}, warmup_rounds=${FARD_WARMUP_ROUNDS}"
echo "  PCEA: ${ENABLE_PCEA}, hierarchical Hellinger aggregation"
echo "  Affordance: ${ENABLE_AFFORDANCE}, lambda=${LAMBDA_AFFORDANCE}, dim=${AFFORDANCE_DIM}"
echo "  Affordance V2: ${ENABLE_AFFORDANCE_V2}, horizon=${AFFORDANCE_ACTION_HORIZON}, decay_end=${AFFORDANCE_ACTION_DECAY_END}, gate_init=${AFFORDANCE_GATE_INIT}"
echo "  Affordance V3 Router-only: ${ENABLE_AFFORDANCE_V3}"
echo "  Affordance Router Lead-in: ${AFFORDANCE_ROUTER_LEAD_IN_FRACTION}"
echo "  Local Steps: ${FL_LOCAL_STEPS}, FL Rounds: ${FL_NUM_ROUNDS}"
echo "  LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  Aux Loss: lambda_aux=${LAMBDA_AUX}"
echo "  Transport Loss: lambda_ot=${LAMBDA_OT}"
echo "  Visual Token Prune: ${USE_VISUAL_TOKEN_PRUNE}"
echo "  Visual Token Prune Summary: ${USE_VISUAL_TOKEN_PRUNE_PRINT_SUMMARY}"
echo "  Visual Token Per-Layer Stats: ${USE_VISUAL_TOKEN_PRUNE_PRINT_STATS}"
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
    src/lerobot/scripts/lerobot_fl_robotwin_moe_routerweighted.py

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
    --policy.scheduler_warmup_steps=2500
    --policy.scheduler_decay_steps=50000
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
    --policy.lambda_ot=${LAMBDA_OT}
    --policy.enable_fard=${ENABLE_FARD}
    --policy.lambda_fard=${LAMBDA_FARD}
    --policy.fard_warmup_rounds=${FARD_WARMUP_ROUNDS}
    --policy.enable_pcea=${ENABLE_PCEA}
     --policy.enable_affordance=${ENABLE_AFFORDANCE}
     --policy.lambda_affordance=${LAMBDA_AFFORDANCE}
     --policy.affordance_dim=${AFFORDANCE_DIM}
     --policy.enable_affordance_v2=${ENABLE_AFFORDANCE_V2}
     --policy.enable_affordance_v3=${ENABLE_AFFORDANCE_V3}
     --policy.affordance_action_horizon=${AFFORDANCE_ACTION_HORIZON}
     --policy.affordance_action_decay_end=${AFFORDANCE_ACTION_DECAY_END}
     --policy.affordance_gate_init=${AFFORDANCE_GATE_INIT}
     --policy.affordance_router_lead_in_fraction=${AFFORDANCE_ROUTER_LEAD_IN_FRACTION}
    --policy.use_visual_token_prune=${USE_VISUAL_TOKEN_PRUNE}

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
    --wandb.project=lerobot_fl_robottwin_fedforesight
    --wandb.mode=${WANDB_MODE}
)

echo "Launching..."

accelerate launch "${ARGS[@]}"


###############################################################################
################################ USAGE ########################################
#
# 0. Set WandB credentials in the remote shell (do not hardcode in scripts)
#    export WANDB_API_KEY="your-new-wandb-key"
#
# 1. Default: 8 GPUs, 8 clients, FARD/PCEA affinity + router-weighted aggregation
#    bash launch/internvla_a1_3b_fl_lora_moe_fardpcea.sh
#
# 2. Single-round end-to-end Affordance smoke test (20 MoE steps ~ 1 lead-in step)
#    FL_NUM_ROUNDS=1 FL_LOCAL_STEPS=1 MOE_STEPS=20 WANDB_MODE=offline \
#      bash launch/internvla_a1_3b_fl_lora_moe_fardpcea.sh
#
# 3. 8 GPU / 8 clients
#    NUM_GPUS=8 FL_NUM_CLIENTS=8 \
#      bash launch/internvla_a1_3b_fl_lora_moe_fardpcea.sh
#
# 4. Custom FARD, Affordance and Aux loss weights
#    LAMBDA_FARD=0.01 LAMBDA_AFFORDANCE=0.01 LAMBDA_AUX=0.001 \
#      bash launch/internvla_a1_3b_fl_lora_moe_fardpcea.sh
#
# 5. 20-round pilot
#    FL_NUM_ROUNDS=20 FL_LOCAL_STEPS=100 MOE_STEPS=100 \
#      bash launch/internvla_a1_3b_fl_lora_moe_fardpcea.sh
#
# 6. Explicitly set remote RoboTwin repo IDs (space-separated)
#    DATASET_REPO_ID="robotwin/aloha-task1 robotwin/aloha-task2" \
#      bash launch/internvla_a1_3b_fl_lora_moe_fardpcea.sh
#
# 7. FedForesight does not support shared-A or AB routing; invalid config fails fast at startup.
#
###############################################################################
