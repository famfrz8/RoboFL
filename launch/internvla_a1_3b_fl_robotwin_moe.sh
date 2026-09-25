#!/usr/bin/env bash
set -euo pipefail

###############################################################################
############ RoboTwin Federated Learning + LoRA-MoE Launch Script #############
#                                                                             #
# Default: 3 GPUs = 3 clients, LoRA-MoE, partition by task category                         #
#                                                                             #
# Training flow:                                                                    #
#   1. Client local training (standard LoRA)                                              #
#   2. Client weights -> MoE experts                                              #
#   3. MoE training (router + experts)                                             #
#   4. Aggregate updated experts and redistribute                                                #
#   5. Local LoRA = mean(Global Expert, Local LoRA)                            #
#   6. Next round loop                                                          #
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
export FL_NUM_CLIENTS="${FL_NUM_CLIENTS:-4}"              # Total number of clients (also the number of MoE experts)
export NUM_GPUS="${NUM_GPUS:-3}"                          # number of GPUs actually used
export FL_LOCAL_STEPS="${FL_LOCAL_STEPS:-100}"            # local training steps per round
export FL_LOCAL_EPOCHS="${FL_LOCAL_EPOCHS:-1}"            # local epochs per round
export FL_NUM_ROUNDS="${FL_NUM_ROUNDS:-50}"               # total FL rounds
export FL_PARTITION="${FL_PARTITION:-by_episode}"         # by_task, by_episode, dirichlet
export FL_DIRICHLET_ALPHA="${FL_DIRICHLET_ALPHA:-0.5}"    # Dirichlet alpha
export FL_HELD_OUT_TASKS="${FL_HELD_OUT_TASKS:-5}"        # number of held-out validation tasks

# Other
export FL_SAVE_FREQ="${FL_SAVE_FREQ:-10}"
export FL_LOG_FREQ="${FL_LOG_FREQ:-200}"
export FL_SEED="${FL_SEED:-42}"

###############################################################################
############################ MoE PARAMETERS ###################################

# MoE core params
export MOE_STEPS="${MOE_STEPS:-50}"                       # MoE training steps per round
export MOE_ROUTER_TYPE="${MOE_ROUTER_TYPE:-soft}"         # router type: soft, topk, noisy_topk
export MOE_K_EXPERTS="${MOE_K_EXPERTS:-4}"                # Top-K experts (for topk/noisy_topk)
export MOE_TEMPERATURE="${MOE_TEMPERATURE:-1.0}"          # router temperature
export MOE_LOSS_WEIGHT="${MOE_LOSS_WEIGHT:-0.01}"         # load-balancing loss weight (note: currently unused in code)
export MOE_LOAD_BALANCE_WEIGHT="${MOE_LOAD_BALANCE_WEIGHT:-0.01}"  # load-balancing loss weight
export MOE_EXPERT_DROPOUT="${MOE_EXPERT_DROPOUT:-0.0}"    # Expert dropout
export MOE_SAVE_FREQ="${MOE_SAVE_FREQ:-5}"                 # MoE checkpoint save frequency (rounds)
export MOE_BATCH_SIZE="${MOE_BATCH_SIZE:-1}"               # MoE training batch size (independent of client)

###############################################################################
############################ LoRA PARAMETERS ##################################

export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

###############################################################################
############################# TCR PARAMETERS ##################################

export TCR_ENABLE="${TCR_ENABLE:-false}"
export TCR_LAMBDA_PROTO="${TCR_LAMBDA_PROTO:-0.5}"
export TCR_LAMBDA_CONTRAST="${TCR_LAMBDA_CONTRAST:-0.5}"
export TCR_MARGIN="${TCR_MARGIN:-0.1}"
export TCR_TAU_KEEP="${TCR_TAU_KEEP:-0.0}"
export TCR_USE_GEN_FEATURE="${TCR_USE_GEN_FEATURE:-true}"
export TCR_PROTOTYPE_MOMENTUM="${TCR_PROTOTYPE_MOMENTUM:-1.0}"

###############################################################################
########################### MODEL & DATASET ###################################

POLICY="qwena1"
PRETRAINED_PATH="path/to/InternVLA-A1-3B"

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

###############################################################################
############################# OUTPUT CONFIG ###################################

BASE_OUTPUT_DIR="outputs/${POLICY}"
JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-fl-moe-${MOE_ROUTER_TYPE}-${FL_NUM_CLIENTS}clients-r${LORA_RANK}-moe${MOE_STEPS}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
FL_OUTPUT_DIR="${OUTPUT_DIR}/global_model"
export FL_OUTPUT_DIR="${FL_OUTPUT_DIR}"

echo "=============================================="
echo "ROBOTWIN FEDERATED LEARNING + LoRA-MoE"
echo "=============================================="
echo "  Clients (Experts): ${FL_NUM_CLIENTS}"
echo "  GPU processes: ${NUM_PROCESSES}"
echo "  Partition: ${FL_PARTITION}"
echo "  Local Steps: ${FL_LOCAL_STEPS}, FL Rounds: ${FL_NUM_ROUNDS}"
echo "  LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  MoE: router=${MOE_ROUTER_TYPE}, k=${MOE_K_EXPERTS}, steps=${MOE_STEPS}"
echo "  TCR: enable=${TCR_ENABLE}, lambda_proto=${TCR_LAMBDA_PROTO}, lambda_contrast=${TCR_LAMBDA_CONTRAST}"
echo "  MoE LR: router=${MOE_LR_ROUTER}, experts=${MOE_LR_EXPERTS}"
echo "  MoE Save Freq: ${MOE_SAVE_FREQ} rounds, Batch Size: ${MOE_BATCH_SIZE}"
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
    src/lerobot/scripts/lerobot_fl_robotwin_moe.py

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
    --policy.optimizer_lr=5.0e-4
    --policy.scheduler_warmup_steps=2000
    --policy.scheduler_decay_steps=100000
    --policy.scheduler_decay_lr=5.0e-6
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l

    # ---- LoRA Settings ----
    --policy.use_lora=true
    --policy.lora_rank=${LORA_RANK}
    --policy.lora_alpha=${LORA_ALPHA}
    --policy.lora_dropout=${LORA_DROPOUT}

    # ---- LoRA-MoE Settings ----
    --policy.enable_loramoe=true
    --policy.loramoe_num_experts=${FL_NUM_CLIENTS}
    --policy.loramoe_router_top_k=${MOE_K_EXPERTS}
    --policy.enable_tcr=${TCR_ENABLE}
    --policy.tcr_lambda_proto=${TCR_LAMBDA_PROTO}
    --policy.tcr_lambda_contrast=${TCR_LAMBDA_CONTRAST}
    --policy.tcr_margin=${TCR_MARGIN}
    --policy.tcr_tau_keep=${TCR_TAU_KEEP}
    --policy.tcr_use_gen_feature=${TCR_USE_GEN_FEATURE}
    --policy.tcr_prototype_momentum=${TCR_PROTOTYPE_MOMENTUM}

    # ---- Dataset ----
    --dataset.type=${POLICY}
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=${USE_EXTERNAL_STATS}
    --dataset.external_stats_path=${HF_HOME}/lerobot/stats/aloha/${ACTION_TYPE}/stats.json

    # ---- Training ----
    --seed=42
    --batch_size=1
    --steps=$((FL_NUM_ROUNDS * FL_LOCAL_STEPS * FL_LOCAL_EPOCHS))
    --save_freq=10000
    --resume=false

    # ---- Logging ----
    --wandb.enable=true
    --wandb.project=lerobot_fl_robottwin_moe
    --wandb.mode=offline
)

echo "Launching LoRA-MoE Federated Learning..."
echo ""
echo "Training Flow:"
echo "  Round Start -> Client Train -> Update Expert -> MoE Train -> Aggregate -> Mean -> Next Round"
echo ""

# DeepSpeed is not needed for this FL+LoRA-MoE training
# The distributed MoE training uses regular DDP via Accelerator
accelerate launch "${ARGS[@]}"


###############################################################################
################################ USAGE ########################################
#
# 1. Default: 3 GPUs, 4 clients, LoRA-MoE with soft routing
#    bash launch/internvla_a1_3b_fl_robotwin_moe.sh
#
# 2. Use Top-K routing (more efficient)
#    MOE_ROUTER_TYPE=topk MOE_K_EXPERTS=2 bash launch/internvla_a1_3b_fl_robotwin_moe.sh
#
# 3. Adjust MoE training strength
#    MOE_STEPS=100 MOE_LR_ROUTER=5e-4 bash launch/internvla_a1_3b_fl_robotwin_moe.sh
#
# 4. Increase load-balancing weight (prevents expert collapse)
#    MOE_LOSS_WEIGHT=0.1 bash launch/internvla_a1_3b_fl_robotwin_moe.sh
#
# 5. More clients + larger LoRA
#    FL_NUM_CLIENTS=8 LORA_RANK=16 bash launch/internvla_a1_3b_fl_robotwin_moe.sh
#
# 6. Quick test (5 rounds)
#    FL_NUM_ROUNDS=5 FL_LOCAL_STEPS=20 MOE_STEPS=10 bash launch/internvla_a1_3b_fl_robotwin_moe.sh
#
# 7. Non-IID data + MoE (MoE handles non-IID well)
#    FL_PARTITION=dirichlet FL_DIRICHLET_ALPHA=0.3 bash launch/internvla_a1_3b_fl_robotwin_moe.sh
#
# 8. Set MoE save frequency and batch size independently
#    MOE_SAVE_FREQ=10 MOE_BATCH_SIZE=2 bash launch/internvla_a1_3b_fl_robotwin_moe.sh
#
# 9. Enable TCR
#    TCR_ENABLE=true TCR_LAMBDA_PROTO=0.2 TCR_LAMBDA_CONTRAST=1.0 bash launch/internvla_a1_3b_fl_robotwin_moe.sh
#
###############################################################################
