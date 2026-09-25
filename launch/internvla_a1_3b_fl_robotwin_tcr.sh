#!/usr/bin/env bash
set -euo pipefail

###############################################################################
############ RoboTwin Federated Learning + TCR Router Launch Script ###########
#                                                                             #
# For the current TCR version of LoRA-MoE FL training.                                            #
# All TCR hyperparams go through --policy.tcr_* so the training script does not read ad-hoc env vars.           #
###############################################################################

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-internvla}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

###############################################################################
########################## Distributed Config #################################

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29504}"

PROC_PER_NODE="${NUM_GPUS:-1}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    MULTI_GPU_FLAG="--multi_gpu"
else
    MULTI_GPU_FLAG=""
fi

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

###############################################################################
####################### Federated Learning Parameters #########################

export FL_NUM_CLIENTS="${FL_NUM_CLIENTS:-4}"
export FL_LOCAL_STEPS="${FL_LOCAL_STEPS:-100}"
export FL_LOCAL_EPOCHS="${FL_LOCAL_EPOCHS:-1}"
export FL_NUM_ROUNDS="${FL_NUM_ROUNDS:-50}"
export FL_PARTITION="${FL_PARTITION:-by_episode}"
export FL_DIRICHLET_ALPHA="${FL_DIRICHLET_ALPHA:-0.5}"
export FL_HELD_OUT_TASKS="${FL_HELD_OUT_TASKS:-5}"
export FL_SAVE_FREQ="${FL_SAVE_FREQ:-10}"
export FL_LOG_FREQ="${FL_LOG_FREQ:-200}"
export FL_SEED="${FL_SEED:-42}"

###############################################################################
############################ LoRA-MoE + TCR ###################################

export MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-${FL_NUM_CLIENTS}}"
export MOE_K_EXPERTS="${MOE_K_EXPERTS:-4}"
export MOE_STEPS="${MOE_STEPS:-50}"
export MOE_BATCH_SIZE="${MOE_BATCH_SIZE:-1}"
export MOE_SAVE_FREQ="${MOE_SAVE_FREQ:-5}"
export MOE_AB_ROUTING="${MOE_AB_ROUTING:-false}"
export MOE_K_EXPERTS_A="${MOE_K_EXPERTS_A:-4}"
export MOE_K_EXPERTS_B="${MOE_K_EXPERTS_B:-4}"

export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

export LAMBDA_AUX="${LAMBDA_AUX:-0.0005}"
export TCR_ENABLE="${TCR_ENABLE:-true}"
export TCR_LAMBDA_PROTO="${TCR_LAMBDA_PROTO:-0.5}"
export TCR_LAMBDA_CONTRAST="${TCR_LAMBDA_CONTRAST:-0.5}"
export TCR_MARGIN="${TCR_MARGIN:-0.1}"
export TCR_TAU_KEEP="${TCR_TAU_KEEP:-0.0}"
export TCR_USE_GEN_FEATURE="${TCR_USE_GEN_FEATURE:-true}"
export TCR_PROTOTYPE_MOMENTUM="${TCR_PROTOTYPE_MOMENTUM:-1.0}"
export TCR_LOSS_WEIGHT="${TCR_LOSS_WEIGHT:-0.001}"

###############################################################################
########################### Model & Dataset ###################################

POLICY="qwena1"
PRETRAINED_PATH="${PRETRAINED_PATH:-path/to/InternVLA-A1-3B}"

DATASET_REPO_ID="${DATASET_REPO_ID:-}"
if [[ -z "${DATASET_REPO_ID}" ]]; then
    DATASET_REPO_ID="$({
      find -L "data/robotwin" -mindepth 2 -maxdepth 2 -type d -name "aloha-*" 2>/dev/null \
      | while read -r d; do
            if [[ -d "$d/meta" && -d "$d/videos" ]]; then
                echo "${d#data/}"
            fi
        done \
      | sort -u \
      | head -n 1
    })"
fi

ACTION_TYPE="${ACTION_TYPE:-delta}"
USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
EXTERNAL_STATS_PATH="${EXTERNAL_STATS_PATH:-${HF_HOME}/lerobot/stats/aloha/${ACTION_TYPE}/stats.json}"

if [[ -z "${DATASET_REPO_ID}" ]]; then
    echo "ERROR: No RoboTwin dataset found in data/robotwin/. Set DATASET_REPO_ID manually if needed."
    exit 1
fi

###############################################################################
############################# Output Config ###################################

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-fl-router-tcr-${FL_NUM_CLIENTS}clients-r${LORA_RANK}}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_DIR}/${JOB_NAME}}"
export FL_OUTPUT_DIR="${OUTPUT_DIR}/global_model"

echo "=============================================="
echo "ROBOTWIN FL + ROUTER TCR"
echo "=============================================="
echo "  Dataset: ${DATASET_REPO_ID}"
echo "  Clients: ${FL_NUM_CLIENTS}, GPU processes: ${NUM_PROCESSES}"
echo "  Partition: ${FL_PARTITION}"
echo "  Local Steps: ${FL_LOCAL_STEPS}, MoE Steps: ${MOE_STEPS}, FL Rounds: ${FL_NUM_ROUNDS}"
echo "  LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "  MoE: experts=${MOE_NUM_EXPERTS}, top_k=${MOE_K_EXPERTS}, ab_routing=${MOE_AB_ROUTING}"
echo "  TCR: enable=${TCR_ENABLE}, lambda_proto=${TCR_LAMBDA_PROTO}, lambda_contrast=${TCR_LAMBDA_CONTRAST}, margin=${TCR_MARGIN}, loss_weight=${TCR_LOSS_WEIGHT}"
echo "  Output: ${OUTPUT_DIR}"
echo "=============================================="

###############################################################################
######################### Build Args And Launch ###############################

ARGS=(
    ${MULTI_GPU_FLAG}
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_fl_robotwin_moe.py

    --output_dir="${OUTPUT_DIR}"
    --num_workers="${NUM_WORKERS:-8}"
    --job_name="${JOB_NAME}"

    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.pretrained_path=${PRETRAINED_PATH}
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr="${OPTIMIZER_LR:-1.0e-4}"
    --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS:-2500}"
    --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS:-50000}"
    --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR:-1.0e-5}"
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l

    --policy.use_lora_moe=true
    --policy.use_lora=false
    --policy.loramoe_num_experts=${MOE_NUM_EXPERTS}
    --policy.loramoe_router_top_k=${MOE_K_EXPERTS}
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

    --dataset.type=${POLICY}
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=${USE_EXTERNAL_STATS}
    --dataset.external_stats_path="${EXTERNAL_STATS_PATH}"

    --seed=${FL_SEED}
    --batch_size="${BATCH_SIZE:-32}"
    --steps=$((FL_NUM_ROUNDS * FL_LOCAL_STEPS * FL_LOCAL_EPOCHS))
    --save_freq=10000
    --resume=false

    --wandb.enable="${WANDB_ENABLE:-true}"
    --wandb.project="${WANDB_PROJECT:-lerobot_fl_robotwin_router_tcr}"
    --wandb.mode="${WANDB_MODE}"
)

accelerate launch "${ARGS[@]}"

###############################################################################
################################ Usage ########################################
#
# Default launch:
#   bash launch/internvla_a1_3b_fl_robotwin_tcr.sh
#
# Quick smoke test:
#   FL_NUM_ROUNDS=1 FL_LOCAL_STEPS=1 MOE_STEPS=1 NUM_GPUS=1 bash launch/internvla_a1_3b_fl_robotwin_tcr.sh
#
# Tune TCR weights:
#   TCR_LAMBDA_PROTO=0.2 TCR_LAMBDA_CONTRAST=1.0 TCR_MARGIN=0.2 bash launch/internvla_a1_3b_fl_robotwin_tcr.sh
#
# Disable generation features, use only understanding features:
#   TCR_USE_GEN_FEATURE=false bash launch/internvla_a1_3b_fl_robotwin_tcr.sh
#
###############################################################################
