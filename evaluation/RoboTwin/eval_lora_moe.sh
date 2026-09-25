#!/usr/bin/env bash

###############################################################################
################################# ENV config ##################################
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

WANDB_TOKEN=${WANDB_TOKEN}

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV=internvla
#CONDA_ENV=lerobot_lab
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-2}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

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
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## EVAL config ####################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd ${PROJ_ROOT}

# LoRA+MoE checkpoint path (must contain config.json and model.safetensors)
LORAMOE_CKPT="${LORAMOE_CKPT:-path/to/loramoe/checkpoint}"
BASE_MODEL_PATH=${BASE_MODEL_PATH:-InternRobotics/InternVLA-A1-3B}

COMPILE_INFERENCE=${COMPILE_INFERENCE:-false}

# LoRA+MoE extra params (keep consistent with training)
LORAMOE_NUM_EXPERTS=${LORAMOE_NUM_EXPERTS:-8}
LORAMOE_ROUTER_TOP_K=${LORAMOE_ROUTER_TOP_K:-4}
LORAMOE_ENABLE_A_EXPERTS=${LORAMOE_ENABLE_A_EXPERTS:-true}
LORAMOE_ENABLE_B_EXPERTS=${LORAMOE_ENABLE_B_EXPERTS:-true}
LORAMOE_SHARE_A_ACROSS_EXPERTS=${LORAMOE_SHARE_A_ACROSS_EXPERTS:-false}
USE_VISUAL_TOKEN_PRUNE=${USE_VISUAL_TOKEN_PRUNE:-false}
DTYPE=${DTYPE:-bfloat16}
LOG_LEVEL=${LOG_LEVEL:-WARNING}

# Affordance parameters (match current Affordance training; set ENABLE_AFFORDANCE=false for old checkpoints)
ENABLE_AFFORDANCE=${ENABLE_AFFORDANCE:-true}
LAMBDA_AFFORDANCE=${LAMBDA_AFFORDANCE:-0.001}
AFFORDANCE_DIM=${AFFORDANCE_DIM:-128}
ENABLE_AFFORDANCE_V2=${ENABLE_AFFORDANCE_V2:-false}
AFFORDANCE_V3_STATE_ROUTER_PRIOR=${AFFORDANCE_V3_STATE_ROUTER_PRIOR:-true}
DISABLE_AFFORDANCE_ROUTER_INPUT=${DISABLE_AFFORDANCE_ROUTER_INPUT:-false}
AFFORDANCE_ACTION_HORIZON=${AFFORDANCE_ACTION_HORIZON:-15}
AFFORDANCE_ACTION_DECAY_END=${AFFORDANCE_ACTION_DECAY_END:-30}
AFFORDANCE_GATE_INIT=${AFFORDANCE_GATE_INIT:-0.05}
AFFORDANCE_ROUTER_LEAD_IN_FRACTION=${AFFORDANCE_ROUTER_LEAD_IN_FRACTION:-0.05}

# AB routing: two routers select lora_A and lora_B respectively
LORAMOE_AB_ROUTING=${LORAMOE_AB_ROUTING:-false}
LORAMOE_ROUTER_TOP_K_A=${LORAMOE_ROUTER_TOP_K_A:-4}
LORAMOE_ROUTER_TOP_K_B=${LORAMOE_ROUTER_TOP_K_B:-4}

BASE_OUTPUT_PATH=${PROJ_ROOT}/evaluation/RoboTwin/output
RUN_NAME=${RUN_NAME:-affordance_loramoe}
TASK_CONFIG=${TASK_CONFIG:-demo_randomized}
TASK_IDX=${TASK_IDX:-1}
TASK_INDICES=${TASK_INDICES:-"1,3,4,10,11,15,18,21,25,31,33,39,41,49"}
TEST_NUM=${TEST_NUM:-100}
RESULT_MARKDOWN_PATH=${RESULT_MARKDOWN_PATH:-${PROJ_ROOT}/evaluation/RoboTwin/output/results_${RUN_NAME}.markdown}
GPU_TAG=${GPU_TAG:-gpu${CUDA_VISIBLE_DEVICES}}

RESUME_TEST_NUM=${RESUME_TEST_NUM:-0}
RESUME_SUCCESS=${RESUME_SUCCESS:-0}
RESUME_NEXT_SEED=${RESUME_NEXT_SEED:-}

RESUME_FLAGS=()
if (( RESUME_TEST_NUM > 0 )); then
    RESUME_FLAGS+=(--args.resume-test-num "${RESUME_TEST_NUM}")
    RESUME_FLAGS+=(--args.resume-success "${RESUME_SUCCESS}")
    if [[ -n "${RESUME_NEXT_SEED}" ]]; then
        RESUME_FLAGS+=(--args.resume-next-seed "${RESUME_NEXT_SEED}")
    fi
fi

echo "Affordance V2: ${ENABLE_AFFORDANCE_V2}, horizon=${AFFORDANCE_ACTION_HORIZON}, decay_end=${AFFORDANCE_ACTION_DECAY_END}, gate_init=${AFFORDANCE_GATE_INIT}"

cd ${PROJ_ROOT}/third_party/RoboTwin

# Build ab_routing args
if [ "$LORAMOE_AB_ROUTING" = "true" ]; then
    AB_ROUTING_FLAG="--args.loramoe-ab-routing"
else
    AB_ROUTING_FLAG="--args.no-loramoe-ab-routing"
fi

if [ "$ENABLE_AFFORDANCE" = "true" ]; then
    AFFORDANCE_FLAG="--args.enable-affordance"
else
    AFFORDANCE_FLAG="--args.no-enable-affordance"
fi

if [ "$ENABLE_AFFORDANCE_V2" = "true" ]; then
    AFFORDANCE_V2_FLAG="--args.enable-affordance-v2"
else
    AFFORDANCE_V2_FLAG="--args.no-enable-affordance-v2"
fi

if [ "$AFFORDANCE_V3_STATE_ROUTER_PRIOR" = "true" ]; then
    AFFORDANCE_V3_STATE_ROUTER_FLAG="--args.affordance-v3-state-router-prior"
else
    AFFORDANCE_V3_STATE_ROUTER_FLAG="--args.no-affordance-v3-state-router-prior"
fi

if [ "$DISABLE_AFFORDANCE_ROUTER_INPUT" = "true" ]; then
    DISABLE_AFFORDANCE_ROUTER_INPUT_FLAG="--args.disable-affordance-router-input"
else
    DISABLE_AFFORDANCE_ROUTER_INPUT_FLAG="--args.no-disable-affordance-router-input"
fi

if [ "$LORAMOE_ENABLE_A_EXPERTS" = "true" ]; then
    ENABLE_A_FLAG="--args.loramoe-enable-a-experts"
else
    ENABLE_A_FLAG="--args.no-loramoe-enable-a-experts"
fi

if [ "$LORAMOE_ENABLE_B_EXPERTS" = "true" ]; then
    ENABLE_B_FLAG="--args.loramoe-enable-b-experts"
else
    ENABLE_B_FLAG="--args.no-loramoe-enable-b-experts"
fi

if [ "$LORAMOE_SHARE_A_ACROSS_EXPERTS" = "true" ]; then
    SHARE_A_FLAG="--args.loramoe-share-a-across-experts"
else
    SHARE_A_FLAG="--args.no-loramoe-share-a-across-experts"
fi

if [ "$USE_VISUAL_TOKEN_PRUNE" = "true" ]; then
    VISUAL_PRUNE_FLAG="--args.use-visual-token-prune"
else
    VISUAL_PRUNE_FLAG="--args.no-use-visual-token-prune"
fi

if [ -n "$TASK_INDICES" ]; then
    IFS=',' read -ra TASK_LIST <<< "$TASK_INDICES"
else
    TASK_LIST=("$TASK_IDX")
fi

for RUN_TASK_IDX in "${TASK_LIST[@]}"; do
    RUN_TASK_IDX="${RUN_TASK_IDX// /}"
    if [ -z "$RUN_TASK_IDX" ]; then
        continue
    fi

    OUTPUT_PATH=${BASE_OUTPUT_PATH}/${TASK_CONFIG}/${RUN_TASK_IDX}_${RUN_NAME}_${GPU_TAG}
    echo "Running task ${RUN_TASK_IDX}; markdown will update after completion: ${RESULT_MARKDOWN_PATH}"

    python ../../evaluation/RoboTwin/inference.py \
        --args.ckpt-path $LORAMOE_CKPT \
        --args.base-model-path $BASE_MODEL_PATH \
        --args.video-dir $OUTPUT_PATH \
        --args.task-config $TASK_CONFIG \
        --args.task-idx $RUN_TASK_IDX \
        --args.test-num $TEST_NUM \
        --args.dtype $DTYPE \
        --args.log-level $LOG_LEVEL \
        --args.loramoe-num-experts $LORAMOE_NUM_EXPERTS \
        --args.loramoe-router-top-k $LORAMOE_ROUTER_TOP_K \
        $AB_ROUTING_FLAG \
        --args.loramoe-router-top-k-a $LORAMOE_ROUTER_TOP_K_A \
        --args.loramoe-router-top-k-b $LORAMOE_ROUTER_TOP_K_B \
        $ENABLE_A_FLAG \
        $ENABLE_B_FLAG \
         $SHARE_A_FLAG \
         $VISUAL_PRUNE_FLAG \
         $AFFORDANCE_FLAG \
         $AFFORDANCE_V2_FLAG \
          $AFFORDANCE_V3_STATE_ROUTER_FLAG \
          $DISABLE_AFFORDANCE_ROUTER_INPUT_FLAG \
         --args.lambda-affordance $LAMBDA_AFFORDANCE \
         --args.affordance-dim $AFFORDANCE_DIM \
         --args.affordance-action-horizon $AFFORDANCE_ACTION_HORIZON \
         --args.affordance-action-decay-end $AFFORDANCE_ACTION_DECAY_END \
         --args.affordance-gate-init $AFFORDANCE_GATE_INIT \
         --args.affordance-router-lead-in-fraction $AFFORDANCE_ROUTER_LEAD_IN_FRACTION \
         --args.result-markdown-path $RESULT_MARKDOWN_PATH \
         "${RESUME_FLAGS[@]}" \
         $(if [ "$COMPILE_INFERENCE" = "false" ]; then echo "--args.no-compile-inference"; fi)
done
