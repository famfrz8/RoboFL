#!/usr/bin/env bash

###############################################################################
################################# ENV config ##################################
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
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

# LoRA checkpoint path
LORA_CKPT="${LORA_CKPT:-path/to/lora/checkpoint}"
# Base model path (used to load LoRA)
BASE_MODEL_PATH=${BASE_MODEL_PATH:-path/to/InternVLA-A1-3B}

COMPILE_INFERENCE=${COMPILE_INFERENCE:-false}
USE_VISUAL_TOKEN_PRUNE=${USE_VISUAL_TOKEN_PRUNE:-false}
DTYPE=${DTYPE:-bfloat16}
LOG_LEVEL=${LOG_LEVEL:-WARNING}
TEST_NUM=${TEST_NUM:-100}

BASE_OUTPUT_PATH=${PROJ_ROOT}/evaluation/RoboTwin/output
TASK_CONFIG=${TASK_CONFIG:-demo_randomized}
TASK_IDX=${TASK_IDX:-9}
TASK_INDICES=${TASK_INDICES:-""}  # optional, multiple tasks e.g. "0,1,2" or range "0-10"
RUN_NAME=${RUN_NAME:-lora_5w}
RESULT_MARKDOWN_PATH=${RESULT_MARKDOWN_PATH:-${PROJ_ROOT}/evaluation/RoboTwin/output/results_${RUN_NAME}.markdown}
GPU_TAG=${GPU_TAG:-gpu${CUDA_VISIBLE_DEVICES}}

if [ "${USE_VISUAL_TOKEN_PRUNE}" = "true" ]; then
    VISUAL_TOKEN_PRUNE_ARG="--args.use-visual-token-prune"
else
    VISUAL_TOKEN_PRUNE_ARG="--args.no-use-visual-token-prune"
fi

if [ -n "$TASK_INDICES" ]; then
    IFS=',' read -ra TASK_LIST <<< "$TASK_INDICES"
else
    TASK_LIST=("$TASK_IDX")
fi

cd ${PROJ_ROOT}/third_party/RoboTwin
for RUN_TASK_IDX in "${TASK_LIST[@]}"; do
    RUN_TASK_IDX="${RUN_TASK_IDX// /}"
    if [ -z "$RUN_TASK_IDX" ]; then
        continue
    fi

    OUTPUT_PATH=${BASE_OUTPUT_PATH}/${TASK_CONFIG}/${RUN_TASK_IDX}_${RUN_NAME}_${GPU_TAG}
    echo "Running task ${RUN_TASK_IDX}; markdown will update after completion: ${RESULT_MARKDOWN_PATH}"

    python ../../evaluation/RoboTwin/inference.py \
        --args.ckpt-path $LORA_CKPT \
        --args.base-model-path $BASE_MODEL_PATH \
        --args.video-dir $OUTPUT_PATH \
        --args.task-config $TASK_CONFIG \
        --args.task-idx $RUN_TASK_IDX \
        --args.test-num $TEST_NUM \
        --args.dtype $DTYPE \
        --args.log-level $LOG_LEVEL \
        --args.no-enable-affordance \
        $VISUAL_TOKEN_PRUNE_ARG \
        --args.result-markdown-path $RESULT_MARKDOWN_PATH \
        $(if [ "$COMPILE_INFERENCE" = "false" ]; then echo "--args.no-compile-inference"; fi)
done
