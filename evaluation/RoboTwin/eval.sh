#!/usr/bin/env bash

###############################################################################
################################# ENV config ##################################
export CUDA_VISIBLE_DEVICES=2
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
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd ${PROJ_ROOT}

PRETRAINED_CKPT="${PRETRAINED_CKPT:-path/to/InternVLA-A1-3B-RoboTwin}"
BASE_OUTPUT_PATH=${PROJ_ROOT}/evaluation/RoboTwin/output
TASK_CONFIG=demo_randomized
TASK_IDX=0 # adjust_bottle
TASK_INDICES=${TASK_INDICES:-""}  # optional, multiple tasks e.g. "0,1,2" or range "0-10"
COMPILE_INFERENCE=${COMPILE_INFERENCE:-true}
USE_PTQ=${USE_PTQ:-false}
PTQ_MODE=${PTQ_MODE:-weight_only_int8}
PTQ_BACKEND=${PTQ_BACKEND:-naive}
PTQ_EXPERTS=${PTQ_EXPERTS:-und gen act}
PTQ_TARGET_MODULES=${PTQ_TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}

OUTPUT_PATH=${BASE_OUTPUT_PATH}/${TASK_CONFIG}/${TASK_IDX}

cd ${PROJ_ROOT}/third_party/RoboTwin
python ../../evaluation/RoboTwin/inference.py \
    --args.ckpt_path $PRETRAINED_CKPT \
    --args.video_dir $OUTPUT_PATH \
    --args.task_config $TASK_CONFIG \
    --args.task_idx $TASK_IDX \
    --args.ptq_mode $PTQ_MODE \
    --args.ptq_backend $PTQ_BACKEND \
    --args.ptq_experts $PTQ_EXPERTS \
    --args.ptq_target_modules $PTQ_TARGET_MODULES \
    $(if [ "$COMPILE_INFERENCE" = "false" ]; then echo "--args.no-compile-inference"; fi) \
    $(if [ "$USE_PTQ" = "true" ]; then echo "--args.use_ptq"; fi) \
    $(if [ -n "$TASK_INDICES" ]; then echo "--args.task_indices $TASK_INDICES"; fi)
