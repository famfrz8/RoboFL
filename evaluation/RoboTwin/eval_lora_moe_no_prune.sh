#!/usr/bin/env bash

###############################################################################
################################# ENV config ##################################
export CUDA_VISIBLE_DEVICES=3
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

WANDB_TOKEN=${WANDB_TOKEN}

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV=internvla
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

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
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

###############################################################################
############################## EVAL config ####################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd ${PROJ_ROOT}

LORAMOE_CKPT="${LORAMOE_CKPT:-path/to/loramoe/checkpoint}"
BASE_MODEL_PATH=${BASE_MODEL_PATH:-path/to/InternVLA-A1-3B}
COMPILE_INFERENCE=${COMPILE_INFERENCE:-true}

LORAMOE_NUM_EXPERTS=${LORAMOE_NUM_EXPERTS:-8}
LORAMOE_ROUTER_TOP_K=${LORAMOE_ROUTER_TOP_K:-4}
LORAMOE_AB_ROUTING=${LORAMOE_AB_ROUTING:-false}
LORAMOE_ROUTER_TOP_K_A=${LORAMOE_ROUTER_TOP_K_A:-4}
LORAMOE_ROUTER_TOP_K_B=${LORAMOE_ROUTER_TOP_K_B:-4}

BASE_OUTPUT_PATH=${PROJ_ROOT}/evaluation/RoboTwin/output
TASK_CONFIG=${TASK_CONFIG:-demo_randomized}
TASK_IDX=${TASK_IDX:-40}
TASK_INDICES=${TASK_INDICES:-"40,42,43,45,46,47"}
RESULT_MARKDOWN_PATH=${RESULT_MARKDOWN_PATH:-${PROJ_ROOT}/evaluation/RoboTwin/output/results_loramoe_routerweight_agg_5w.markdown}

cd ${PROJ_ROOT}/third_party/RoboTwin

if [ "$LORAMOE_AB_ROUTING" = "true" ]; then
    AB_ROUTING_FLAG="--args.loramoe_ab_routing"
else
    AB_ROUTING_FLAG="--args.no_loramoe_ab_routing"
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

    OUTPUT_PATH=${BASE_OUTPUT_PATH}/${TASK_CONFIG}/${RUN_TASK_IDX}_noprune_routerweightedagg_5w_gpu3
    echo "Running task ${RUN_TASK_IDX} without visual token pruning; markdown will update after completion: ${RESULT_MARKDOWN_PATH}"

    python ../../evaluation/RoboTwin/inference.py \
        --args.ckpt_path $LORAMOE_CKPT \
        --args.base_model_path $BASE_MODEL_PATH \
        --args.video_dir $OUTPUT_PATH \
        --args.task_config $TASK_CONFIG \
        --args.task_idx $RUN_TASK_IDX \
        --args.loramoe_num_experts $LORAMOE_NUM_EXPERTS \
        --args.loramoe_router_top_k $LORAMOE_ROUTER_TOP_K \
        $AB_ROUTING_FLAG \
        --args.no_use_visual_token_prune \
        --args.loramoe_router_top_k_a $LORAMOE_ROUTER_TOP_K_A \
        --args.loramoe_router_top_k_b $LORAMOE_ROUTER_TOP_K_B \
        --args.result_markdown_path $RESULT_MARKDOWN_PATH \
        $(if [ "$COMPILE_INFERENCE" = "false" ]; then echo "--args.no-compile-inference"; fi)
done
