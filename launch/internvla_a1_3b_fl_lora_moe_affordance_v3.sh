#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ENABLE_AFFORDANCE_V3=true
export AFFORDANCE_ROUTER_LEAD_IN_FRACTION=0.0

# Keep weighted aggregation, but isolate it from the previous FARD/PCEA and
# visual-pruning ablations.
export MOE_ROUTER_WEIGHTED_AGGREGATION=true
export ENABLE_FARD=false
export ENABLE_PCEA=false
export USE_VISUAL_TOKEN_PRUNE=false

export LAMBDA_AFFORDANCE="${LAMBDA_AFFORDANCE:-0.001}"

exec bash "${SCRIPT_DIR}/internvla_a1_3b_fl_lora_moe_routerweighted.sh" "$@"
