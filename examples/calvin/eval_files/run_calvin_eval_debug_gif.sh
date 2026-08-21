#!/bin/bash

# CKPT=/path/to/pytorch_model.pt \
# PORT=5695 \
# NUM_SEQUENCES=10 \
# bash examples/calvin/eval_files/run_calvin_eval_debug_gif.sh

set -euo pipefail

###########################################################################################
# CALVIN evaluation with debug GIF saving.
#
# This script only runs the CALVIN evaluator. Start the policy server separately first, for
# example with examples/calvin/eval_files/run_policy_server.sh or your own server command.
###########################################################################################

export PYTHONPATH=$(pwd):${PYTHONPATH:-}

# Python used for the CALVIN environment.
CALVIN_PYTHON=${CALVIN_PYTHON:-python}

# Policy server endpoint.
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-5695}

# Model/eval paths. Override these from the shell if needed:
#   CKPT=/path/to/pytorch_model.pt bash examples/calvin/eval_files/run_calvin_eval_debug_gif.sh
CKPT=${CKPT:-/home/user01/jiangnan/starVLA/results/Checkpoints/starvla_rynnbrain_calvin_task_ABC_D_memory_dit_inter10_step10_0513/final_model/pytorch_model.pt}
UNNORM_KEY=${UNNORM_KEY:-franka}
DATASET_PATH=${DATASET_PATH:-/mnt/data/jiangnan/calvin/task_D_D}
CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH:-/mnt/data/jiangnan/calvin/calvin/calvin_models/conf}
EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH:-/home/user01/jiangnan/starVLA/examples/calvin/eval_files/eval_sequences.json}
NUM_SEQUENCES=${NUM_SEQUENCES:-1000}

CKPT_TAG=$(basename "$(dirname "$(dirname "${CKPT}")")")_$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}" .pt)
EVAL_LOG_DIR=${EVAL_LOG_DIR:-/home/user01/jiangnan/starVLA/tmp/calvin/eval_debug_gifs/${CKPT_TAG}_$(date +"%Y%m%d_%H%M%S")}
mkdir -p "${EVAL_LOG_DIR}"

echo "CALVIN debug GIF eval"
echo "  server: ${HOST}:${PORT}"
echo "  ckpt: ${CKPT}"
echo "  logs: ${EVAL_LOG_DIR}"

"${CALVIN_PYTHON}" /home/user01/jiangnan/starVLA/examples/calvin/eval_files/eval_calvin.py \
  --args.pretrained-path "${CKPT}" \
  --args.unnorm-key "${UNNORM_KEY}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.dataset_path "${DATASET_PATH}" \
  --args.calvin_config_path "${CALVIN_CONFIG_PATH}" \
  --args.eval_sequences_path "${EVAL_SEQUENCES_PATH}" \
  --args.num_sequences "${NUM_SEQUENCES}" \
  --args.debug True \
  --args.eval-log-dir "${EVAL_LOG_DIR}"
