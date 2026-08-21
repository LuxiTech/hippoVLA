#!/bin/bash

cd /home/user01/jiangnan/starVLA
source /home/user01/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/home/user01/jiangnan/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/home/user01/miniconda3/envs/libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo


host="127.0.0.1"
base_port=5696
unnorm_key="franka"
your_ckpt=results/Checkpoints/0420_libero4in1_RynnBrain8OFT_memory/checkpoints/steps_60000_pytorch_model.pt
# export DEBUG=false

max_memory=5
interval=10

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}


task_suite_name=libero_spatial  # switched from libero_10 for quick debug
num_trials_per_task=50           # reduced for quick debug
video_out_path="results/${task_suite_name}/${folder_name}"


${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero_3.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.max_memory $max_memory \
    --args.interval $interval
