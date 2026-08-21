#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=$(pwd):${PYTHONPATH:-}

export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_HCA=mlx5_2,mlx5_3

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# === Robolatent config (no memory) ===
Framework_name=RynnBrainOFT
freeze_module_list=''

base_vlm=playground/Pretrained_models/RynnBrain-CoP-8B

config_yaml=examples/Robolatent/train_files/starvla_robolatent_rynnbrainoft.yaml

robolatent_data_root=/root/autodl-tmp/datasets
data_mix=robolatent_pickxtimes
run_root_dir=./results/Checkpoints
run_id=robolatent_pickxtimes_RynnBrainOFT_nomemory
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.memory false \
  --framework.action_model.action_dim 7 \
  --framework.action_model.state_dim 7 \
  --framework.action_model.future_action_window_size 7 \
  --framework.action_model.num_actions_chunk 8 \
  --datasets.vla_data.data_root_dir ${robolatent_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.action_type joint \
  --datasets.vla_data.per_device_batch_size 4 \
  --datasets.vla_data.video_backend decord \
  --datasets.vla_data.memory false \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 20 \
  --trainer.eval_interval 500 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Robolatent \
  --wandb_entity zkril-cug
