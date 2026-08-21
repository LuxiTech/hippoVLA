# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# # used for check save when communication
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
# export NCCL_SOCKET_TIMEOUT_MS=360000
export TMPDIR=${TMPDIR:-/tmp}
ulimit -n 65535 2>/dev/null || true
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=RynnBrainOFT
freeze_module_list=''
# base_vlm=playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action
base_vlm=/home/user01/jiangnan/starVLA/playground/Pretrained_models/RynnBrain-CoP-8B
# config_yaml=./examples/calvin/train_files/starvla_train_calvin.yaml
config_yaml=/home/user01/jiangnan/starVLA/examples/calvin/train_files/starvla_train_calvin_rynnbrain.yaml
DIT_TYPE="DiT-B"
calvin_data_root=/mnt/data/jiangnan/lerobot
data_mix=calvin_task_ABC_D
run_root_dir=./results/Checkpoints
run_id=starvla_rynnbrain_calvin_task_ABC_D_nomemory
export action_input_dim=2048
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.memory False \
  --datasets.vla_data.data_root_dir ${calvin_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --datasets.vla_data.memory False \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.is_resume True \
  --trainer.gradient_accumulation_steps 8 \
  --trainer.learning_rate.base 1.0e-05 \
  --trainer.learning_rate.qwen_vl_interface 5.0e-06 \
  --trainer.learning_rate.action_model 5.0e-05 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project Calvin_ABCD_RynnBrain \
  --wandb_entity rorschachkelvin-luxi-tech

#  --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
