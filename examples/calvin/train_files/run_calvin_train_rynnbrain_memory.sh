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
export TOKENIZERS_PARALLELISM=false
ulimit -n 65535 2>/dev/null || true
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=RynnBrainOFT
freeze_module_list=''
# base_vlm=playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action
base_vlm=/home/user01/jiangnan/starVLA/playground/Pretrained_models/RynnBrain-CoP-8B
config_yaml=/home/user01/jiangnan/starVLA/examples/calvin/train_files/starvla_train_calvin_rynnbrain_memory.yaml
# curriculum_path=/home/user01/jiangnan/starVLA/examples/calvin/train_files/curriculum_calvin_abcd/calvin_curriculum.json
# action_model_type was L1RegressionActionHead originally
DIT_TYPE="DiT-B"
calvin_data_root=/mnt/data/jiangnan/lerobot
data_mix=calvin_task_ABC_D
run_root_dir=./results/Checkpoints
run_id=starvla_rynnbrain_calvin_task_ABC_D_memory_dit_inter5_step5_0514
export action_input_dim=2048

NUM_PROCESSES=8
CPU_THREADS_PER_PROCESS=4
PER_DEVICE_BATCH_SIZE=4
GRAD_ACCUM_STEPS=8
#original memory steps is 10
MEMORY_STEPS=5
MEMORY_INTERVAL=5
NUM_WORKERS_PER_PROCESS=1
PREFETCH_FACTOR=2
EVAL_INTERVAL=1000
SAVE_INTERVAL=5000
LOGGING_FREQUENCY=10
MAX_TRAIN_STEPS=100000
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --num_cpu_threads_per_process ${CPU_THREADS_PER_PROCESS} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.memory True \
  --framework.qwenvl.max_memory_step ${MEMORY_STEPS} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${calvin_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --datasets.vla_data.memory True \
  --datasets.vla_data.max_step ${MEMORY_STEPS} \
  --datasets.vla_data.interval ${MEMORY_INTERVAL} \
  --datasets.vla_data.video_backend torchvision_av \
  --datasets.vla_data.curriculum_enabled False \
  --datasets.vla_data.curriculum_path ${curriculum_path} \
  --datasets.vla_data.num_workers ${NUM_WORKERS_PER_PROCESS} \
  --datasets.vla_data.pin_memory True \
  --datasets.vla_data.persistent_workers True \
  --datasets.vla_data.prefetch_factor ${PREFETCH_FACTOR} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps ${MAX_TRAIN_STEPS} \
  --trainer.save_interval ${SAVE_INTERVAL} \
  --trainer.logging_frequency ${LOGGING_FREQUENCY} \
  --trainer.eval_interval ${EVAL_INTERVAL} \
  --trainer.is_resume True \
  --trainer.gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --trainer.learning_rate.base 1.0e-05 \
  --trainer.learning_rate.vlm_interface 5.0e-06 \
  --trainer.learning_rate.action_model 5.0e-05 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project Calvin_ABCD_RynnBrain_memory \
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
