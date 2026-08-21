# 先在一个终端中起服务

cd /home/user01/jiangnan/starVLA
conda activate starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}
CUDA_VISIBLE_DEVICES=2 python deployment/model_server/server_policy.py \
  --ckpt_path /mnt/data/jiangnan/ckpts/StarVLA-Calvin-D/checkpoints/steps_30000_pytorch_model.pt \
  --port 5694 \
  --use_bf16


cd /home/user01/jiangnan/starVLA
conda activate starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}
CUDA_VISIBLE_DEVICES=2 python deployment/model_server/server_policy.py \
  --ckpt_path /home/user01/jiangnan/starVLA/results/Checkpoints/starvla_rynnbrain_calvin_task_ABC_D_memory/checkpoints/steps_30000_pytorch_model.pt \
  --port 5694 \
  --use_bf16

# 再开一个终端跑测评 
cd /home/user01/jiangnan/starVLA
conda activate calvin310
export PYTHONPATH=$(pwd):${PYTHONPATH}
python examples/calvin/eval_files/eval_calvin.py \
  --args.pretrained-path /mnt/data/jiangnan/ckpts/StarVLA-Calvin-D/checkpoints/steps_30000_pytorch_model.pt \
  --args.unnorm-key franka \
  --args.host 127.0.0.1 \
  --args.port 5694 \
  --args.dataset_path /mnt/data/jiangnan/calvin/task_D_D \
  --args.calvin_config_path /mnt/data/jiangnan/calvin/calvin/calvin_models/conf \
  --args.eval_sequences_path /home/user01/jiangnan/starVLA/examples/calvin/eval_files/eval_sequences.json 
  
# 转换数据
/home/user01/miniconda3/envs/lerobot033/bin/python \
  /home/user01/jiangnan/starVLA/examples/calvin/convert_to_lerobot_starvla.py \
  --zip-path /mnt/data/jiangnan/calvin/task_ABC_D.zip \
  --output-root /mnt/data/jiangnan/lerobot \
  --repo-id task_ABC_D_lerobot \
  --splits training


cd /home/user01/jiangnan/starVLA
conda activate calvin310
export PYTHONPATH=$(pwd):${PYTHONPATH}
python examples/calvin/eval_files/eval_calvin.py \
  --args.pretrained-path /home/user01/jiangnan/starVLA/results/Checkpoints/starvla_rynnbrain_calvin_task_ABC_D_memory/checkpoints/steps_30000_pytorch_model.pt \
  --args.unnorm-key franka \
  --args.host 127.0.0.1 \
  --args.port 5694 \
  --args.dataset_path /mnt/data/jiangnan/calvin/task_D_D \
  --args.calvin_config_path /mnt/data/jiangnan/calvin/calvin/calvin_models/conf \
  --args.eval_sequences_path /home/user01/jiangnan/starVLA/examples/calvin/eval_files/eval_sequences.json \
  --args.num_sequences 1000

  # 修改
  /home/user01/jiangnan/starVLA/starVLA/dataloader/gr00t_lerobot/mixtures.py
  添加
  ```python
    "calvin_task_ABC_D": [
        ("task_ABC_D_lerobot", 1.0, "libero_franka"),
    ],
```

# 跑RynnBrain nomemory 测评 (权重在 /home/user01/jiangnan/starVLA/results/Checkpoints/starvla_rynnbrain_calvin_task_ABC_D_nomemory/final_model)

## 先在一个终端中起服务
cd /home/user01/jiangnan/starVLA
conda activate starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}
CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
  --ckpt_path /home/user01/jiangnan/starVLA/results/Checkpoints/starvla_rynnbrain_calvin_task_ABC_D_nomemory/final_model/pytorch_model.pt \
  --port 5694 \
  --use_bf16

## 再开一个终端跑测评
cd /home/user01/jiangnan/starVLA
conda activate calvin310
export PYTHONPATH=$(pwd):${PYTHONPATH}
python examples/calvin/eval_files/eval_calvin.py \
  --args.pretrained-path /home/user01/jiangnan/starVLA/results/Checkpoints/starvla_rynnbrain_calvin_task_ABC_D_nomemory/final_model/pytorch_model.pt \
  --args.unnorm-key franka \
  --args.host 127.0.0.1 \
  --args.port 5694 \
  --args.dataset_path /mnt/data/jiangnan/calvin/task_D_D \
  --args.calvin_config_path /mnt/data/jiangnan/calvin/calvin/calvin_models/conf \
  --args.eval_sequences_path /home/user01/jiangnan/starVLA/examples/calvin/eval_files/eval_sequences.json \
  --args.num_sequences 1000

# 跑RynnBrain memory 测评 (权重在 /home/user01/jiangnan/starVLA/results/Checkpoints/starvla_rynnbrain_calvin_task_ABC_D_memory_0425/final_model)

## 先在一个终端中起服务
cd /home/user01/jiangnan/starVLA
conda activate starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}
# 如果 final_model 还没生成，可临时改用 checkpoints/steps_95000_pytorch_model.pt
CUDA_VISIBLE_DEVICES=1 python deployment/model_server/server_policy.py \
  --ckpt_path /home/user01/jiangnan/starVLA/results/Checkpoints/starvla_rynnbrain_calvin_task_ABC_D_memory_0425/final_model/pytorch_model.pt \
  --port 5695 \
  --use_bf16

## 再开一个终端跑测评
cd /home/user01/jiangnan/starVLA
conda activate calvin310
export PYTHONPATH=$(pwd):${PYTHONPATH}
python examples/calvin/eval_files/eval_calvin.py \
  --args.pretrained-path /home/user01/jiangnan/starVLA/results/Checkpoints/starvla_rynnbrain_calvin_task_ABC_D_memory_0425/final_model/pytorch_model.pt \
  --args.unnorm-key franka \
  --args.host 127.0.0.1 \
  --args.port 5695 \
  --args.dataset_path /mnt/data/jiangnan/calvin/task_D_D \
  --args.calvin_config_path /mnt/data/jiangnan/calvin/calvin/calvin_models/conf \
  --args.eval_sequences_path /home/user01/jiangnan/starVLA/examples/calvin/eval_files/eval_sequences.json \
  --args.num_sequences 1000

# 跑RynnBrain circular memory 测评
cd /home/user01/jiangnan/starVLA
conda activate starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}
CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
  --ckpt_path /home/user01/jiangnan/starVLA/results/Checkpoints/starvla_rynnbrain_calvin_task_ABC_D_memory_dit_inter5_step5/final_model/pytorch_model.pt \
  --port 5695 \
  --use_bf16
  
cd /home/user01/jiangnan/starVLA
conda activate calvin310
export PYTHONPATH=$(pwd):${PYTHONPATH}
python examples/calvin/eval_files/eval_calvin.py \
  --args.pretrained-path /home/user01/jiangnan/starVLA/results/Checkpoints/starvla_rynnbrain_calvin_task_ABC_D_memory_dit_inter5_step5/final_model/pytorch_model.pt \
  --args.unnorm-key franka \
  --args.host 127.0.0.1 \
  --args.port 5695 \
  --args.dataset_path /mnt/data/jiangnan/calvin/task_D_D \
  --args.calvin_config_path /mnt/data/jiangnan/calvin/calvin/calvin_models/conf \
  --args.eval_sequences_path /home/user01/jiangnan/starVLA/examples/calvin/eval_files/eval_sequences.json \
  --args.num_sequences 1000