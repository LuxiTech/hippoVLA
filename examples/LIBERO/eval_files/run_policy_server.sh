#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/home/user01/miniconda3/envs/starVLA/bin/python
your_ckpt=results/Checkpoints/0420_libero4in1_RynnBrain8OFT_memory/checkpoints/steps_60000_pytorch_model.pt
gpu_id=3
port=5696
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
