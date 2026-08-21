# 用12.3CUDA编译
export CUDA_HOME=/usr/local/cuda-12.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
hash -r

which nvcc
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"

cd /mnt/data/jiangnan/RoboTwin/envs/curobo
pip install -e . --no-build-isolation

=======================================================================

cd /home/user01/jiangnan/starVLA
conda activate starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

CUDA_VISIBLE_DEVICES=6 python deployment/model_server/server_policy.py \
  --ckpt_path /home/user01/.cache/huggingface/hub/models--StarVLA--Qwen3-VL-OFT-RoboTwin2-All/snapshots/727645249e6bfbff6870db4637e4b9d32e6be346/checkpoints/steps_140000_pytorch_model.pt \
  --port 5695 \
  --use_bf16



conda activate RoboTwin
cd /home/user01/jiangnan/starVLA/examples/Robotwin/eval_files
bash eval.sh grab_roller demo_clean my_test_v1 0 6

# 基础操作任务
bash eval.sh beat_block_hammer demo_clean my_test_v1 0 0
bash eval.sh pick_dual_bottles demo_clean my_test_v1 0 0
bash eval.sh lift_pot demo_clean my_test_v1 0 0
bash eval.sh grab_roller demo_clean my_test_v1 0 0

# 物品放置任务
bash eval.sh place_object_basket demo_clean my_test_v1 0 0
bash eval.sh place_empty_cup demo_clean my_test_v1 0 0
bash eval.sh place_shoe demo_clean my_test_v1 0 0
bash eval.sh place_dual_shoes demo_clean my_test_v1 0 0

# 交互操作任务
bash eval.sh handover_block demo_clean my_test_v1 0 0
bash eval.sh stack_blocks_two demo_clean my_test_v1 0 0
bash eval.sh click_bell demo_clean my_test_v1 0 0
bash eval.sh press_stapler demo_clean my_test_v1 0 0

# 工具使用任务 
bash eval.sh open_laptop demo_clean my_test_v1 0 0
bash eval.sh open_microwave demo_clean my_test_v1 0 0
bash eval.sh turn_switch demo_clean my_test_v1 0 0
bash eval.sh scan_object demo_clean my_test_v1 0 0

# 复杂序列任务
bash eval.sh dump_bin_bigbin demo_clean my_test_v1 0 0
bash eval.sh hanging_mug demo_clean my_test_v1 0 0
bash eval.sh put_object_cabinet demo_clean my_test_v1 0 0

# 排序整理任务 
bash eval.sh blocks_ranking_size demo_clean my_test_v1 0 0
bash eval.sh place_bread_basket demo_clean my_test_v1 0 0


# 报错：
怀疑是显卡架构与pytorch版本冲突 cuda编译时出错


error occurs: CUDA error: an illegal instruction was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

error occurs: CUDA error: an illegal instruction was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

error occurs: CUDA error: an illegal instruction was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

error occurs: CUDA error: an illegal instruction was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

error occurs: CUDA error: an illegal instruction was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.