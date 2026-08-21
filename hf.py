from huggingface_hub import snapshot_download

# snapshot_download(repo_id="HuggingFaceFW/fineweb_100BT",
#                   repo_type="dataset",local_dir="/mnt/data/jiangnan/dataset")
# snapshot_download(repo_id="IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
#                   repo_type="dataset",local_dir="./playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot")
# snapshot_download(repo_id="IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
#                   repo_type="dataset",local_dir="./playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot")
# snapshot_download(repo_id="IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
#                   repo_type="dataset",local_dir="./playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot")
# snapshot_download(repo_id="Qwen/Qwen3-VL-4B-Instruct",
#                   local_dir="./playground/Pretrained_models/Qwen3-VL-4B-Instruct")
snapshot_download(
    repo_id="Alibaba-DAMO-Academy/RynnBrain-CoP-8B",
    repo_type="model",
    local_dir="/root/autodl-tmp/hybridVLA/RynnBrain-CoP-8B",
    token=True,
    max_workers=2,
)
hf download Alibaba-DAMO-Academy/RynnBrain-CoP-8B \
  --repo-type model \
  --local-dir ./playground/Pretrained_models/RynnBrain-CoP-8B \
  --local-dir-use-symlinks False

# snapshot_download(repo_id="ovoovo/Qwen2.5-3b-car",
#                   repo_type="model",local_dir="./playground/Pretrained_models/Qwen2.5-3b-car")