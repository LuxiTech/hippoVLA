cd /home/user01/jiangnan/starVLA
conda activate starVLA
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python examples/calvin/eval_files/eval_calvin.py \
  --args.pretrained-path /mnt/data/jiangnan/ckpts/StarVLA-Calvin-D/checkpoints/steps_30000_pytorch_model.pt \
  --args.unnorm-key franka \
  --args.host 127.0.0.1 \
  --args.port 5694 \
  --args.dataset_path /mnt/data/jiangnan/calvin/task_D_D \
  --args.calvin_config_path /mnt/data/jiangnan/calvin/calvin/calvin_models/conf \
  --args.eval_sequences_path /home/user01/jiangnan/starVLA/examples/calvin/eval_files/eval_sequences.json \
  --args.num_sequences 1000