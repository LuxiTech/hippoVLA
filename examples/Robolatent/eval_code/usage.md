# SERVER
python deployment/model_server/server_policy.py \
  --ckpt_path /data/share-folder/linsixu_deploy/robolatent-pickx/steps_6000_pytorch_model.pt \
  --base_vlm_path /data/share-folder/linsixu_deploy/hippovla-starVLA-jiangnan-real/playground/Pretrained_models/RynnBrain-CoP-8B \
  --port 5678 \
  --use_bf16 \
  --idle_timeout -1


# CLIENT
python -m examples.new_deploy.pi_infer   --host 127.0.0.1   --port 5678   --task PickXtimes   --dataset_statistics /home/agilex/Sixu/dataset_statistics.json   --action_horizon 4   --max_memory_frames 5   --memory_interval 10   --verify_reset   --reset_tolerance 0.03   --reset_verify_timeout 8   --reset_qpos=-0.020427,0.006367,-0.000820,0.031277,0.340995,-0.095401,0.000200,-0.033144,0.001989,-0.000645,0.187453,0.000000,-0.378064,0.066700   --save_server_inputs   --server_input_dump_limit 80   --clip_gripper   --gripper_binary_threshold 0.5   --left_wrist_brightness_offset 15   --log_every_n_steps 1   --save_dir runs/robolatent_eval_online_real_wrist_bright