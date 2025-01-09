eval $(ssh-agent -s)
bash infra/babysit-tpu-vm-ema.sh test-ema -z us-central2-b -t v4-128 --preemptible -- \
WANDB_API_KEY=1c85c63399be786e59026e288175122f49a434b0 \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_100M_constant_lr2e-3.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/test_ema  \
--optimizer.min_lr_ratio 0.0 \
--trainer.use_ema True
