eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh debug -z us-central2-b -t v4-128 --preemptible -- \
WANDB_API_KEY=$WANDB_API_KEY \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama3_small_4k.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/soap/lr4e-3_step10000_1b  \
--optimizer.learning_rate 4e-3 \
--optimizer.type adamw \
--trainer.wandb.name adamw_4k \
--trainer.num_train_steps 10001 \
--optimizer.warmup 1000 \
--optimizer.min_lr_ratio 0.0 \
--optimizer.lr_schedule linear \
--optimizer.learning_rate 0.008