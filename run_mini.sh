eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh debug -z us-central2-b -t v4-32 --preemptible -- \
WANDB_API_KEY=$WANDB_API_KEY \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_nano.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/debug/mini_debug  \
--trainer.wandb.name nano_mini_debug \
--trainer.num_train_steps 10001 \
--optimizer.min_lr_ratio 0.0 \
--optimizer.lr_schedule linear \
--optimizer.type mini \
--optimizer.cooldown 0.0 \
--optimizer.max_grad_norm 0.0