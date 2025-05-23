eval $(ssh-agent -s)
bash infra/spin-up-vm.sh debug -z europe-west4-b -t v5litepod-128 --preemptible

gcloud compute tpus tpu-vm ssh debug --zone europe-west4-b --command="cd levanter && git pull" --worker=all
gcloud compute tpus tpu-vm scp --zone europe-west4-b --worker=all config/llama2_nano.yaml debug:levanter/config/llama2_nano.yaml
gcloud compute tpus tpu-vm scp --zone europe-west4-b --worker=all src/levanter/models/llama.py debug:levanter/src/levanter/models/llama.py


bash infra/babysit-tpu-vm.sh debug -z europe-west4-b -t v5litepod-128 --preemptible -- \
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
