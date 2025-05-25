eval $(ssh-agent -s)
OPTIMIZER_TYPE=soap
# bash infra/spin-up-vm.sh debug4 -z us-central2-b -t v4-128 --preemptible
gcloud compute tpus tpu-vm ssh debug4 --zone us-central2-b --command="cd levanter && git pull" --worker=all
gcloud compute tpus tpu-vm ssh debug4 --zone us-central2-b --command="source venv*/bin/activate && pip install -U "jax[tpu]==0.4.38" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html" --worker=all
gcloud compute tpus tpu-vm ssh debug4 --zone us-central2-b --command="source venv*/bin/activate && pip install -U jaxlib==0.4.38" --worker=all
gcloud compute tpus tpu-vm scp --zone us-central2-b --worker=all config/llama2_nano.yaml debug4:levanter/config/llama2_nano.yaml
gcloud compute tpus tpu-vm scp --zone us-central2-b --worker=all src/levanter/models/llama.py debug4:levanter/src/levanter/models/llama.py
gcloud compute tpus tpu-vm scp --zone us-central2-b --worker=all src/levanter/optim/soap.py debug4:levanter/src/levanter/optim/soap.py


gcloud compute tpus tpu-vm ssh debug4 --zone us-central2-b --command="WANDB_API_KEY=$WANDB_API_KEY \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_nano.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/debug4/${OPTIMIZER_TYPE}_debug4  \
--trainer.wandb.name nano_${OPTIMIZER_TYPE}_debug4 \
--trainer.num_train_steps 501 \
--trainer.train_batch_size 128 \
--optimizer.min_lr_ratio 0.0 \
--optimizer.lr_schedule linear \
--optimizer.type ${OPTIMIZER_TYPE} \
--optimizer.cooldown 0.0 \
--optimizer.max_grad_norm 0.0 \
--optimizer.partition_grads_into_blocks False \
--optimizer.merge_small_dims True" --worker=all
