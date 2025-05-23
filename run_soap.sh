eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh soap -z europe-west4-b -t v5litepod-256 --preemptible -- \
WANDB_API_KEY=$WANDB_API_KEY \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_1b_adam.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/soap/lr4e-3_step10000_1b  \
--optimizer.learning_rate 4e-3 \
--optimizer.type soap \
--trainer.wandb.name soap_block_256_precond_1_10k_1b \
--optimizer.partition_grads_into_blocks True \
--optimizer.precondition_frequency 1 \
--trainer.num_train_steps 10001 \
--optimizer.min_lr_ratio 0.0 \
--optimizer.lr_schedule linear \
--optimizer.cooldown 0.0 \
--model.seq_len 4096 \
--model.hidden_dim 1536 \
--model.intermediate_dim 6144 \
--model.num_layers 32 \
--model.num_heads 24 \
--model.num_kv_heads 24 \
--trainer.train_batch_size 512