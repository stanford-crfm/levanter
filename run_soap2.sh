eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh debug2 -z us-central2-b -t v4-128 --preemptible -- \
WANDB_API_KEY=$WANDB_API_KEY \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_1b_adam.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/soap/lr4e-3_step10000_1b  \
--optimizer.learning_rate 4e-3 \
--optimizer.type mudam \
--trainer.wandb.name mudam_debug_reduce_to_muon_warmup1000_beta0.9_debug_inverse_precond \
--optimizer.partition_grads_into_blocks False \
--optimizer.merge_small_dims False \
--trainer.num_train_steps 10001 \
--optimizer.warmup 1000 \
--optimizer.shampoo_beta 0.9 \
--optimizer.min_lr_ratio 0.0 \
--optimizer.lr_schedule linear \
--optimizer.cooldown 0.0 \
--model.seq_len 8192 \
--model.hidden_dim 768 \
--model.intermediate_dim 3072 \
--model.num_layers 12 \
--model.num_heads 12 \
--model.num_kv_heads 12 \
--trainer.train_batch_size 128 