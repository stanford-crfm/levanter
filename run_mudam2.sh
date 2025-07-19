eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh debug -z europe-west4-b -t v5litepod-64 --preemptible -- \
WANDB_API_KEY=$WANDB_API_KEY \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_1b_adam.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/debug_mudam/mudam_reduced_to_muon_with_explicit_inverse_sqrt  \
--optimizer.type mudam2 \
--optimizer.merge_small_dims True \
--trainer.wandb.name mudam_reduced_to_muon_with_explicit_inverse_sqrt \
--trainer.num_train_steps 10001 \
--optimizer.warmup 0 \
--optimizer.shampoo_beta 0.0 \
--optimizer.learning_rate 8e-3 \
--optimizer.adam_lr 2e-3 \
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