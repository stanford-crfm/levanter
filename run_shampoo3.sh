eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh debug3 -z europe-west4-b -t v5litepod-128 --preemptible -- \
WANDB_API_KEY=$WANDB_API_KEY \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_1b_adam.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/debug_mudam/shampoo_implicit  \
--optimizer.type shampoo3 \
--optimizer.merge_small_dims True \
--trainer.wandb.name shampoo_implicit_lr2e-1_small_eps \
--trainer.num_train_steps 10001 \
--optimizer.warmup 1000 \
--optimizer.momentum 0.9 \
--optimizer.shampoo_beta 0.98 \
--optimizer.grafting False \
--optimizer.learning_rate 2e-1 \
--optimizer.adam_lr 2e-3 \
--optimizer.epsilon 1e-15 \
--optimizer.min_lr_ratio 0.0 \
--optimizer.lr_schedule linear \
--optimizer.cooldown 0.0 \
--model.seq_len 8192 \
--model.hidden_dim 768 \
--model.intermediate_dim 3072 \
--model.num_layers 12 \
--model.num_heads 12 \
--model.num_kv_heads 12 \
--trainer.train_batch_size 128 \
--optimizer.weight_decay 0.001