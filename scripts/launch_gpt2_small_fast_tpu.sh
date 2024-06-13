# Launches the "gpt_small_fast" model on a TPU node

python infra/launch.py --tpu_name levanter-itest-32 --preemptible --zone us-east1-d --tpu_type v3-32 --foreground -- \
    python src/levanter/main/train_lm.py \
    --config_path config/gpt2_small_fast.yaml \
    --trainer.checkpointer.base_path gs://levanter-checkpoints/gpt-itest/ --trainer.checkpointer.save_interval 30m $*
