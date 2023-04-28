# Launches the "gpt_small_fast" model on a TPU node

bash infra/babysit-tpu-vm.sh levanter-itest-32 -z us-east1-d --preemptible -t v3-32 -- \
    WANDB_API_KEY=$WANDB_API_KEY levanter/infra/run.sh python levanter/examples/gpt2_example.py \
    --config_path levanter/config/gpt2_small_fast.yaml \
    --trainer.checkpointer.base_path gs://levanter-checkpoints/gpt-itest/ --trainer.checkpointer.save_interval 30m $*
