# Launches the "gpt_small_fast" model on a TPU node

if [ -z "$WANDB_API_KEY" ]; then
  echo "Error: WANDB_API_KEY not set"
  exit 1
fi

if [ -z "$GIT_BRANCH" ]; then
  GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
fi

echo "Launching GPT2 small fast on TPU with git branch $GIT_BRANCH"

bash infra/babysit-tpu-vm.sh levanter-itest-32 -p -z us-east1-d -t v3-32 -b $GIT_BRANCH -- \
    XLA_FLAGS="--xla_dump_to=/tmp/output_folder/xla_dumps --xla_dump_hlo_pass_re=.*" \
    WANDB_API_KEY=$WANDB_API_KEY levanter/infra/run.sh python levanter/src/levanter/main/train_lm.py \
    --config_path levanter/config/gpt2_small_fast.yaml \
    --trainer.checkpointer.base_path gs://levanter-checkpoints/gpt-itest/ --trainer.checkpointer.save_interval 30m $*
