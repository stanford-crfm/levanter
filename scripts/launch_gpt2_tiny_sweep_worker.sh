# Launches the "gpt_small_fast" model on a TPU node

if [ -z "$WANDB_API_KEY" ]; then
  echo "Error: WANDB_API_KEY not set"
  exit 1
fi

if [ -z "$WANDB_SWEEP_ID" ]; then
  echo "Error: WANDB_SWEEP_ID not set"
  exit 1
fi


if [ -z "$GIT_BRANCH" ]; then
  GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
fi

# generate a random name for the instance with prefix
PREFIX="wandb-sweep-"
# need LC_ALL=C to make tr work on mac
SUFFIX=$(cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | fold -w 4 | head -n 1)

INSTANCE_NAME="$PREFIX$SUFFIX"

echo "Launching sweep agent on TPU with git branch $GIT_BRANCH"

bash infra/babysit-tpu-vm.sh $INSTANCE_NAME -p -z us-central1-f -t v2-8 -b $GIT_BRANCH -- \
    WANDB_API_KEY=$WANDB_API_KEY WANDB_SWEEP_ID=$WANDB_SWEEP_ID WANDB_PROJECT=ezekiel \
    levanter/infra/run.sh python levanter/scripts/wandb_sweep_agent.py \
    --config_path levanter/config/gpt2_tiny_base.yaml \
    $*
