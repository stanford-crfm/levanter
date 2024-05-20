#!/bin/bash

# Define the TPU, zone, project, and worker variables
TPU_NAME="trace"
ZONE="us-central2-b"
PROJECT="hai-gcp-models"
WORKER="all"
# Define the training command with environment variables
CMD_TRAIN="WANDB_API_KEY=\$WANDB_API_KEY \
    levanter/infra/run.sh python levanter/src/levanter/main/train_lm.py \
    --config_path levanter/config/olmo_7b_contd.yaml \
    --trainer.checkpointer.base_path gs://levanter-checkpoints/olmo_7b_trace_lev/"

# Run the training command
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --project=$PROJECT --worker=$WORKER --command "$CMD_TRAIN"
