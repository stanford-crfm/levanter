#!/bin/bash

# Define the TPU, zone, project, and worker variables
TPU_NAME="trace"
ZONE="us-central2-b"
PROJECT="hai-gcp-models"
WORKER="all"

# Define the setup command
CMD_SETUP="source venv310/bin/activate && pip install ai2-olmo"

# Run the setup command
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --project=$PROJECT --worker=$WORKER --command "$CMD_SETUP"
