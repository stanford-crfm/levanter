#!/bin/bash

# TRC recently made a change where they unceremoniously kill TPU VMs whenever they need capacity for paying customers
# Understandable, but we have to work around it.
# This script runs on a non-TPU VM (some server somewhere) and periodically relaunches the TPU VM if it's not running
# and restarts the process
# My preference would be to use pdsh for this, but we don't reliably have it on our internal cluster...

# Syntax: babysit-tpu-vm.sh <args to spin-up-vm.sh> -- command to run on the vm

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# first extract the args for just spin-up-vm.sh and pass them to the helper

CREATION_ARGS=()

# just take until we get the --
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --)
      shift
      break
      ;;
    *)
      CREATION_ARGS+=("$1")
      shift
      ;;
  esac
done

# set the cmd args:
CMD_ARGS="$*"


# now source the helper
source $SCRIPT_DIR/helpers/parse-tpu-creation-args.sh "${CREATION_ARGS[@]}"

# error out if we didn't set a name
if [ -z "$VM_NAME" ]; then
  echo "Error: VM name not set"
  exit 1
fi

# check if the VM is running
# if not, spin it up
# if it is, just run the command
while true; do
  # check if it's there
  gcloud compute tpus tpu-vm describe --zone $ZONE $VM_NAME &> /dev/null
  if [ $? -eq 0 ]; then
    # check if it's running
    STATE=$(gcloud compute tpus tpu-vm describe --zone $ZONE $VM_NAME | grep state | awk '{print $2}')
    if [ "$STATE" != "READY" ]; then
      echo "VM $VM_NAME is not in READY state, state is $STATE"
      echo "Deleting VM $VM_NAME"
      yes | gcloud compute tpus tpu-vm delete --zone $ZONE $VM_NAME
    else
      # run the command
      echo "Running command on VM $VM_NAME"
      echo "Running command: $CMD_ARGS"
      echo gcloud compute tpus tpu-vm ssh --zone=$ZONE $VM_NAME --command=\"$CMD_ARGS\" --worker=all
      gcloud compute tpus tpu-vm ssh --zone=$ZONE $VM_NAME --command="$CMD_ARGS" --worker=all
      if [ $? -eq 0 ]; then
        echo "Command succeeded. Exiting"
        exit 0
      else
        echo "Command failed"
      fi
    fi
  else
    echo "VM $VM_NAME not found, creating it"
    bash "$SCRIPT_DIR"/spin-up-vm.sh "${CREATION_ARGS[@]}"
  fi
  echo "Sleeping for 1 minute"
  sleep 60
done
