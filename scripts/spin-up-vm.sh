#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

. "${SCRIPT_DIR}"/helpers/parse-tpu-creation-args.sh "$@"

# error out if we didn't set a name
if [ -z "$VM_NAME" ]; then
  echo "Error: VM name not set"
  exit 1
fi

# first delete if we're supposed to
if [ "$AUTODELETE" = "true" ]; then
  # check if it's there
  gcloud compute tpus tpu-vm describe --zone $ZONE $VM_NAME &> /dev/null
  if [ $? -eq 0 ]; then
    echo "Deleting existing VM $VM_NAME"
    gcloud compute tpus tpu-vm delete --zone $ZONE $VM_NAME
  fi
fi



# create the vm
# spin loop until we get a good error code
echo "Creating VM $VM_NAME"
# create the command. note that --preemptible doesn't accept a value, so just append it if we want it
CMD="gcloud alpha compute tpus tpu-vm create $VM_NAME \
  --zone=$ZONE \
  --accelerator-type=$TYPE \
  --version=$VM_IMAGE"
if [ "$PREEMPTIBLE" = true ]; then
  CMD="$CMD --preemptible"
fi
echo "Running command: $CMD"

while ! $CMD; do
  echo "Error creating VM, retrying in 5 seconds"
  sleep 5
done

echo "Giving the VM a few seconds to start up"
sleep 15

echo "Adding ssh keys just in case..."
echo ssh-add ~/.ssh/google_compute_engine
ssh-add ~/.ssh/google_compute_engine

# upload the setup-tpu-vm.sh script
# note that gcloud scp doesn't always work... so we do it a few times to just be sure
for i in {1..5}; do
  echo "Uploading setup-tpu-vm.sh to VM $VM_NAME"
  gcloud compute tpus tpu-vm scp --zone=$ZONE $SCRIPT_DIR/setup-tpu-vm.sh $VM_NAME:~/ --worker=all
  # check to see if the file exists on all nodes
  if gcloud compute tpus tpu-vm ssh --zone=$ZONE $VM_NAME --command="ls ~/setup-tpu-vm.sh" --worker=all; then
    break
  fi
  echo "Error uploading setup-tpu-vm.sh, retrying in 5 seconds"
  sleep 5
  if 5 == $i; then
    echo "Error uploading setup-tpu-vm.sh, giving up. Note that the machine is still (probably) running"
    exit 1
  fi
done

# run the setup script
gcloud compute tpus tpu-vm ssh --zone=$ZONE $VM_NAME --command="bash ~/setup-tpu-vm.sh" --worker=all

# print out the IP addresses
echo "VM $VM_NAME IP addresses:"
gcloud compute tpus tpu-vm describe --zone $ZONE $VM_NAME | awk '/externalIp: (.*)/ {print $2}'
