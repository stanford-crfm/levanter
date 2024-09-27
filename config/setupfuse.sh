export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list


curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc

sudo apt-get update

sudo apt-get install gcsfuse

mkdir /home/ahmed/trace_models/

gcsfuse gs://levanter-data/trace/trace_models/ /home/ahmed/trace_models/