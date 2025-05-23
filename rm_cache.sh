# export CLOUDSDK_CORE_PROJECT="optimal-tide-388320"
TPU_NAME=$1
ZONE=us-central2-b
# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command "rm -r .cache/huggingface"



gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all \
--command "rm -rf /tmp/isabelle*; rm -rf /tmp/ray; rm -r ~/.cache; rm -r ~/ckpt_tmp; sudo rm /var/log/kern*; sudo journalctl --vacuum-size=100M; sudo truncate -s 0 /var/log/syslog; sudo systemctl restart rsyslog"

for i in {1..3}
do
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --command="singularity instance stop -a; source ~/venv310/bin/activate; ray stop; killall repl; pkill -e -f -u kaiyue python" --worker=all
done


gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command "rm -rf ~/high_level_sampler; rm -rf ~/low_level_sampler"