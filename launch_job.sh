EXP_NAME="ivan-ray-multislice"
TPU_TYPE="v4-8"
NODE_COUNT=2
TPU_ZONE="us-central2-b"

TPU_NAME="$EXP_NAME-$TPU_TYPE-$NODE_COUNT-slice"
RUN_ID=$TPU_NAME
RETRIES=1

CMD="ray start --head --port=6379 --num-cpus=0 && python ray_scripts/ray_tpu_task.py"
CMD="echo 'hello world'"
echo $CMD
python infra/launch.py --retries=$RETRIES --foreground \
    --tpu_name $TPU_NAME --tpu_type $TPU_TYPE --zone $TPU_ZONE --node_count $NODE_COUNT \
    --run_id $RUN_ID -- $CMD
