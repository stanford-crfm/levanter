EXP_NAME="ivan-ray-multislice"
TPU_TYPE="v4-128"
NODE_COUNT=2
TPU_ZONE="us-central2-b"

DATE=$(TZ='America/Los_Angeles' date +%m%d)
UNIQUE_ID=$(uuidgen | md5sum | head -c 8)
TPU_NAME="$EXP_NAME-$UNIQUE_ID-$TPU_ZONE-$TPU_TYPE"
RUN_ID="$EXP_NAME-$UNIQUE_ID-$TPU_ZONE-$TPU_TYPE-$DATE"
RETRIES=100

CMD="python ray_scripts/ray_tpu_task.py"
echo $CMD
python infra/launch.py --retries=$RETRIES --foreground \
    --tpu_name $TPU_NAME --tpu_type $TPU_TYPE --zone $TPU_ZONE --node_count $NODE_COUNT \
    --run_id $RUN_ID -- $CMD
