CONFIG_PATH="config/llama_1b.yaml"
EXP_NAME="1b-llama-tok"
TPU_TYPE="v5litepod-256"
TPU_ZONE="us-west4-a"
GS_BUCKET="gs://marin-ckpt-us-w4"  # gs://marin-ckpt-us-w4, gs://marin-ckpt-us-c2

DATE=$(TZ='America/Los_Angeles' date +%m%d)
UNIQUE_ID=$(uuidgen | md5sum | head -c 8)
TPU_NAME="ivan-preempt-$EXP_NAME-$UNIQUE_ID-$TPU_ZONE-$TPU_TYPE"
WANDB_RUN_ID="ttt-$EXP_NAME-$TPU_ZONE-$TPU_TYPE-$UNIQUE_ID-$DATE"
WANDB_RUN_NAME="time-to-train_$EXP_NAME-$TPU_ZONE-$TPU_TYPE-$UNIQUE_ID-$DATE"
RETRIES=10000

# Log launch timestamp
DATETIME=$(TZ='America/Los_Angeles' date "+%Y-%m-%d %I:%M:%S %p %Z")
echo "$WANDB_RUN_NAME,$DATETIME" >> launch.log

CMD="python src/levanter/main/train_lm.py \
    --config_path $CONFIG_PATH \
    --trainer.wandb.resume true \
    --trainer.wandb.id $WANDB_RUN_ID \
    --trainer.wandb.name $WANDB_RUN_NAME \
    --trainer.load_checkpoint_path $GS_BUCKET/ivan-ttt/$WANDB_RUN_NAME/$WANDB_RUN_ID/  \
    --trainer.checkpointer.base_path $GS_BUCKET/ivan-ttt/$WANDB_RUN_NAME \
    --trainer.checkpointer.save_interval 600m"
echo $CMD
python infra/launch.py --retries=$RETRIES --foreground --tpu_name $TPU_NAME --tpu_type $TPU_TYPE --zone $TPU_ZONE --run_id $WANDB_RUN_ID \
    --env LIBTPU_INIT_ARGS "--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true" \
    -- $CMD
