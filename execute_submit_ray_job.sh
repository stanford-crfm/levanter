COMMAND="pip install git+https://github.com/stanford-crfm/levanter.git@ivan-ray-jobs fire && WANDB_API_KEY=be441272f3bd2812a2eb009739e26a202f14d7ba \
    WANDB_PROJECT=marin \
    python src/levanter/main/train_lm_ray.py --config_path config/gpt2_nano.yaml"

echo $COMMAND
ray job submit --address http://127.0.0.1:8265 --working-dir . \
    --runtime-env ray_runtime_env.yaml \
    --entrypoint-num-cpus 448 \
    --no-wait \
    -- $COMMAND
