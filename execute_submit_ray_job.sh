ray job submit --address http://127.0.0.1:8265 --working-dir . \
    --runtime-env ray_runtime_env.yaml \
    --entrypoint-num-cpus 7168 \
    --no-wait \
    -- python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml
