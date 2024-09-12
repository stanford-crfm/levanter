python infra/launch.py --foreground -e WANDB_MODE offline --tpu_name levanter-itest-32-$(whoami) --zone us-central2-b --tpu_type v4-32 --preemptible -- \
    python -m levanter.main.train_lm \
    --config_path config/gpt2_small_fast.yaml \
    --trainer.steps_per_eval 10 \
    --trainer.checkpointer.save_interval 12h $*
