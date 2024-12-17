eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh muon-debug -z us-central2-b -t v4-128 --preemptible -- \
WANDB_API_KEY=[WANDB_API_KEY] \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_100M_muon.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/muon/llama2_100M_constant  \
--optimizer.type muon \
--trainer.num_train_steps 10000 \
--trainer.load_checkpoint_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/muon/llama2_100M_constant/tjo9vxfb/step-4000
